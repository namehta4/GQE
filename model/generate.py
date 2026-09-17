#!/usr/bin/env python3
"""
Custom autoregressive generation loop for HamiltonianConditionedLM.

Standard HF `model.generate()` cannot be used here: whenever the model
samples the <ham> token, that position's embedding for the NEXT step must be
replaced with the Hamiltonian encoder's output rather than the token's own
learned embedding, and HF's generation utilities have no hook for
substituting a specific position's embedding once a token id has been
sampled (see the NotImplementedError in gemma_hamiltonian_model.py). This
module implements that loop directly: manual KV-cache management, top-p
sampling, and <ham>-embedding substitution at every step.

All candidates for one batch start from the identical <bos> prompt (the
Hamiltonian is injected via embedding substitution, not prompt text), so
every sequence in a generation batch has the same length at every step --
this lets us skip left/right padding entirely during generation and only
truncate each sequence at its own <eos> afterward.

VALIDATION WARNING: not exercised against a live `transformers`/CUDA-Q
install in this environment. The main risk is version-specific behavior of
HF's `past_key_values` / `use_cache` API when driving a model purely via
`inputs_embeds` (no `input_ids`) across incremental steps -- some model
implementations expect `cache_position` or `position_ids` to be passed
explicitly once `past_key_values` is non-empty. If generation produces
repeated/garbage tokens after the first few steps, check that first.

Usage:
  python generate.py \\
      --checkpoint checkpoints_12q_eps5/best_checkpoint.pt \\
      --tokenizer-dir training_examples_12q_eps5 \\
      --test-jsonl training_examples_12q_eps5/test.jsonl \\
      --num-candidates 16 --max-new-tokens 600 --top-p 0.95 \\
      --output generated_12q_eps5.jsonl
"""
import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from gemma_hamiltonian_model import HamiltonianConditionedLM, build_gemma3_backbone  # noqa: E402
from hamiltonian_encoder import HamiltonianEncoder  # noqa: E402
from operator_tokenizer import OperatorTokenizer  # noqa: E402


def load_model_from_checkpoint(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    a = ckpt["args"]

    backbone = build_gemma3_backbone(
        vocab_size=ckpt["vocab_size"],
        hidden_size=a["hidden_size"],
        num_layers=a["num_layers"],
        num_heads=a["num_heads"],
        num_kv_heads=a["num_kv_heads"],
        context_length=a["context_length"],
        sliding_window=a["sliding_window"],
        rope_theta=a["rope_theta"],
    )
    encoder = HamiltonianEncoder(
        input_dim=ckpt["hamiltonian_dim"],
        hidden_dim=a["encoder_hidden_dim"],
        output_dim=a["hidden_size"],
        depth=a["encoder_depth"],
        ffn_mult=a["encoder_ffn_mult"],
        dropout=0.0,  # no dropout at inference
    )
    model = HamiltonianConditionedLM(backbone, encoder, ckpt["ham_token_id"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def sample_top_p(logits, top_p):
    """Nucleus sampling: keep the smallest set of tokens whose cumulative
    probability mass is >= top_p (always keeping at least the top token),
    renormalize, and sample. top_p=1.0 degenerates to plain multinomial
    sampling from the full softmax."""
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    remove_mask = (cumulative - sorted_probs) > top_p
    sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    sampled_sorted_idx = torch.multinomial(sorted_probs, 1).squeeze(-1)
    return sorted_idx.gather(1, sampled_sorted_idx.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def generate(
    model,
    tokenizer,
    hamiltonian,
    num_candidates=16,
    max_new_tokens=600,
    temperature=1.0,
    top_p=0.95,
    device="cpu",
):
    """hamiltonian: (n_conformers, d) or (d,) float tensor of ALREADY-NORMALIZED
    Hamiltonian coefficients (same normalization as training -- see
    hamiltonian_encoder.HamiltonianNormalizer). Returns a list (one entry per
    conformer) of lists (one entry per candidate) of raw generated token-id
    sequences, truncated at (and including) each sequence's first <eos>."""
    model.eval()
    backbone = model.backbone
    encoder = model.hamiltonian_encoder
    ham_token_id = model.ham_token_id
    bos_id, eos_id, pad_id = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id

    if hamiltonian.dim() == 1:
        hamiltonian = hamiltonian.unsqueeze(0)
    hamiltonian = hamiltonian.to(device)
    n_conformers = hamiltonian.shape[0]

    ham_embed_per_conformer = encoder(hamiltonian)  # (n_conformers, hidden), one encoder call
    ham_embed_batch = ham_embed_per_conformer.repeat_interleave(num_candidates, dim=0)
    batch_size = n_conformers * num_candidates

    embed_layer = backbone.get_input_embeddings()

    current_ids = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    generated = [[bos_id] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    inputs_embeds = embed_layer(current_ids)
    attention_mask = torch.ones((batch_size, 1), dtype=torch.long, device=device)
    past_key_values = None

    for _ in range(max_new_tokens):
        outputs = backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :] / max(temperature, 1e-6)

        next_ids = sample_top_p(logits, top_p)
        next_ids = torch.where(finished, torch.full_like(next_ids, pad_id), next_ids)

        for i in range(batch_size):
            if not finished[i]:
                generated[i].append(int(next_ids[i].item()))

        finished = finished | (next_ids == eos_id)

        # The one non-standard step: substitute the Hamiltonian embedding for
        # any newly-sampled <ham> token before it becomes next step's input.
        next_embeds = embed_layer(next_ids).unsqueeze(1)
        ham_positions = next_ids == ham_token_id
        if ham_positions.any():
            next_embeds[ham_positions, 0, :] = ham_embed_batch[ham_positions]

        inputs_embeds = next_embeds
        attention_mask = torch.cat(
            [attention_mask, torch.ones((batch_size, 1), dtype=torch.long, device=device)], dim=1
        )

        if finished.all():
            break

    truncated = []
    for ids in generated:
        if eos_id in ids:
            cut = ids.index(eos_id)
            ids = ids[: cut + 1]
        truncated.append(ids)

    return [truncated[c * num_candidates : (c + 1) * num_candidates] for c in range(n_conformers)]


_OP_PATTERN = re.compile(r"<op(\d+)>([^<]*)")


def parse_generated_sequence(token_ids, tokenizer):
    """Decode token ids into a list of (pool_index, coefficient) pairs,
    silently dropping malformed operator-coefficient pairs (e.g. a
    coefficient with two decimal points) -- mirrors the paper's footnote
    that invalid parts of generated sequences are discarded rather than
    causing a hard failure (Sec. 3.2.3, footnote 2)."""
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    operator_sequence = []
    n_invalid = 0
    for match in _OP_PATTERN.finditer(text):
        try:
            idx = int(match.group(1))
            coeff = float(match.group(2))
        except ValueError:
            n_invalid += 1
            continue
        operator_sequence.append((idx, coeff))
    return operator_sequence, n_invalid


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument(
        "--test-jsonl",
        required=True,
        help="test.jsonl (or val/train) from build_training_examples.py -- "
        "its 'hamiltonian' field is already normalized consistently with "
        "training, so no separate scale file is needed here",
    )
    p.add_argument("--num-candidates", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=600)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument(
        "--batch-conformers",
        type=int,
        default=1,
        help="Conformers processed per generation call (actual batch size = "
        "this * --num-candidates); increase if GPU memory allows",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = p.parse_args()

    tokenizer = OperatorTokenizer.from_pretrained(args.tokenizer_dir)
    model = load_model_from_checkpoint(args.checkpoint, args.device)

    rows = []
    with open(args.test_jsonl) as f:
        for line in f:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    n_total_invalid = 0
    with open(args.output, "w") as out_f:
        for start in range(0, len(rows), args.batch_conformers):
            chunk = rows[start : start + args.batch_conformers]
            print(f"[{start + len(chunk)}/{len(rows)}] generating...")

            ham_batch = torch.tensor([r["hamiltonian"] for r in chunk], dtype=torch.float32)
            per_conformer_sequences = generate(
                model,
                tokenizer,
                ham_batch,
                num_candidates=args.num_candidates,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=args.device,
            )

            for row, candidate_token_ids in zip(chunk, per_conformer_sequences):
                candidates = []
                for token_ids in candidate_token_ids:
                    operator_sequence, n_invalid = parse_generated_sequence(token_ids, tokenizer)
                    n_total_invalid += n_invalid
                    candidates.append(
                        {"operator_sequence": operator_sequence, "n_invalid_pairs": n_invalid}
                    )
                out_f.write(
                    json.dumps(
                        {
                            "conformer_id": row["conformer_id"],
                            "e_ref": row["e_ref"],
                            "e_hf": row["e_hf"],
                            "n_operators_reference": row["n_operators"],
                            "candidates": candidates,
                        }
                    )
                    + "\n"
                )
                out_f.flush()

    print(f"Wrote {args.output}. Total malformed operator-coefficient pairs discarded: {n_total_invalid}")
    print(
        "NEXT STEP: evaluate each candidate's energy (apply operator_sequence "
        "to the Hartree-Fock state via cudaq/cudaq_solvers statevector "
        "simulation) to pick the best-of-N per conformer -- not yet wired up "
        "in this script."
    )


if __name__ == "__main__":
    main()
