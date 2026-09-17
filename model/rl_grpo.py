#!/usr/bin/env python3
"""
GRPO/DAPO reinforcement-learning post-training for the trained-from-scratch
Gemma model, reproducing the "Reinforcement learning" paragraph of Section
3.2.3 in ADAPT-GQE (arXiv:2607.22468): the model is optimized directly on
circuit energy (Eq. 1 reward) rather than token-level agreement with
ADAPT-VQE, using Group Relative Policy Optimization with the DAPO loss
variant (token-level loss normalization to remove length bias, "clip-higher"
asymmetric clipping) plus a KL penalty against the pretrained reference
policy to keep updates conservative.

Depends on every earlier stage being wired up: model/generate.py for
rollout sampling, chemistry/circuit_energy.py for the reward (which itself
depends on test_circuit_energy.py having been validated -- do not run this
at scale before that passes), and a pretrained checkpoint from
model/pretrain.py to initialize from.

This script also logs every rollout (operator_sequence + energy) to a JSONL
"distillation database" -- Section 3.2.3's self-distillation stage reads
from exactly this kind of log, though the filtering/SFT-round logic itself
is a separate next piece, not implemented here.

Usage:
  python rl_grpo.py \\
      --checkpoint checkpoints_12q_eps5/best_checkpoint.pt \\
      --tokenizer-dir training_examples_12q_eps5 \\
      --train-jsonl training_examples_12q_eps5/train.jsonl \\
      --xyz-dir full_dataset \\
      --basis 6-31g --charge 0 --spin 0 \\
      --n-active-electrons 6 --n-active-orbitals 6 --pool uccgsd \\
      --group-size 16 --conformers-per-step 4 --epochs-per-batch 2 \\
      --lr 4e-6 --beta 0.28 --eps-high 0.001 \\
      --output-dir rl_checkpoints_12q_eps5 \\
      --rollout-log rl_checkpoints_12q_eps5/rollouts.jsonl
"""
import argparse
import copy
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "chemistry"))

from generate import generate, load_model_from_checkpoint, parse_generated_sequence  # noqa: E402
from operator_tokenizer import OperatorTokenizer  # noqa: E402

from circuit_energy import build_ansatz_kernel, compute_reward, evaluate_energy  # noqa: E402
from evaluate_generated_circuits import build_molecule_and_pool, resolve_xyz_path  # noqa: E402


class MoleculePoolCache:
    """Avoids rebuilding the PySCF/cudaq_solvers molecule + operator pool
    for the same conformer across repeated RL steps -- conformers WILL
    repeat over the course of an RL run drawn from a finite training pool."""

    def __init__(self, args, xyz_dir):
        self.args = args
        self.xyz_dir = xyz_dir
        self._cache = {}

    def get(self, conformer_id):
        if conformer_id not in self._cache:
            xyz_path = resolve_xyz_path(conformer_id, self.xyz_dir)
            self._cache[conformer_id] = build_molecule_and_pool(xyz_path, self.args)
        return self._cache[conformer_id]


def pad_sequences(list_of_id_lists, pad_id, device):
    max_len = max(len(ids) for ids in list_of_id_lists)
    input_ids = torch.full((len(list_of_id_lists), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(list_of_id_lists), max_len), dtype=torch.long)
    for i, ids in enumerate(list_of_id_lists):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, : len(ids)] = 1
    return input_ids.to(device), attention_mask.to(device)


def token_log_probs(model, input_ids, attention_mask, hamiltonian):
    """Per-token log-prob of the ACTUAL next token at every position,
    aligned with input_ids[:, 1:] / attention_mask[:, 1:]."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, hamiltonian=hamiltonian)
    log_probs_all = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]
    return log_probs_all.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)


def collect_rollouts(model, tokenizer, conformer_rows, pool_cache, ansatz, args, device):
    """Generate --group-size candidates per conformer, score their circuit
    energy, and return everything needed for the DAPO update: padded
    sequences, repeated Hamiltonian vectors, per-sequence advantages, and a
    response mask. Also returns per-rollout log records for the
    self-distillation database."""
    ham_batch = torch.tensor([r["hamiltonian"] for r in conformer_rows], dtype=torch.float32)

    per_conformer_sequences = generate(
        model,
        tokenizer,
        ham_batch,
        num_candidates=args.group_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )

    all_token_ids, all_hamiltonians, all_advantages, log_records = [], [], [], []

    for row, candidate_token_ids in zip(conformer_rows, per_conformer_sequences):
        cid = row["conformer_id"]
        try:
            molecule, pool, n_qubits, n_electrons = pool_cache.get(cid)
        except Exception as e:
            print(f"  skipping conformer {cid}: molecule/pool build failed: {e}")
            continue

        rewards, sequences_for_group = [], []
        for token_ids in candidate_token_ids:
            op_sequence, n_invalid = parse_generated_sequence(token_ids, tokenizer)
            try:
                e_circuit = evaluate_energy(
                    molecule, pool, op_sequence, n_qubits, n_electrons, ansatz=ansatz
                )
                reward = compute_reward(row["e_hf"], row["e_ref"], e_circuit, r_max=args.reward_r_max)
            except Exception as e:
                print(f"    candidate energy eval failed for {cid}: {e}")
                e_circuit, reward = None, -args.reward_r_max  # worst-case penalty, not from the paper

            rewards.append(reward)
            sequences_for_group.append(token_ids)
            log_records.append(
                {
                    "conformer_id": cid,
                    "operator_sequence": op_sequence,
                    "n_operators": len(op_sequence),
                    "n_invalid_pairs": n_invalid,
                    "energy": e_circuit,
                    "reward": reward,
                }
            )

        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-6)

        for token_ids, adv in zip(sequences_for_group, advantages.tolist()):
            all_token_ids.append(token_ids)
            all_hamiltonians.append(row["hamiltonian"])
            all_advantages.append(adv)

    if not all_token_ids:
        return None

    input_ids, attention_mask = pad_sequences(all_token_ids, tokenizer.pad_token_id, device)
    hamiltonian = torch.tensor(all_hamiltonians, dtype=torch.float32, device=device)
    advantages = torch.tensor(all_advantages, dtype=torch.float32, device=device)
    response_mask = attention_mask[:, 1:].float()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "hamiltonian": hamiltonian,
        "advantages": advantages,
        "response_mask": response_mask,
        "log_records": log_records,
    }


def dapo_loss(new_log_probs, old_log_probs, ref_log_probs, advantages, response_mask, eps_low, eps_high, beta):
    """DAPO loss (Sec. 3.2.3): clip-higher PPO-style clipped objective with
    TOKEN-LEVEL normalization (sum over all tokens in the batch / total
    valid token count, rather than averaging per-sequence first) to remove
    the length bias vanilla GRPO has toward short sequences, plus a k3-style
    KL penalty against the frozen pretrained reference policy."""
    ratio = torch.exp(new_log_probs - old_log_probs)
    adv = advantages.unsqueeze(-1)  # broadcast per-sequence advantage over all its tokens

    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * adv
    policy_term = -torch.min(unclipped, clipped)

    # Schulman's k3 KL estimator: unbiased, always >= 0, low variance.
    log_ratio_ref = ref_log_probs - new_log_probs
    kl_term = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0

    per_token_loss = (policy_term + beta * kl_term) * response_mask
    total_tokens = response_mask.sum().clamp(min=1.0)
    return per_token_loss.sum() / total_tokens


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True, help="Pretrained checkpoint from pretrain.py")
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--train-jsonl", required=True, help="train.jsonl from build_training_examples.py")
    p.add_argument("--xyz-dir", required=True)

    p.add_argument("--basis", default="6-31g")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--spin", type=int, default=0)
    p.add_argument("--n-active-electrons", type=int, required=True)
    p.add_argument("--n-active-orbitals", type=int, required=True)
    p.add_argument("--pool", choices=["uccsd", "uccgsd", "upccgsd"], required=True)

    p.add_argument("--group-size", type=int, default=16, help="Generations per conformer (paper: 16)")
    p.add_argument("--conformers-per-step", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=600)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--reward-r-max", type=float, default=30.0)

    p.add_argument("--epochs-per-batch", type=int, default=2, help="Gradient updates per rollout (paper: 2-4)")
    p.add_argument("--lr", type=float, default=4e-6)
    p.add_argument("--beta", type=float, default=0.28, help="KL penalty coefficient")
    p.add_argument("--eps-low", type=float, default=0.001)
    p.add_argument("--eps-high", type=float, default=0.001)
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    p.add_argument("--num-steps", type=int, default=1000)
    p.add_argument("--save-every-steps", type=int, default=50)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--rollout-log", required=True, help="Self-distillation database (JSONL, appended)")
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    tokenizer = OperatorTokenizer.from_pretrained(args.tokenizer_dir)
    model = load_model_from_checkpoint(args.checkpoint, args.device)
    model.train()

    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p_ in ref_model.parameters():
        p_.requires_grad_(False)

    with open(args.train_jsonl) as f:
        train_rows = [json.loads(line) for line in f]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ansatz = build_ansatz_kernel()
    pool_cache = MoleculePoolCache(args, args.xyz_dir)

    for step in range(1, args.num_steps + 1):
        conformer_rows = random.sample(
            train_rows, min(args.conformers_per_step, len(train_rows))
        )

        model.eval()
        with torch.no_grad():
            rollout = collect_rollouts(
                model, tokenizer, conformer_rows, pool_cache, ansatz, args, args.device
            )
        model.train()

        if rollout is None:
            print(f"step {step}: no valid rollouts, skipping")
            continue

        with open(args.rollout_log, "a") as log_f:
            for record in rollout["log_records"]:
                log_f.write(json.dumps(record) + "\n")

        mean_reward = sum(r["reward"] for r in rollout["log_records"]) / len(rollout["log_records"])
        mean_ops = sum(r["n_operators"] for r in rollout["log_records"]) / len(rollout["log_records"])

        with torch.no_grad():
            old_log_probs = token_log_probs(
                model, rollout["input_ids"], rollout["attention_mask"], rollout["hamiltonian"]
            )
            ref_log_probs = token_log_probs(
                ref_model, rollout["input_ids"], rollout["attention_mask"], rollout["hamiltonian"]
            )

        for epoch in range(args.epochs_per_batch):
            new_log_probs = token_log_probs(
                model, rollout["input_ids"], rollout["attention_mask"], rollout["hamiltonian"]
            )
            loss = dapo_loss(
                new_log_probs,
                old_log_probs,
                ref_log_probs,
                rollout["advantages"],
                rollout["response_mask"],
                args.eps_low,
                args.eps_high,
                args.beta,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        print(
            f"step {step}: mean_reward={mean_reward:.3f} mean_n_operators={mean_ops:.1f} "
            f"last_epoch_loss={loss.item():.4f}"
        )

        if step % args.save_every_steps == 0:
            ckpt_path = os.path.join(args.output_dir, f"rl_checkpoint_step{step}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "step": step,
                    "mean_reward": mean_reward,
                },
                ckpt_path,
            )
            print(f"  saved {ckpt_path}")

    print("RL post-training complete.")


if __name__ == "__main__":
    main()
