#!/usr/bin/env python3
"""
Pretrain the trained-from-scratch Gemma model on ADAPT-VQE reference circuits
via next-token prediction (teacher forcing), reproducing the "Pretraining"
paragraph of Section 3.2.3 in ADAPT-GQE (arXiv:2607.22468): causal LM loss
on operator sequences conditioned on the corresponding Hamiltonian
coefficients, with early stopping on validation loss.

This is a plain PyTorch training loop rather than HF Trainer, since
HamiltonianConditionedLM is a thin nn.Module wrapper (not a PreTrainedModel)
around the Gemma backbone -- Trainer's checkpointing assumes save_pretrained
support that this wrapper does not (and does not need to) provide.

Usage:
  python pretrain.py \\
      --train-jsonl training_examples_12q_eps5/train.jsonl \\
      --val-jsonl training_examples_12q_eps5/val.jsonl \\
      --tokenizer-dir training_examples_12q_eps5 \\
      --hidden-size 1024 --num-layers 16 --num-heads 8 --num-kv-heads 8 \\
      --context-length 1024 --sliding-window 1024 --rope-theta 10000 \\
      --encoder-hidden-dim 2048 --encoder-depth 4 --encoder-ffn-mult 2.0 \\
      --encoder-dropout 0.2 \\
      --epochs 20 --batch-size 8 --lr 4e-5 --weight-decay 0.1 \\
      --output-dir checkpoints_12q_eps5
"""
import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
from gemma_hamiltonian_model import HamiltonianConditionedLM, build_gemma3_backbone  # noqa: E402
from hamiltonian_encoder import HamiltonianEncoder  # noqa: E402
from operator_tokenizer import OperatorTokenizer  # noqa: E402


class SequenceDataset(Dataset):
    def __init__(self, jsonl_path):
        self.rows = []
        with open(jsonl_path) as f:
            for line in f:
                self.rows.append(json.loads(line))
        if not self.rows:
            raise RuntimeError(f"{jsonl_path} contains no examples")
        self.hamiltonian_dim = len(self.rows[0]["hamiltonian"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {
            "input_ids": row["input_ids"],
            "hamiltonian": row["hamiltonian"],
        }


def make_collate_fn(pad_token_id):
    def collate(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        hamiltonian = torch.tensor([b["hamiltonian"] for b in batch], dtype=torch.float32)

        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids = torch.tensor(b["input_ids"], dtype=torch.long)
            input_ids[i, :n] = ids
            attention_mask[i, :n] = 1
            labels[i, :n] = ids

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "hamiltonian": hamiltonian,
        }

    return collate


def run_training_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    epochs,
    grad_accum_steps,
    max_grad_norm,
    eval_every_steps,
    patience,
    output_dir,
    device,
    checkpoint_name="best_checkpoint.pt",
    extra_checkpoint_fields=None,
):
    """Shared training loop used by both pretrain.py (from-scratch causal LM
    pretraining) and self_distillation's distill_sft.py (continued SFT from
    an existing checkpoint on a self-distillation round's filtered dataset)
    -- the loop itself doesn't care how the model/data got here, only that
    it's a HamiltonianConditionedLM + matching DataLoaders. Returns the best
    validation loss achieved. `extra_checkpoint_fields` should carry
    whatever load_model_from_checkpoint() needs (args/vocab_size/
    hamiltonian_dim/ham_token_id) so downstream stages can reload the
    result without special-casing which script produced it.
    """
    best_val_loss = float("inf")
    evals_without_improvement = 0
    global_step = 0
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            if (i + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % eval_every_steps == 0:
                    val_loss = evaluate(model, val_loader, device)
                    print(
                        f"epoch {epoch} step {global_step}: "
                        f"train_loss={outputs.loss.item():.4f} val_loss={val_loss:.4f}"
                    )
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        evals_without_improvement = 0
                        ckpt = {
                            "model_state_dict": model.state_dict(),
                            "val_loss": val_loss,
                            "global_step": global_step,
                        }
                        if extra_checkpoint_fields:
                            ckpt.update(extra_checkpoint_fields)
                        torch.save(ckpt, os.path.join(output_dir, checkpoint_name))
                    else:
                        evals_without_improvement += 1
                        if evals_without_improvement >= patience:
                            print(
                                f"Early stopping: no val-loss improvement for "
                                f"{patience} evaluations (best={best_val_loss:.4f})"
                            )
                            _write_metrics(output_dir, best_val_loss, global_step, True)
                            return best_val_loss

    print(f"Training complete. Best val_loss={best_val_loss:.4f}")
    _write_metrics(output_dir, best_val_loss, global_step, False)
    return best_val_loss


def _write_metrics(output_dir, best_val_loss, global_step, stopped_early):
    """Machine-readable training result, so external tooling (e.g. an
    orchestrator deciding whether another self-distillation round is worth
    running) doesn't have to scrape stdout."""
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "best_val_loss": best_val_loss,
                "global_step": global_step,
                "stopped_early": stopped_early,
            },
            f,
        )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        n_tokens = (batch["labels"] != -100).sum().item()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens
    model.train()
    return total_loss / max(total_tokens, 1)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--tokenizer-dir", required=True)

    p.add_argument("--hidden-size", type=int, required=True)
    p.add_argument("--num-layers", type=int, required=True)
    p.add_argument("--num-heads", type=int, required=True)
    p.add_argument("--num-kv-heads", type=int, required=True)
    p.add_argument("--context-length", type=int, required=True)
    p.add_argument("--sliding-window", type=int, required=True)
    p.add_argument("--rope-theta", type=float, default=10_000)

    p.add_argument("--encoder-hidden-dim", type=int, required=True)
    p.add_argument("--encoder-depth", type=int, default=4)
    p.add_argument("--encoder-ffn-mult", type=float, default=2.0)
    p.add_argument("--encoder-dropout", type=float, default=0.2)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument(
        "--eval-every-steps", type=int, default=200, help="Run validation every N optimizer steps"
    )
    p.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Stop after this many evaluations with no val-loss improvement",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = OperatorTokenizer.from_pretrained(args.tokenizer_dir)

    train_ds = SequenceDataset(args.train_jsonl)
    val_ds = SequenceDataset(args.val_jsonl)
    collate = make_collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    backbone = build_gemma3_backbone(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        context_length=args.context_length,
        sliding_window=args.sliding_window,
        rope_theta=args.rope_theta,
    )
    encoder = HamiltonianEncoder(
        input_dim=train_ds.hamiltonian_dim,
        hidden_dim=args.encoder_hidden_dim,
        output_dim=args.hidden_size,
        depth=args.encoder_depth,
        ffn_mult=args.encoder_ffn_mult,
        dropout=args.encoder_dropout,
    )
    model = HamiltonianConditionedLM(backbone, encoder, tokenizer.ham_token_id).to(args.device)

    decay_params = [p_ for p_ in model.parameters() if p_.requires_grad and p_.ndim > 1]
    no_decay_params = [p_ for p_ in model.parameters() if p_.requires_grad and p_.ndim <= 1]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )

    steps_per_epoch = max(1, len(train_loader) // args.grad_accum_steps)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    run_training_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        epochs=args.epochs,
        grad_accum_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
        eval_every_steps=args.eval_every_steps,
        patience=args.patience,
        output_dir=args.output_dir,
        device=args.device,
        checkpoint_name="best_checkpoint.pt",
        extra_checkpoint_fields={
            "args": vars(args),
            "hamiltonian_dim": train_ds.hamiltonian_dim,
            "vocab_size": tokenizer.vocab_size,
            "ham_token_id": tokenizer.ham_token_id,
        },
    )


if __name__ == "__main__":
    main()
