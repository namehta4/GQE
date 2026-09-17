#!/usr/bin/env python3
"""
Run one self-distillation SFT round: continue training an existing
checkpoint (typically the previous RL checkpoint, or a prior distillation
round's output) on a round's filtered dataset from
build_distillation_dataset.py. Reproduces the SFT half of the "RL and
self-distillation loop" in Fig. 3(b) of ADAPT-GQE (arXiv:2607.22468).

Architecture is loaded from the input checkpoint (unchanged); only the
weights are fine-tuned further, and only the optimizer/schedule/data differ
from pretrain.py -- both scripts share the same run_training_loop().

Usage:
  python distill_sft.py \\
      --checkpoint rl_checkpoints_12q_eps5/rl_checkpoint_step500.pt \\
      --tokenizer-dir training_examples_12q_eps5 \\
      --train-jsonl distill_round1_12q_eps5/train.jsonl \\
      --val-jsonl training_examples_12q_eps5/val.jsonl \\
      --epochs 5 --batch-size 8 --lr 4e-5 \\
      --output-dir distill_round1_12q_eps5/checkpoints
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from generate import load_model_from_checkpoint  # noqa: E402
from operator_tokenizer import OperatorTokenizer  # noqa: E402
from pretrain import SequenceDataset, make_collate_fn, run_training_loop  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True, help="Checkpoint to continue training from")
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--train-jsonl", required=True, help="From build_distillation_dataset.py")
    p.add_argument("--val-jsonl", required=True, help="Reuse the ORIGINAL validation split")

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--eval-every-steps", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--checkpoint-name", default="best_checkpoint.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = OperatorTokenizer.from_pretrained(args.tokenizer_dir)

    raw_ckpt = torch.load(args.checkpoint, map_location=args.device)
    model = load_model_from_checkpoint(args.checkpoint, args.device)
    model.train()

    train_ds = SequenceDataset(args.train_jsonl)
    val_ds = SequenceDataset(args.val_jsonl)
    collate = make_collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

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

    # Preserve the ORIGINAL architecture metadata (unchanged by fine-tuning)
    # so downstream stages (further RL, further distillation, generation)
    # can load this checkpoint exactly like a pretrain.py checkpoint.
    extra_fields = {
        "args": raw_ckpt["args"],
        "hamiltonian_dim": raw_ckpt["hamiltonian_dim"],
        "vocab_size": raw_ckpt["vocab_size"],
        "ham_token_id": raw_ckpt["ham_token_id"],
    }

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
        checkpoint_name=args.checkpoint_name,
        extra_checkpoint_fields=extra_fields,
    )


if __name__ == "__main__":
    main()
