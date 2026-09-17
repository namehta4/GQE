#!/usr/bin/env python3
"""
Pipeline orchestrator for ADAPT-GQE: automates the manual run-order in
PIPELINE.md into a self-driving controller with resume, retries, a
smoke-test-before-full-run pattern for the two most expensive/riskiest
stages, and adaptive self-distillation stopping (based on validation-loss
improvement, not a fixed round count).

This is classical pipeline automation, not an AI-judgment agent: every
decision it makes (skip/retry/stop) is a deterministic rule over subprocess
exit codes and metrics.json files that model/pretrain.py's run_training_loop
already writes -- no new correctness risk is introduced beyond "did the
underlying script succeed."

Usage:
  python run_pipeline.py --config config.yaml
  python run_pipeline.py --config config.yaml --only-stage manifest
  python run_pipeline.py --config config.yaml --force refs_15
  python run_pipeline.py --config config.yaml --dry-run
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import yaml

from stages import build_all_stages, _script


def load_state(state_path):
    if os.path.exists(state_path):
        with open(state_path) as f:
            return json.load(f)
    return {}


def save_state(state_path, state):
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def outputs_exist(stage):
    return bool(stage.outputs) and all(os.path.exists(p) for p in stage.outputs)


def topological_order(stages):
    by_name = {s.name: s for s in stages}
    visited, order = set(), []

    def visit(name, stack):
        if name in visited:
            return
        if name in stack:
            raise RuntimeError(f"Cycle detected in stage graph at '{name}'")
        stack.add(name)
        for dep in by_name[name].depends_on:
            if dep not in by_name:
                raise RuntimeError(f"Stage '{name}' depends on unknown stage '{dep}'")
            visit(dep, stack)
        stack.discard(name)
        visited.add(name)
        order.append(by_name[name])

    for s in stages:
        visit(s.name, set())
    return order


def run_command(name, cmd, log_dir, max_attempts=2, backoff_sec=30):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")
    cmd_str = [str(c) for c in cmd]
    print(f"[{name}] running: {' '.join(cmd_str)}")

    for attempt in range(1, max_attempts + 1):
        with open(log_path, "a") as log_f:
            log_f.write(f"\n=== attempt {attempt}: {' '.join(cmd_str)} ===\n")
            log_f.flush()
            result = subprocess.run(cmd_str, stdout=log_f, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            print(f"[{name}] OK (see {log_path})")
            return True
        print(
            f"[{name}] FAILED (exit {result.returncode}), attempt {attempt}/{max_attempts} "
            f"-- see {log_path}"
        )
        if attempt < max_attempts:
            time.sleep(backoff_sec)
    return False


def run_stage(stage, workdir, state, force_names):
    log_dir = os.path.join(workdir, "logs")
    force = stage.name in force_names

    if not force and state.get(stage.name, {}).get("status") == "done":
        print(f"[{stage.name}] already marked done, skipping (use --force {stage.name} to redo)")
        return True
    if not force and outputs_exist(stage):
        print(f"[{stage.name}] outputs already exist, marking done and skipping")
        state[stage.name] = {"status": "done", "skipped": True}
        return True

    if stage.run_fn is not None:
        try:
            stage.run_fn(workdir, state)
            ok = True
        except Exception as e:
            print(f"[{stage.name}] FAILED with exception: {e}")
            ok = False
    else:
        max_attempts = 2 if stage.retryable else 1
        ok = run_command(stage.name, stage.command, log_dir, max_attempts=max_attempts)

    state[stage.name] = {"status": "done" if ok else "failed", "skipped": False}
    return ok


def run_dag(stages, workdir, state, only_stage, force_names):
    ordered = topological_order(stages)
    if only_stage:
        wanted = {only_stage} | {s.name for s in ordered if only_stage in s.depends_on}
        # only_stage plus its transitive deps
        by_name = {s.name: s for s in ordered}

        def deps_of(name, acc):
            for d in by_name[name].depends_on:
                if d not in acc:
                    acc.add(d)
                    deps_of(d, acc)

        needed = {only_stage}
        deps_of(only_stage, needed)
        ordered = [s for s in ordered if s.name in needed]

    for stage in ordered:
        ok = run_stage(stage, workdir, state, force_names)
        save_state(os.path.join(workdir, "state.json"), state)
        if not ok:
            print(f"Stopping: stage '{stage.name}' failed. Fix and re-run to resume.")
            return False
    return True


# ---------------------------------------------------------------------------
# Post-pretraining per-dataset-config pipeline: baseline eval, adaptive
# self-distillation loop, final eval, metrics/plots. These aren't part of
# the static DAG above because the self-distillation round count is dynamic.
# ---------------------------------------------------------------------------

def run_generate_and_evaluate(label, checkpoint, dctx, workdir, test_jsonl, gen_cfg, state):
    generated_path = os.path.join(workdir, f"generated_{dctx['name']}_{label}.jsonl")
    best_of_n_path = os.path.join(workdir, f"best_of_n_{dctx['name']}_{label}.jsonl")
    log_dir = os.path.join(workdir, "logs")

    gen_cmd = [
        "python", _script("model", "generate.py"),
        "--checkpoint", checkpoint, "--tokenizer-dir", dctx["training_dir"],
        "--test-jsonl", test_jsonl,
        "--num-candidates", str(gen_cfg.get("num_candidates", 16)),
        "--max-new-tokens", str(gen_cfg.get("max_new_tokens", 600)),
        "--top-p", str(gen_cfg.get("top_p", 0.95)),
        "--output", generated_path,
    ]
    if not run_command(f"{dctx['name']}_{label}_generate", gen_cmd, log_dir):
        return None

    eval_cmd = [
        "python", _script("chemistry", "evaluate_generated_circuits.py"),
        "--generated", generated_path, "--xyz-dir", dctx["manifest_dir"],
        "--basis", dctx["basis"], "--charge", str(dctx["charge"]), "--spin", str(dctx["spin"]),
        "--n-active-electrons", str(dctx["n_active_electrons"]),
        "--n-active-orbitals", str(dctx["n_active_orbitals"]), "--pool", dctx["pool"],
        "--output", best_of_n_path,
    ]
    if not run_command(f"{dctx['name']}_{label}_evaluate", eval_cmd, log_dir):
        return None
    return best_of_n_path


def run_self_distillation(dctx, sd_cfg, workdir, state):
    """Adaptive stopping: keeps running RL+distill rounds while validation
    loss keeps improving by more than `val_loss_improvement_threshold`,
    instead of blindly following a fixed round count."""
    name = dctx["name"]
    max_rounds = sd_cfg.get("max_rounds", 3)
    improvement_threshold = sd_cfg.get("val_loss_improvement_threshold", 0.01)
    rl_epochs_schedule = sd_cfg.get("rl_epochs_schedule", [2, 3, 4])
    rl_lr_schedule = sd_cfg.get("rl_lr_schedule", [4e-6, 3e-6, 2e-6])
    topn_schedule = sd_cfg.get("distill_topn_schedule", [8, 4, 2])
    threshold_schedule = sd_cfg.get("distill_threshold_mha_schedule", [5.0, 3.4, 1.6])
    qubit_key = f"{dctx['n_active_electrons']}e{dctx['n_active_orbitals']}o"

    current_ckpt = os.path.join(dctx["pretrain_ckpt_dir"], "best_checkpoint.pt")
    circuits_jsonl = os.path.join(workdir, f"adaptvqe_{name}", "circuits.jsonl")
    ham_dir = os.path.join(workdir, f"hamiltonians_{qubit_key}")
    sd_root = os.path.join(workdir, f"distill_{name}")

    prev_val_loss, rollout_logs = None, []

    for round_num in range(1, max_rounds + 1):
        idx = min(round_num - 1, len(rl_epochs_schedule) - 1)
        round_dir = os.path.join(sd_root, f"round{round_num}")
        os.makedirs(round_dir, exist_ok=True)
        rollout_log = os.path.join(round_dir, "rollouts.jsonl")
        rollout_logs.append(rollout_log)

        rl_cmd = [
            "python", _script("model", "rl_grpo.py"),
            "--checkpoint", current_ckpt, "--tokenizer-dir", dctx["training_dir"],
            "--train-jsonl", os.path.join(dctx["training_dir"], "train.jsonl"),
            "--xyz-dir", dctx["manifest_dir"],
            "--basis", dctx["basis"], "--charge", str(dctx["charge"]), "--spin", str(dctx["spin"]),
            "--n-active-electrons", str(dctx["n_active_electrons"]),
            "--n-active-orbitals", str(dctx["n_active_orbitals"]), "--pool", dctx["pool"],
            "--epochs-per-batch", str(rl_epochs_schedule[idx]), "--lr", str(rl_lr_schedule[idx]),
            "--output-dir", os.path.join(round_dir, "rl_checkpoints"),
            "--rollout-log", rollout_log,
        ]
        if not run_command(f"{name}_distill_r{round_num}_rl", rl_cmd, os.path.join(workdir, "logs")):
            print(f"[{name}] RL failed in round {round_num}, stopping self-distillation.")
            break

        rl_ckpts = sorted(glob.glob(os.path.join(round_dir, "rl_checkpoints", "rl_checkpoint_step*.pt")))
        if not rl_ckpts:
            print(f"[{name}] No RL checkpoint produced in round {round_num}, stopping.")
            break
        latest_rl_ckpt = rl_ckpts[-1]

        distill_data_dir = os.path.join(round_dir, "distill_data")
        build_cmd = ["python", _script("model", "build_distillation_dataset.py"),
                     "--circuits-jsonl", circuits_jsonl]
        for log in rollout_logs:
            build_cmd += ["--rollout-log", log]
        build_cmd += [
            "--hamiltonian-dir", ham_dir,
            "--hamiltonian-scale", os.path.join(dctx["training_dir"], "hamiltonian_scale.npy"),
            "--tokenizer-dir", dctx["training_dir"],
            "--top-n", str(topn_schedule[idx]),
            "--energy-threshold-mha", str(threshold_schedule[idx]),
            "--output-dir", distill_data_dir,
        ]
        if not run_command(f"{name}_distill_r{round_num}_build", build_cmd, os.path.join(workdir, "logs")):
            print(f"[{name}] Building distillation dataset failed in round {round_num}, stopping.")
            break

        distill_ckpt_dir = os.path.join(round_dir, "distill_checkpoint")
        sft_cmd = [
            "python", _script("model", "distill_sft.py"),
            "--checkpoint", latest_rl_ckpt, "--tokenizer-dir", dctx["training_dir"],
            "--train-jsonl", os.path.join(distill_data_dir, "train.jsonl"),
            "--val-jsonl", os.path.join(dctx["training_dir"], "val.jsonl"),
            "--output-dir", distill_ckpt_dir,
        ]
        if not run_command(f"{name}_distill_r{round_num}_sft", sft_cmd, os.path.join(workdir, "logs")):
            print(f"[{name}] Distillation SFT failed in round {round_num}, stopping.")
            break

        metrics_path = os.path.join(distill_ckpt_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            print(f"[{name}] No metrics.json after round {round_num}, stopping.")
            break
        with open(metrics_path) as f:
            val_loss = json.load(f)["best_val_loss"]
        current_ckpt = os.path.join(distill_ckpt_dir, "best_checkpoint.pt")

        if prev_val_loss is not None:
            improvement = prev_val_loss - val_loss
            print(f"[{name}] round {round_num}: val_loss={val_loss:.4f} (improvement={improvement:.4f})")
            if improvement < improvement_threshold:
                print(
                    f"[{name}] Stopping self-distillation early after round {round_num}: "
                    f"improvement {improvement:.4f} < threshold {improvement_threshold}"
                )
                break
        else:
            print(f"[{name}] round {round_num}: val_loss={val_loss:.4f}")
        prev_val_loss = val_loss

    return current_ckpt


def run_post_pretrain_pipeline(dctx, cfg, workdir, state):
    name = dctx["name"]
    gen_cfg = cfg.get("generation", {})
    test_jsonl = os.path.join(dctx["training_dir"], "test.jsonl")
    pretrain_ckpt = os.path.join(dctx["pretrain_ckpt_dir"], "best_checkpoint.pt")

    print(f"\n=== [{name}] baseline (pretraining-only) evaluation ===")
    run_generate_and_evaluate("pretrain", pretrain_ckpt, dctx, workdir, test_jsonl, gen_cfg, state)

    if cfg.get("self_distillation", {}).get("enabled", True):
        print(f"\n=== [{name}] RL + self-distillation ===")
        final_ckpt = run_self_distillation(dctx, cfg.get("self_distillation", {}), workdir, state)

        print(f"\n=== [{name}] post-distillation evaluation ===")
        run_generate_and_evaluate(
            "post_distill", final_ckpt, dctx, workdir, test_jsonl, gen_cfg, state
        )

        pretrain_results = os.path.join(workdir, f"best_of_n_{name}_pretrain.jsonl")
        post_results = os.path.join(workdir, f"best_of_n_{name}_post_distill.jsonl")
        if os.path.exists(pretrain_results) and os.path.exists(post_results):
            print(f"\n=== [{name}] metrics ===")
            metrics_cmd = [
                "python", _script("evaluation", "compute_metrics.py"),
                "--results", f"pretrain:{pretrain_results}",
                "--results", f"post_distill:{post_results}",
                "--epsilon-mha", str(dctx.get("tolerance_mha", 5.0)),
                "--output", os.path.join(workdir, f"summary_{name}.csv"),
                "--output-json", os.path.join(workdir, f"distributions_{name}.json"),
            ]
            run_command(f"{name}_metrics", metrics_cmd, os.path.join(workdir, "logs"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--only-stage", default=None, help="Run only this stage (+ its dependencies)")
    p.add_argument("--force", action="append", default=[], help="Re-run this stage even if already done")
    p.add_argument("--dry-run", action="store_true", help="Print the planned stage order and exit")
    p.add_argument(
        "--dag-only", action="store_true",
        help="Run only the static DAG (data gen through pretraining), skip "
        "the post-pretrain eval/self-distillation loop",
    )
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    workdir = os.path.abspath(cfg["workdir"])
    os.makedirs(workdir, exist_ok=True)
    state_path = os.path.join(workdir, "state.json")
    state = load_state(state_path)

    stages, dataset_ctxs = build_all_stages(cfg, workdir)

    if args.dry_run:
        for s in topological_order(stages):
            marker = "done" if state.get(s.name, {}).get("status") == "done" else "pending"
            print(f"  [{marker}] {s.name}  (depends_on={s.depends_on})")
        return

    ok = run_dag(stages, workdir, state, args.only_stage, set(args.force))
    save_state(state_path, state)
    if not ok:
        sys.exit(1)

    if not args.dag_only and not args.only_stage:
        for dctx in dataset_ctxs:
            dctx["tolerance_mha"] = next(
                d["tolerance_mha"] for d in cfg["dataset_configs"] if d["name"] == dctx["name"]
            )
            run_post_pretrain_pipeline(dctx, cfg, workdir, state)
            save_state(state_path, state)

    print("\nPipeline run complete. See state.json and logs/ for details.")


if __name__ == "__main__":
    main()
