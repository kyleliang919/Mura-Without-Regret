#!/usr/bin/env python3
import argparse, subprocess, os, sys

# -------------------------------------------------
# Accelerate prefix: 8 GPUs on a single node
# Override by exporting ACCEL_PREFIX if you want DS/FSDP.
# -------------------------------------------------
ACCEL_PREFIX = os.environ.get(
    "ACCEL_PREFIX",
    "accelerate launch --num_processes 8 --mixed_precision bf16 --gpu_ids all --rdzv_backend c10d"
)

def with_accel(cmd: str) -> str:
    return f"{ACCEL_PREFIX} {cmd}"

def run(cmd, dry):
    print(cmd)
    if not dry:
        subprocess.run(cmd, shell=True, check=True)

# -------------------------------------------------
# SFT helpers (TRL CLI)
# -------------------------------------------------
def add_common_sft_flags(base, outdir, run_name, lr="2e-4", per_device=1, grad_accum=4, epochs=1,
                         lora_r=256, lora_alpha=16, lora_dropout=0.0, target="all-linear"):
    """
    - Logs to wandb
    - LoRA on all linear modules (rank 256 for SFT)
    - Keep effective batch modest (LoRA can be sensitive to huge batches)
    """
    extra = f"""\
        --use_peft \
        --lora_target_modules {target} \
        --lora_r {lora_r} \
        --lora_alpha {lora_alpha} \
        --lora_dropout {lora_dropout} \
        --learning_rate {lr} \
        --per_device_train_batch_size {per_device} \
        --gradient_accumulation_steps {grad_accum} \
        --num_train_epochs {epochs} \
        --report_to wandb \
        --logging_dir wandb_logs \
        --output_dir {outdir} \
        --run_name {run_name} \
        --ddp_find_unused_parameters False \
    """
    return base + " " + " ".join(extra.split())

def mk_sft_cmd(model, dataset, outdir, run_name, **kw):
    # Use TRL CLI directly; it accepts Accelerate args like --num_processes
    TRL_BIN = os.environ.get("TRL_BIN", "trl")
    accel = "--num_processes 8 --mixed_precision bf16 --gpu_ids all"
    base = f"{TRL_BIN} sft {accel} --model_name_or_path {model} --dataset_name {dataset}"
    cmd = add_common_sft_flags(base, outdir, run_name, **kw)
    return cmd

# -------------------------------------------------
# GRPO helpers (reference script)
# -------------------------------------------------
def mk_grpo_cmd_with_ref_script(grpo_py, model, dataset,
                                outdir, run_name,
                                use_lora=True,
                                lr=None,
                                lora_r=1, lora_alpha=32, lora_dropout=0.0,
                                max_prompt_len=None, max_completion_len=None,
                                per_device=1, grad_accum=4,
                                num_generations=8, gen_batch=8,
                                steps=None, push_to_hub=False):
    """
    Uses the official GRPO reference script.
    """
    pieces = [
        f"{sys.executable} {grpo_py}",
        f"--model_name_or_path {model}",
        f"--dataset_name {dataset}",
        f"--output_dir {outdir}",
        f"--run_name {run_name}",
        f"--per_device_train_batch_size {per_device}",
        f"--gradient_accumulation_steps {grad_accum}",
        f"--generation_batch_size {gen_batch}",
        f"--num_generations {num_generations}",
        "--report_to wandb",
        "--logging_dir wandb_logs",
    ]
    if lr: pieces += [f"--learning_rate {lr}"]
    if max_prompt_len: pieces += [f"--max_prompt_length {max_prompt_len}"]
    if max_completion_len: pieces += [f"--max_completion_length {max_completion_len}"]
    if steps: pieces += [f"--max_steps {steps}"]
    if push_to_hub: pieces += ["--push_to_hub"]
    if use_lora:
        pieces += [
            "--use_peft",
            "--lora_target_modules all-linear",
            f"--lora_r {lora_r}",
            f"--lora_alpha {lora_alpha}",
            f"--lora_dropout {lora_dropout}",
        ]
    return with_accel(" ".join(pieces))

# -------------------------------------------------
# Experiment matrix
# -------------------------------------------------
def all_experiments():
    exps = {}

    # ---- SFT (LoRA rank 256, LR 2e-4) ----
    sft_defs = [
        ("sft-llama-1b-tulu3",       "meta-llama/Llama-3.2-1B-Instruct", "allenai/tulu-3-sft-mixture"),
        ("sft-llama-1b-openthoughts","meta-llama/Llama-3.2-1B-Instruct", "open-thoughts/OpenThoughts-114k"),
        ("sft-llama-8b-tulu3",       "meta-llama/Llama-3.1-8B-Instruct", "allenai/tulu-3-sft-mixture"),
        ("sft-llama-8b-openthoughts","meta-llama/Llama-3.1-8B-Instruct", "open-thoughts/OpenThoughts-114k"),
    ]
    for name, model, dataset in sft_defs:
        outdir = f"runs/{name}"
        exps[name] = [mk_sft_cmd(
            model=model, dataset=dataset,
            outdir=outdir, run_name=name,
            lr="2e-4", per_device=1, grad_accum=4, epochs=1,
            lora_r=256, lora_alpha=16, lora_dropout=0.0
        )]

    # ---- GRPO (LoRA small rank) ----
    grpo_py = os.environ.get("GRPO_PY", "grpo.py")
    grpo_defs = [
        ("grpo-llama8b-gsm8k",   "meta-llama/Llama-3.1-8B-Base", "openai/gsm8k"),
        ("grpo-llama8b-deepmath","meta-llama/Llama-3.1-8B-Base", "HuggingFaceH4/DeepMath-103K"),
        ("grpo-qwen8b-deepmath", "Qwen/Qwen3-8B-Base",           "HuggingFaceH4/DeepMath-103K"),
    ]
    for name, model, dataset in grpo_defs:
        outdir = f"runs/{name}"
        exps[name] = [mk_grpo_cmd_with_ref_script(
            grpo_py=grpo_py, model=model, dataset=dataset,
            outdir=outdir, run_name=name,
            use_lora=True, lr="5e-5",
            lora_r=1, lora_alpha=32, lora_dropout=0.0,
            per_device=1, grad_accum=4,
            num_generations=8, gen_batch=8
        )]

    # ---- GRPO: SmolLM3 LoRA vs FullFT ----
    name = "grpo-smollm3-lora"
    exps[name] = [mk_grpo_cmd_with_ref_script(
        grpo_py=grpo_py,
        model="HuggingFaceTB/SmolLM3-3B",
        dataset="HuggingFaceH4/OpenR1-Math-220k-default-verified",
        outdir=f"runs/{name}", run_name=name,
        use_lora=True, lr="1e-5",
        lora_r=1, lora_alpha=32, lora_dropout=0.0,
        max_prompt_len=1024, max_completion_len=4096,
        per_device=1, grad_accum=4,
        num_generations=8, gen_batch=8
    )]

    name = "grpo-smollm3-fullft"
    exps[name] = [mk_grpo_cmd_with_ref_script(
        grpo_py=grpo_py,
        model="HuggingFaceTB/SmolLM3-3B",
        dataset="HuggingFaceH4/OpenR1-Math-220k-default-verified",
        outdir=f"runs/{name}", run_name=name,
        use_lora=False, lr="1e-6",
        max_prompt_len=1024, max_completion_len=4096,
        per_device=1, grad_accum=4,
        num_generations=8, gen_batch=8
    )]

    return exps

# -------------------------------------------------
# CLI
# -------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Central launcher for 'LoRA Without Regret' experiments (multi-GPU, wandb).")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_list = sub.add_parser("list", help="List all experiment commands.")
    sp_list.add_argument("--filter", type=str, default="", help="Substring filter on experiment names.")

    sp_one = sub.add_parser("run", help="Run a single experiment by name.")
    sp_one.add_argument("name", type=str)
    sp_one.add_argument("--dry", action="store_true")

    sp_many = sub.add_parser("run-many", help="Run many experiments by substrings.")
    sp_many.add_argument("patterns", nargs="+")
    sp_many.add_argument("--dry", action="store_true")

    sp_all = sub.add_parser("run-all", help="Run ALL experiments.")
    sp_all.add_argument("--dry", action="store_true")

    args = p.parse_args()
    exps = all_experiments()

    if args.cmd == "list":
        for k, v in exps.items():
            if args.filter and args.filter not in k:
                continue
            print(f"# {k}")
            for c in v:
                print(c)
            print()
        return

    if args.cmd == "run":
        if args.name not in exps:
            print(f"Unknown experiment: {args.name}. Use `list` to see names.")
            sys.exit(2)
        for c in exps[args.name]:
            run(c, args.dry)
        return

    if args.cmd == "run-many":
        names = [k for k in exps if any(pat in k for pat in args.patterns)]
        if not names:
            print("No experiments matched.")
            sys.exit(2)
        for n in names:
            for c in exps[n]:
                run(c, args.dry)
        return

    if args.cmd == "run-all":
        for n in sorted(exps.keys()):
            for c in exps[n]:
                run(c, args.dry)
        return

if __name__ == "__main__":
    main()
