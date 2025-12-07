#!/usr/bin/env python3
import argparse, subprocess, os, sys
from pathlib import Path
import textwrap, shutil, json, datetime

# -----------------------------
# ANSI colors for pretty logs
# -----------------------------
RESET = "\033[0m"
RED = "\033[31m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
CYAN = "\033[36m"

# -------------------------------------------------
# Repo / sitecustomize plumbing (for MURA)
# -------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
SITECUSTOMIZE_SRC = REPO_ROOT / "sitecustomize.py"   # your file lives next to the launcher


def _read_shebang_python(bin_name: str) -> str | None:
    """Return absolute python path from the shebang of a console_script (trl/accelerate)."""
    try:
        bin_path = shutil.which(bin_name)
        if not bin_path:
            return None
        with open(bin_path, "rb") as f:
            first = f.readline().decode("utf-8", "ignore").strip()
        if first.startswith("#!"):
            py = first[2:].strip().split()[0]
            return py
    except Exception:
        return None


def _py_exec(pybin: str, code: str) -> str:
    out = subprocess.check_output([pybin, "-c", code], text=True)
    return out.strip()


def _install_into_site_packages(pybin: str, src_file: Path):
    """Copy sitecustomize.py and drop a .pth hook into pybin's site-packages."""
    code = textwrap.dedent(
        """
        import site, sys, os, shutil, json
        paths = []
        try: paths += site.getsitepackages()
        except Exception: pass
        paths.append(site.getusersitepackages())
        paths = [p for p in paths if p and os.path.isdir(p)]
        print(json.dumps(paths))
        """
    )
    paths = json.loads(_py_exec(pybin, code))
    if not paths:
        raise RuntimeError(f"No site-packages found for {pybin}")

    # Prefer a writable path
    target = None
    for p in paths:
        if os.access(p, os.W_OK):
            target = p
            break
    if target is None:
        # last resort: try user site even if not marked writable
        target = paths[-1]

    target_py = os.path.join(target, "sitecustomize.py")
    target_pth = os.path.join(target, "mura_sitecustomize.pth")

    # copy sitecustomize.py
    shutil.copy2(str(src_file), target_py)

    # write a .pth that forces import (belt & suspenders)
    with open(target_pth, "w") as f:
        f.write(str(REPO_ROOT) + "\n")
        f.write("import sitecustomize\n")

    print(f"{CYAN}[launcher] Installed sitecustomize -> {target_py}{RESET}")
    print(f"{CYAN}[launcher] Installed .pth hook -> {target_pth}{RESET}")


def _ensure_sitecustomize_everywhere():
    """Install sitecustomize into the TRL and Accelerate envs if present."""
    if not SITECUSTOMIZE_SRC.exists():
        print(f"{YELLOW}[launcher] WARNING: {SITECUSTOMIZE_SRC} not found; skipping patch install.{RESET}")
        return

    seen = set()
    for bin_name in ("trl", "accelerate"):
        py = _read_shebang_python(bin_name)
        if py and py not in seen:
            try:
                _install_into_site_packages(py, SITECUSTOMIZE_SRC)
                seen.add(py)
            except Exception as e:
                print(f"{YELLOW}[launcher] WARNING: failed installing into {py}: {e}{RESET}")

    # Also export env so it propagates to ranks
    os.environ["PYTHONPATH"] = f"{str(REPO_ROOT)}:{os.environ.get('PYTHONPATH', '')}"


def _enable_or_disable_mura(enable: bool):
    """
    Knob to turn MURA on/off.
    """
    if enable:
        _ensure_sitecustomize_everywhere()
        os.environ["USE_MURA"] = "1"
        os.environ.setdefault("LORA_NS_NO_COMPILE", "0")
        print(f"{GREEN}[launcher] MURA enabled (USE_MURA=1).{RESET}")
    else:
        os.environ["USE_MURA"] = "0"
        print(f"{YELLOW}[launcher] MURA disabled (USE_MURA=0).{RESET}")


# -------------------------------------------------
# Accelerate prefix: 8 GPUs on a single node
# -------------------------------------------------
ACCEL_PREFIX = os.environ.get(
    "ACCEL_PREFIX",
    "accelerate launch --num_processes 8 --mixed_precision bf16 --gpu_ids all --rdzv_backend c10d",
)


def with_accel(cmd: str) -> str:
    return f"{ACCEL_PREFIX} {cmd}"


def run(cmd, dry):
    print(cmd)
    if not dry:
        subprocess.run(cmd, shell=True, check=True)


# -------------------------------------------------
# Helper: encode hyperparams into name + timestamp
# -------------------------------------------------
def hp_suffix(
    lr=None,
    lora_r=None,
    lora_alpha=None,
    lora_dropout=None,
    optim=None,
    weight_decay=None,
    adam_beta1=None,
    adam_beta2=None,
):
    parts = []
    if lr is not None:
        parts.append(f"lr{lr}")
    if lora_r is not None:
        parts.append(f"r{lora_r}")
    if lora_alpha is not None:
        parts.append(f"a{lora_alpha}")
    if lora_dropout is not None:
        parts.append(f"drop{lora_dropout}")
    if optim:
        parts.append(f"opt{optim}")
    if weight_decay is not None:
        parts.append(f"wd{weight_decay}")
    if adam_beta1 is not None:
        parts.append(f"b1{adam_beta1}")
    if adam_beta2 is not None:
        parts.append(f"b2{adam_beta2}")

    if not parts:
        return ""

    tag = "_".join(parts).replace("/", "-")
    return "__" + tag


def apply_hp_suffix_to_name_and_dir(base_run_name, base_outdir, timestamp=None, **hp_kwargs):
    """
    Append HP suffix and timestamp suffix to run_name and outdir.
    """
    suffix = hp_suffix(**hp_kwargs)
    run_name = base_run_name + suffix
    outdir = base_outdir + suffix
    if timestamp is not None:
        ts_tag = f"__ts{timestamp}"
        run_name += ts_tag
        outdir += ts_tag
    return run_name, outdir


# -------------------------------------------------
# SFT helpers (TRL CLI)
# -------------------------------------------------
def add_common_sft_flags(
    base,
    outdir,
    run_name,
    lr,
    per_device,
    grad_accum,
    epochs,
    lora_r,
    lora_alpha,
    lora_dropout,
    target="all-linear",
    optim=None,
    weight_decay=None,
    adam_beta1=None,
    adam_beta2=None,
    resume_from=None,
):
    bits = [
        "--use_peft",
        f"--lora_target_modules {target}",
        f"--lora_r {lora_r}",
        f"--lora_alpha {lora_alpha}",
        f"--lora_dropout {lora_dropout}",
        f"--learning_rate {lr}",
        "--lr_scheduler_type cosine",
        "--warmup_ratio 0.1",
        f"--per_device_train_batch_size {per_device}",
        f"--gradient_accumulation_steps {grad_accum}",
        f"--num_train_epochs {epochs}",
        "--report_to wandb",
        "--logging_dir wandb_logs",
        f"--output_dir {outdir}",
        f"--run_name {run_name}",
        "--ddp_find_unused_parameters False",
    ]
    if optim:
        bits.append(f"--optim {optim}")
    if weight_decay is not None:
        bits.append(f"--weight_decay {weight_decay}")
    if adam_beta1 is not None:
        bits.append(f"--adam_beta1 {adam_beta1}")
    if adam_beta2 is not None:
        bits.append(f"--adam_beta2 {adam_beta2}")
    if resume_from:
        bits.append(f"--resume_from_checkpoint {resume_from}")

    extra = " " + " ".join(bits)
    return base + extra


def mk_sft_cmd(
    model,
    dataset,
    base_outdir,
    base_run_name,
    lr,
    per_device,
    grad_accum,
    epochs,
    lora_r,
    lora_alpha,
    lora_dropout,
    optim=None,
    weight_decay=None,
    adam_beta1=None,
    adam_beta2=None,
    timestamp=None,
    resume_from=None,
):
    TRL_BIN = os.environ.get("TRL_BIN", "trl")
    accel = "--num_processes 8 --mixed_precision bf16 --gpu_ids all"
    base = f"{TRL_BIN} sft {accel} --model_name_or_path {model} --dataset_name {dataset}"

    # Build hp+timestamp-suffixed run_name/outdir
    run_name, outdir = apply_hp_suffix_to_name_and_dir(
        base_run_name,
        base_outdir,
        timestamp=timestamp,
        lr=lr,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        optim=optim,
        weight_decay=weight_decay,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
    )

    cmd = add_common_sft_flags(
        base,
        outdir,
        run_name,
        lr=lr,
        per_device=per_device,
        grad_accum=grad_accum,
        epochs=epochs,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        optim=optim,
        weight_decay=weight_decay,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        resume_from=resume_from,
    )
    return cmd


# -------------------------------------------------
# GRPO helpers (reference script)
# -------------------------------------------------
def mk_grpo_cmd_with_ref_script(
    grpo_py,
    model,
    dataset,
    base_outdir,
    base_run_name,
    use_lora=True,
    lr=None,
    lora_r=1,
    lora_alpha=32,
    lora_dropout=0.0,
    max_prompt_len=None,
    max_completion_len=None,
    per_device=1,
    grad_accum=4,
    num_generations=8,
    gen_batch=8,
    steps=None,
    push_to_hub=False,
    optim=None,
    weight_decay=None,
    adam_beta1=None,
    adam_beta2=None,
    timestamp=None,
    resume_from=None,
):
    # Build hp+timestamp-suffixed run_name/outdir
    run_name, outdir = apply_hp_suffix_to_name_and_dir(
        base_run_name,
        base_outdir,
        timestamp=timestamp,
        lr=lr,
        lora_r=lora_r if use_lora else None,
        lora_alpha=lora_alpha if use_lora else None,
        lora_dropout=lora_dropout if use_lora else None,
        optim=optim,
        weight_decay=weight_decay,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
    )

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
    if lr:
        pieces += [f"--learning_rate {lr}"]
    if max_prompt_len:
        pieces += [f"--max_prompt_length {max_prompt_len}"]
    if max_completion_len:
        pieces += [f"--max_completion_length {max_completion_len}"]
    if steps:
        pieces += [f"--max_steps {steps}"]
    if push_to_hub:
        pieces += ["--push_to_hub"]
    if use_lora:
        pieces += [
            "--use_peft",
            "--lora_target_modules all-linear",
            f"--lora_r {lora_r}",
            f"--lora_alpha {lora_alpha}",
            f"--lora_dropout {lora_dropout}",
        ]
    if optim:
        pieces += [f"--optim {optim}"]
    if weight_decay is not None:
        pieces += [f"--weight_decay {weight_decay}"]
    if adam_beta1 is not None:
        pieces += [f"--adam_beta1 {adam_beta1}"]
    if adam_beta2 is not None:
        pieces += [f"--adam_beta2 {adam_beta2}"]
    if resume_from:
        pieces += [f"--resume_from_checkpoint {resume_from}"]

    return with_accel(" ".join(pieces))


# -------------------------------------------------
# Experiment matrix (builders with per-experiment defaults)
# -------------------------------------------------
def all_experiments(args, timestamp):
    """
    Returns a dict: { base_name: builder(base_run_name, base_outdir) -> cmd_str }
    """
    exps = {}
    grpo_py = os.environ.get("GRPO_PY", "grpo.py")

    # Optimizer-related overrides (None => use library defaults)
    optim = args.optim
    weight_decay = args.weight_decay
    adam_beta1 = args.adam_beta1
    adam_beta2 = args.adam_beta2
    resume_from = args.resume_from

    # LoRA overrides (if None, use per-experiment defaults)
    lora_r_sft_override = args.lora_r_sft
    lora_alpha_sft_override = args.lora_alpha_sft
    lora_dropout_sft_override = args.lora_dropout_sft

    lora_r_grpo_override = args.lora_r_grpo
    lora_alpha_grpo_override = args.lora_alpha_grpo
    lora_dropout_grpo_override = args.lora_dropout_grpo

    # Learning rate global overrides (per-family)
    lr_sft_override = args.lr_sft
    lr_grpo_override = args.lr_grpo
    lr_smollm3_lora_override = args.lr_smollm3_lora
    lr_smollm3_fullft_override = args.lr_smollm3_fullft

    # ----- SFT -----
    def add_sft(name, model, dataset, defaults):
        def builder(base_run_name, base_outdir):
            lr = lr_sft_override if lr_sft_override is not None else defaults["lr"]
            lora_r = lora_r_sft_override if lora_r_sft_override is not None else defaults["lora_r"]
            lora_alpha = (
                lora_alpha_sft_override if lora_alpha_sft_override is not None else defaults["lora_alpha"]
            )
            lora_dropout = (
                lora_dropout_sft_override if lora_dropout_sft_override is not None else defaults["lora_dropout"]
            )
            return mk_sft_cmd(
                model=model,
                dataset=dataset,
                base_outdir=base_outdir,
                base_run_name=base_run_name,
                lr=lr,
                per_device=defaults["per_device"],
                grad_accum=defaults["grad_accum"],
                epochs=defaults["epochs"],
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                optim=optim,
                weight_decay=weight_decay,
                adam_beta1=adam_beta1,
                adam_beta2=adam_beta2,
                timestamp=timestamp,
                resume_from=resume_from,
            )
        exps[name] = builder

    sft_defaults = {
        "lr": 2e-4,
        "per_device": 4,
        "grad_accum": 1,
        "epochs": 1,
        "lora_r": 256,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
    }

    add_sft("sft-llama-1b-tulu3",       "meta-llama/Llama-3.2-1B-Instruct", "allenai/tulu-3-sft-mixture", sft_defaults)
    add_sft("sft-llama-1b-openthoughts","meta-llama/Llama-3.2-1B-Instruct", "open-thoughts/OpenThoughts-114k", sft_defaults)
    add_sft("sft-llama-8b-tulu3",       "meta-llama/Llama-3.1-8B-Instruct", "allenai/tulu-3-sft-mixture", sft_defaults)
    add_sft("sft-llama-8b-openthoughts","meta-llama/Llama-3.1-8B-Instruct", "open-thoughts/OpenThoughts-114k", sft_defaults)

    # ----- GRPO with LoRA (LLaMA/Qwen) -----
    def add_grpo_lora(
        name,
        model,
        dataset,
        defaults,
        is_smollm3=False,
    ):
        def builder(base_run_name, base_outdir):
            if is_smollm3:
                base_lr = (
                    lr_smollm3_lora_override
                    if lr_smollm3_lora_override is not None
                    else defaults["lr"]
                )
            else:
                base_lr = lr_grpo_override if lr_grpo_override is not None else defaults["lr"]

            lora_r = lora_r_grpo_override if lora_r_grpo_override is not None else defaults["lora_r"]
            lora_alpha = (
                lora_alpha_grpo_override if lora_alpha_grpo_override is not None else defaults["lora_alpha"]
            )
            lora_dropout = (
                lora_dropout_grpo_override
                if lora_dropout_grpo_override is not None
                else defaults["lora_dropout"]
            )
            return mk_grpo_cmd_with_ref_script(
                grpo_py=grpo_py,
                model=model,
                dataset=dataset,
                base_outdir=base_outdir,
                base_run_name=base_run_name,
                use_lora=True,
                lr=base_lr,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                max_prompt_len=defaults.get("max_prompt_len"),
                max_completion_len=defaults.get("max_completion_len"),
                per_device=defaults["per_device"],
                grad_accum=defaults["grad_accum"],
                num_generations=defaults["num_generations"],
                gen_batch=defaults["gen_batch"],
                optim=optim,
                weight_decay=weight_decay,
                adam_beta1=adam_beta1,
                adam_beta2=adam_beta2,
                timestamp=timestamp,
                resume_from=resume_from,
            )
        exps[name] = builder

    grpo_defaults = {
        "lr": 5e-5,
        "per_device": 4,
        "grad_accum": 1,
        "lora_r": 1,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "num_generations": 8,
        "gen_batch": 8,
    }

    add_grpo_lora("grpo-llama8b-gsm8k",
                  "meta-llama/Llama-3.1-8B-Base",
                  "openai/gsm8k",
                  grpo_defaults)

    add_grpo_lora("grpo-llama8b-deepmath",
                  "meta-llama/Llama-3.1-8B-Base",
                  "HuggingFaceH4/DeepMath-103K",
                  grpo_defaults)

    add_grpo_lora("grpo-qwen8b-deepmath",
                  "Qwen/Qwen3-8B-Base",
                  "HuggingFaceH4/DeepMath-103K",
                  grpo_defaults)

    # SmolLM3 GRPO with LoRA
    smollm3_lora_defaults = {
        "lr": 1e-5,
        "per_device": 1,
        "grad_accum": 4,
        "lora_r": 1,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "num_generations": 8,
        "gen_batch": 8,
        "max_prompt_len": 1024,
        "max_completion_len": 4096,
    }

    add_grpo_lora("grpo-smollm3-lora",
                  "HuggingFaceTB/SmolLM3-3B",
                  "HuggingFaceH4/OpenR1-Math-220k-default-verified",
                  smollm3_lora_defaults,
                  is_smollm3=True)

    # ----- GRPO full fine-tune (SmolLM3) -----
    def add_grpo_fullft(
        name,
        model,
        dataset,
        defaults,
    ):
        def builder(base_run_name, base_outdir):
            base_lr = (
                lr_smollm3_fullft_override
                if lr_smollm3_fullft_override is not None
                else defaults["lr"]
            )
            return mk_grpo_cmd_with_ref_script(
                grpo_py=grpo_py,
                model=model,
                dataset=dataset,
                base_outdir=base_outdir,
                base_run_name=base_run_name,
                use_lora=False,
                lr=base_lr,
                max_prompt_len=defaults.get("max_prompt_len"),
                max_completion_len=defaults.get("max_completion_len"),
                per_device=defaults["per_device"],
                grad_accum=defaults["grad_accum"],
                num_generations=defaults["num_generations"],
                gen_batch=defaults["gen_batch"],
                optim=optim,
                weight_decay=weight_decay,
                adam_beta1=adam_beta1,
                adam_beta2=adam_beta2,
                timestamp=timestamp,
                resume_from=resume_from,
            )
        exps[name] = builder

    smollm3_fullft_defaults = {
        "lr": 1e-6,
        "per_device": 1,
        "grad_accum": 4,
        "num_generations": 8,
        "gen_batch": 8,
        "max_prompt_len": 1024,
        "max_completion_len": 4096,
    }

    add_grpo_fullft("grpo-smollm3-fullft",
                    "HuggingFaceTB/SmolLM3-3B",
                    "HuggingFaceH4/OpenR1-Math-220k-default-verified",
                    smollm3_fullft_defaults)

    return exps


# -------------------------------------------------
# Name resolution helpers (base name vs -mura)
# -------------------------------------------------
def resolve_experiment_for_run(requested_name: str, use_mura: bool, exps: dict):
    suffix = "-mura"
    if requested_name in exps:
        base_name = requested_name
        requested_has_suffix = requested_name.endswith(suffix)
    elif requested_name.endswith(suffix) and requested_name[:-len(suffix)] in exps:
        base_name = requested_name[:-len(suffix)]
        requested_has_suffix = True
    else:
        print(f"{RED}Unknown experiment: {requested_name}. Use 'list' to see names.{RESET}")
        sys.exit(2)

    if use_mura:
        run_name = base_name + suffix
        if not requested_has_suffix:
            print(
                f"{CYAN}[launcher] INFO: enabling --mura; "
                f"using run_name '{run_name}' (requested '{requested_name}').{RESET}"
            )
    else:
        run_name = base_name
        if requested_has_suffix:
            print(
                f"{YELLOW}[launcher WARNING] Requested '{requested_name}' but --mura is OFF; "
                f"using baseline run_name '{run_name}' (no MURA).{RESET}"
            )

    return base_name, run_name


def display_name_for_list(base_name: str, use_mura: bool) -> str:
    return base_name + ("-mura" if use_mura else "")


# -------------------------------------------------
# CLI
# -------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Central launcher for 'LoRA Without Regret' experiments (multi-GPU, wandb)."
    )
    # MURA knob (OFF by default)
    p.add_argument(
        "--mura",
        action="store_true",
        help="Enable MURA sitecustomize patching (default is OFF).",
    )

    # Global resume-from
    p.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path or identifier passed as --resume_from_checkpoint to TRL/GRPO scripts.",
    )

    # base storage
    p.add_argument(
        "--storage-base",
        type=str,
        default="/data/",
        help="Base directory for all experiment outputs.",
    )

    # Hyperparameter overrides
    hp = p.add_argument_group("Hyperparameter overrides")

    # LoRA for SFT
    hp.add_argument("--lora-r-sft", type=int, default=None, help="Override LoRA rank for SFT.")
    hp.add_argument("--lora-alpha-sft", type=float, default=None, help="Override LoRA alpha for SFT.")
    hp.add_argument("--lora-dropout-sft", type=float, default=None, help="Override LoRA dropout for SFT.")

    # LoRA for GRPO (incl. SmolLM3 LoRA)
    hp.add_argument("--lora-r-grpo", type=int, default=None, help="Override LoRA rank for GRPO.")
    hp.add_argument("--lora-alpha-grpo", type=float, default=None, help="Override LoRA alpha for GRPO.")
    hp.add_argument("--lora-dropout-grpo", type=float, default=None, help="Override LoRA dropout for GRPO.")

    # Learning rates
    hp.add_argument("--lr-sft", type=float, default=None, help="Override LR for all SFT experiments.")
    hp.add_argument("--lr-grpo", type=float, default=None, help="Override LR for GRPO (LLaMA/Qwen LoRA).")
    hp.add_argument("--lr-smollm3-lora", type=float, default=None, help="Override LR for SmolLM3 LoRA GRPO.")
    hp.add_argument("--lr-smollm3-fullft", type=float, default=None, help="Override LR for SmolLM3 full FT GRPO.")

    # Optimizer / weight decay / betas
    hp.add_argument("--optim", type=str, default=None, help="Optimizer type (e.g. adamw_torch, adamw_hf, sgd).")
    hp.add_argument("--weight-decay", type=float, default=None, help="Weight decay.")
    hp.add_argument("--adam-beta1", type=float, default=None, help="Adam beta1.")
    hp.add_argument("--adam-beta2", type=float, default=None, help="Adam beta2.")

    sub = p.add_subparsers(dest="cmd", required=True)

    sp_list = sub.add_parser("list", help="List all experiment commands.")
    sp_list.add_argument("--filter", type=str, default="", help="Substring filter on experiment names (after suffix).")

    sp_one = sub.add_parser("run", help="Run a single experiment by name.")
    sp_one.add_argument("name", type=str)
    sp_one.add_argument("--dry", action="store_true")

    sp_many = sub.add_parser("run-many", help="Run many experiments by substrings.")
    sp_many.add_argument("patterns", nargs="+")
    sp_many.add_argument("--dry", action="store_true")

    sp_all = sub.add_parser("run-all", help="Run ALL experiments.")
    sp_all.add_argument("--dry", action="store_true")

    args = p.parse_args()

    # Per-launch timestamp (shared by all runs in this invocation)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Turn MURA on/off before building commands
    _enable_or_disable_mura(enable=args.mura)

    exps = all_experiments(args, timestamp)

    if args.cmd == "list":
        for base_name, builder in sorted(exps.items()):
            display = display_name_for_list(base_name, args.mura)
            if args.filter and args.filter not in display:
                continue
            base_outdir = f"{args.storage_base}/runs/{display}"
            cmd = builder(display, base_outdir)
            print(f"# {display}")
            print(cmd)
            print()
        return

    if args.cmd == "run":
        base_name, run_name = resolve_experiment_for_run(args.name, args.mura, exps)
        base_outdir = f"{args.storage_base}/runs/{run_name}"
        cmd = exps[base_name](run_name, base_outdir)
        run(cmd, args.dry)
        return

    if args.cmd == "run-many":
        selected_bases = []
        for base_name in exps:
            display = display_name_for_list(base_name, args.mura)
            if any(pat in display for pat in args.patterns):
                selected_bases.append(base_name)

        if not selected_bases:
            print(f"{RED}No experiments matched given patterns.{RESET}")
            sys.exit(2)

        for base_name in selected_bases:
            display = display_name_for_list(base_name, args.mura)
            base_outdir = f"{args.storage_base}/runs/{display}"
            cmd = exps[base_name](display, base_outdir)
            run(cmd, args.dry)
        return

    if args.cmd == "run-all":
        for base_name, builder in sorted(exps.items()):
            display = display_name_for_list(base_name, args.mura)
            base_outdir = f"{args.storage_base}/runs/{display}"
            cmd = builder(display, base_outdir)
            run(cmd, args.dry)
        return


if __name__ == "__main__":
    main()