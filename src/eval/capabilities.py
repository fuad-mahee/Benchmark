"""RQ5: capability preservation (GSM8K / MMLU) via lm-evaluation-harness.

Requires:  pip install lm_eval   (heavy; optional install, see requirements.txt)

We shell out to the lm_eval CLI so its own result JSONs land in results/ and are
citable as-is. For the GC-repaired model pass the adapter dir; for GP there is no
weight change (hook-based), so capability impact must be measured with hooks
installed - lm_eval cannot do that natively. Strategy for GP: report base-model
scores (weights unchanged => identical logits when hooks are inactive) and discuss
hook-active behavior separately; GlitchCleaner's Table 4 claim that GP degrades
Yi-6B GSM8K is therefore itself questionable - worth a paragraph in the thesis.
"""
import subprocess

from ..common.config import results_dir


def lm_eval_command(hf_id: str, adapter_dir: str | None = None,
                    tasks: str = "gsm8k,mmlu", out_name: str = "base") -> list[str]:
    model_args = f"pretrained={hf_id},dtype=float16"
    if adapter_dir:
        model_args += f",peft={adapter_dir}"
    out = results_dir("side_effects", out_name)
    return [
        "lm_eval", "--model", "hf",
        "--model_args", model_args,
        "--tasks", tasks,
        "--batch_size", "auto",
        "--output_path", str(out),
    ]


def run_lm_eval(hf_id: str, adapter_dir: str | None = None,
                tasks: str = "gsm8k,mmlu", out_name: str = "base") -> int:
    cmd = lm_eval_command(hf_id, adapter_dir, tasks, out_name)
    print("running:", " ".join(cmd))
    return subprocess.call(cmd)
