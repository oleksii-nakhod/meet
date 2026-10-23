"""Score every decision point with our speak/silent LoRA (Qwen3-4B, trained on AMI meetings, meeting-disjoint)
and with the zero-shot base model. Same prompt as training. Run on a GPU (Slurm: sbatch policy_job.sh)."""
import json
import os
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTER = os.path.join(HERE, "..", "..", "evidence", "third_voice", "our_lora_adapter_meeting_disjoint")
TH = 0.53  # chosen on the validation meetings during training
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
ID_SPEAK = tok.encode("<decision>SPEAK", add_special_tokens=False)[3]
ID_SILENT = tok.encode("<decision>SILENT", add_special_tokens=False)[3]


def prompt(r):
    ctx = "\n".join(f"{t['speaker']}: {t['text']}" for t in r["context_turns"][-12:]) or "(none)"
    cur = f"{r['current_turn']['speaker']}: {r['current_turn']['text']}"
    return (f"Multi-party meeting. You are speaker {r['target_speaker']}. After the current utterance there is a pause. "
            f"Should you SPEAK now or stay SILENT?\nContext:\n{ctx}\nCurrent:\n{cur}\n<decision>S")


def p_speak(model, rows):
    tok.padding_side = "left"
    out = []
    with torch.inference_mode():
        for r in rows:
            b = tok([prompt(r)], return_tensors="pt").to("cuda")
            lg = model(**b).logits[:, -1].float()
            out.append(torch.softmax(torch.stack([lg[:, ID_SPEAK], lg[:, ID_SILENT]], 1), 1)[0, 0].item())
    return out


rows = [json.loads(line) for line in open(os.path.join(HERE, "out", "decision_points.jsonl"))]
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).cuda().eval()
t0 = time.time()
zs = p_speak(base, rows)
lora = PeftModel.from_pretrained(base, ADAPTER).eval()
t1 = time.time()
ft = p_speak(lora, rows)
ms = (time.time() - t1) / len(rows) * 1000
for r, a, b in zip(rows, zs, ft):
    r["p_zero_shot"], r["p_ours"] = round(a, 4), round(b, 4)
    r["ours"] = "SPEAK" if b >= TH else "SILENT"
    r["zero_shot"] = "SPEAK" if a >= 0.5 else "SILENT"
with open(os.path.join(HERE, "out", "decisions_qwen.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps({k: r[k] for k in ("id", "gold", "p_zero_shot", "p_ours", "ours", "zero_shot")}) + "\n")
print(f"{ms:.0f} ms/decision (unbatched)", torch.cuda.get_device_name())
