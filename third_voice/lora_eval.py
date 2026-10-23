"""Score the two trained speak/silent LoRAs (Qwen3-4B; AMI meetings, and synthetic meetings + AMI) on the old
meeting, dev and test sets. Same prompt as training (no fact sheet: these models never saw one).
-> out/lora_scores.json {set: {key: {"ami": p, "synth": p}}}. Thresholds: 0.53 (AMI, from its validation
meetings) and 0.03 (synthetic, from held-out synthetic meetings), as chosen at training time."""
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import tv_eval

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTERS = {"ami": os.path.join(HERE, "..", "..", "evidence", "third_voice", "our_lora_adapter_meeting_disjoint"),
            "synth": os.path.join(HERE, "out", "lora_synth")}
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
ID_SPEAK = tok.encode("<decision>SPEAK", add_special_tokens=False)[3]
ID_SILENT = tok.encode("<decision>SILENT", add_special_tokens=False)[3]


def prompt(r):
    ctx = "\n".join(f"{t['speaker']}: {t['text']}" for t in r["context_turns"][-12:]) or "(none)"
    cur = f"{r['current_turn']['speaker']}: {r['current_turn']['text']}"
    return (f"Multi-party meeting. You are speaker {r['target_speaker']}. After the current utterance there is a pause. "
            f"Should you SPEAK now or stay SILENT?\nContext:\n{ctx}\nCurrent:\n{cur}\n<decision>S")


def p_speak(model, rows, bs=8):
    tok.padding_side = "left"
    out = []
    with torch.inference_mode():
        for i in range(0, len(rows), bs):
            b = tok([prompt(r) for r in rows[i:i + bs]], return_tensors="pt", padding=True).to("cuda")
            lg = model(**b).logits[:, -1].float()
            out += torch.softmax(torch.stack([lg[:, ID_SPEAK], lg[:, ID_SILENT]], 1), 1)[:, 0].tolist()
    return out


sets = {s: tv_eval.load(s) for s in ("meeting", "dev", "test")}
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).cuda().eval()
res = {s: {tv_eval.key(r): {} for r in rows} for s, rows in sets.items()}
for name, path in ADAPTERS.items():
    model = PeftModel.from_pretrained(base, path).eval()
    for s, rows in sets.items():
        for r, p in zip(rows, p_speak(model, rows)):
            res[s][tv_eval.key(r)][name] = round(p, 4)
    base = model.unload()
json.dump(res, open(os.path.join(HERE, "out", "lora_scores.json"), "w"))
print("done")
