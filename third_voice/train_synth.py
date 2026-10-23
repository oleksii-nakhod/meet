"""Fine-tune the speak/silent LoRA on synthetic in-domain meetings (+ a slice of AMI for generality), then test on
the hand-written meeting (AI = "Higgs", never seen in training). Same prompt and recipe as the AMI model
(LoRA r=32, alpha=64, decision-token loss). Threshold chosen on held-out synthetic meetings.
Run on a GPU: sbatch train_synth_job.sh"""
import bisect
import json
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
AMI = os.path.join(HERE, "..", "..", "evidence", "third_voice", "md_train.jsonl")
random.seed(0); torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
ID_SPEAK = tok.encode("<decision>SPEAK", add_special_tokens=False)[3]
ID_SILENT = tok.encode("<decision>SILENT", add_special_tokens=False)[3]


def prompt(r):
    ctx = "\n".join(f"{t['speaker']}: {t['text']}" for t in r["context_turns"][-12:]) or "(none)"
    cur = f"{r['current_turn']['speaker']}: {r['current_turn']['text']}"
    return (f"Multi-party meeting. You are speaker {r['target_speaker']}. After the current utterance there is a pause. "
            f"Should you SPEAK now or stay SILENT?\nContext:\n{ctx}\nCurrent:\n{cur}\n<decision>S")


load = lambda f: [json.loads(line) for line in open(f)]  # noqa: E731
synth = load(os.path.join(HERE, "out", "synth_points.jsonl"))
meetings = sorted({r["meeting"] for r in synth})
random.shuffle(meetings)
val_m = set(meetings[: max(1, len(meetings) // 10)])
train = [r for r in synth if r["meeting"] not in val_m]
val = [r for r in synth if r["meeting"] in val_m]
ami = load(AMI)
train += random.sample(ami, min(len(ami), len(train) // 2))  # keep some real meeting behaviour
random.shuffle(train)
test = load(os.path.join(HERE, "out", "decision_points.jsonl"))
for r in test:
    r["decision"] = r["gold"]
print(f"train {len(train)} (synthetic {len(train) - len(train) // 3} + AMI), val {len(val)}, test {len(test)}", flush=True)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).cuda()
model.gradient_checkpointing_enable(); model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)


def p_speak(rows, bs=16):
    model.eval(); out = []
    tok.padding_side = "left"
    with torch.inference_mode():
        for i in range(0, len(rows), bs):
            b = tok([prompt(r) for r in rows[i:i + bs]], return_tensors="pt", padding=True).to("cuda")
            lg = model(**b).logits[:, -1].float()
            out += torch.softmax(torch.stack([lg[:, ID_SPEAK], lg[:, ID_SILENT]], 1), 1)[:, 0].tolist()
    model.train(); return out


def bacc(ps, rows, th):
    y = [r["decision"] == "SPEAK" for r in rows]
    tp = sum(p >= th and t for p, t in zip(ps, y)); fn = sum(p < th and t for p, t in zip(ps, y))
    tn = sum(p < th and not t for p, t in zip(ps, y)); fp = sum(p >= th and not t for p, t in zip(ps, y))
    return .5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp)), (tp, fn, tn, fp)


def auroc(ps, rows):
    pos = [p for p, r in zip(ps, rows) if r["decision"] == "SPEAK"]; neg = sorted(p for p, r in zip(ps, rows) if r["decision"] != "SPEAK")
    return sum(bisect.bisect_left(neg, p) + .5 * (bisect.bisect_right(neg, p) - bisect.bisect_left(neg, p)) for p in pos) / (len(pos) * len(neg))


t0 = time.time()
model.train(); tok.padding_side = "right"; BS, ACC = 4, 4
for step in range(0, len(train), BS):
    batch = train[step:step + BS]
    texts = [prompt(r) + ("PEAK" if r["decision"] == "SPEAK" else "ILENT") for r in batch]
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
    labels = torch.full_like(enc.input_ids, -100)
    for j, r in enumerate(batch):
        L = len(tok(prompt(r), add_special_tokens=False).input_ids)
        labels[j, L] = enc.input_ids[j, L]
    loss = model(**enc, labels=labels).loss / ACC; loss.backward()
    if (step // BS + 1) % ACC == 0: opt.step(); opt.zero_grad()
    if step % 1000 == 0: print(f"step {step}/{len(train)} loss {loss.item() * ACC:.3f} {time.time() - t0:.0f}s", flush=True)

pv = p_speak(val)
TH = max((bacc(pv, val, i / 100)[0], i / 100) for i in range(1, 100))[1]
print(f"VAL (held-out synthetic meetings): AUROC {auroc(pv, val):.3f}, balanced acc {bacc(pv, val, TH)[0]:.3f} at threshold {TH}", flush=True)
pt = p_speak(test)
b, cm = bacc(pt, test, TH)
print(f"TEST (hand-written meeting, AI='Higgs'): balanced acc {b:.3f}  (tp, fn, tn, fp) = {cm}  AUROC {auroc(pt, test):.3f}", flush=True)
json.dump([{"id": r["id"], "gold": r["gold"], "p": round(p, 4)} for r, p in zip(test, pt)],
          open(os.path.join(HERE, "out", "decisions_synth_lora.json"), "w"), indent=1)
model.save_pretrained(os.path.join(HERE, "out", "lora_synth"))
