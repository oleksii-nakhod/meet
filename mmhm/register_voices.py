"""Register the library voices in common/voices/voices.json under your own API key (voice ids are per key)
and write the new ids back. The voices API is rate-limited, so requests are spaced out."""
import json
import os
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mmhm.higgs import BASE, HDR  # noqa: E402

path = os.path.join(HERE, "voices", "voices.json")
cfg = json.load(open(path))
for name, v in cfg.items():
    if name.startswith("_"):
        continue
    for attempt in range(6):
        r = requests.post(f"{BASE}/audio/voices", headers=HDR, timeout=60,
                          files={"ref_audio": open(os.path.join(HERE, "voices", v["ref_audio"]), "rb")},
                          data={"ref_text": open(os.path.join(HERE, "voices", v["ref_text"])).read()})
        if r.status_code == 200:
            v["voice"] = r.json()["voice_id"]
            print(name, v["voice"])
            break
        time.sleep(10 * (attempt + 1))
    time.sleep(5)
json.dump(cfg, open(path, "w"), indent=1)
