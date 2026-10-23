"""Load the voice-library voice/portrait pairs (voices/voices.json)."""
import base64
import json
import os

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "voices")


def get(name):
    v = json.load(open(os.path.join(DIR, "voices.json")))[name]
    return {**v, "portrait_path": os.path.join(DIR, v["portrait"])}


def portrait_data_uri(name):
    p = get(name)["portrait_path"]
    return "data:image/jpeg;base64," + base64.b64encode(open(p, "rb").read()).decode()
