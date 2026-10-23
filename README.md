# Meet

A small video call (2–5 people, joined by link) with an AI participant, Higgs, built on Boson's Higgs Audio for the
Boson AI hackathon 2026. Full write-up, results and test commands: [`meet/README.md`](meet/README.md).

## Layout

- `meet/`: the call (server relay, browser UI) and the Higgs participant (`agent.py`, `floor.py`, `avatar.py`)
- `mmhm/`: Mm-hm, the turn-taking engine `meet` imports (`../mmhm`): *when* to speak, backchannels, barge-in
- `third_voice/`: Third Voice, the group-conversation judge `meet` imports (`../third_voice`): *whether* to speak

## Run

    python -m venv .venv && . .venv/bin/activate
    pip install -r mmhm/requirements.txt
    export BOSON_API_KEY=...
    cd meet && python server.py --port 8600                     # http://localhost:8600/
    cd meet && python server.py --port 8600 --https --host 0.0.0.0   # LAN (self-signed cert)
    MEET_AGENT=stub python server.py --port 8600                # no Higgs calls

`meet/README.md` refers to `./run.sh meet`; that is `cd meet && python server.py --port 8600` above.
`meet/public.sh` (public trycloudflare link) expects the venv at `../.venv` relative to this repo.

Tests without API calls (the key only has to be set, any value): `cd meet && BOSON_API_KEY=x python -m pytest tests/test_floor.py tests/test_agent_rules.py`.
