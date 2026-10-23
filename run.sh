#!/usr/bin/env bash
# Meet: a video call for 2-5 people with Higgs (Boson AI's Higgs Audio) as a participant.
#   ./run.sh [serve] [--https --host 0.0.0.0]   the server on :8600 (MEET_ADMIN=token guards room creation)
#   ./run.sh public                             on a public https://*.trycloudflare.com link (prints the admin link)
#   ./run.sh check                              6 interactive scenes against real Higgs (~3 min): does improved Higgs answer?
#   ./run.sh tests                              unit tests, no API calls
# Python: .venv/bin/python if present (PYTHON overrides). BOSON_API_KEY: the environment, else .env.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-$HERE/.venv/bin/python}"; [ -x "$PY" ] || PY=python3
export PYTHON="$PY" PYTHONWARNINGS=ignore
cd "$HERE"
cmd="${1:-serve}"; shift || true
case "$cmd" in
  serve)  exec "$PY" server.py --port 8600 "$@" ;;
  public) exec ./public.sh ;;
  check)  exec "$PY" tests/scenarios.py --modes improved --par 3 --port 8691 --tag check \
            --only greet_1to1 followups_1to1 humans_talk group_fact bc_1to1 barge_1to1 ;;
  tests)  BOSON_API_KEY="${BOSON_API_KEY:-unused}" exec "$PY" -m pytest -q tests/test_floor.py tests/test_agent_rules.py ;;
  *)      sed -n '2,7p' "$0" ;;
esac
