#!/usr/bin/env bash
# A public https link for the meeting (e.g. for people outside your network), via a Cloudflare quick tunnel.
# Room creation is locked behind MEET_ADMIN so strangers with the link can't create rooms.
set -euo pipefail
cd "$(dirname "$0")"
command -v cloudflared >/dev/null || { echo "cloudflared is not installed (brew install cloudflared)"; exit 1; }
PORT="${MEET_PORT:-8600}"
if [ -z "${MEET_ADMIN:-}" ] && ! grep -qE '^MEET_ADMIN=.+' .env 2>/dev/null; then
  export MEET_ADMIN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(9))')"
  echo "No MEET_ADMIN set: using a one-off admin token for this run."
fi
.venv/bin/python -m meet.server --port "$PORT" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
sleep 2
echo "Create rooms at: <the trycloudflare link below>/?admin=${MEET_ADMIN:-<your MEET_ADMIN>}"
cloudflared tunnel --url "http://127.0.0.1:$PORT"
