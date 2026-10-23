#!/usr/bin/env bash
# Meet on a public link: the server on localhost:8600 plus a Cloudflare quick tunnel (https://<random>.trycloudflare.com,
# no account; mic and camera work because it's HTTPS). Anyone with a room link + passcode can join and use Higgs on
# your API key, so share room links only with people you trust; creating rooms needs the admin token printed below.
#   ./public.sh            (Ctrl-C stops both)
# Env: MEET_ADMIN (default: a random token), MEET_BUDGET (Higgs requests per server run, default 5000),
#      CLOUDFLARED (default ~/.local/opt/cloudflared/cloudflared).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../../.venv/bin/python"
CF="${CLOUDFLARED:-$HOME/.local/opt/cloudflared/cloudflared}"
export MEET_ADMIN="${MEET_ADMIN:-$("$PY" -c 'import secrets; print(secrets.token_urlsafe(9))')}"
cd "$HERE" && mkdir -p out
CFP=""
"$PY" server.py --port 8600 --host 127.0.0.1 & SRV=$!
trap 'kill $SRV $CFP 2>/dev/null' EXIT
# QUIC (UDP) is often blocked on university networks; http2 runs over TCP 443
"$CF" tunnel --no-autoupdate --protocol http2 --url http://localhost:8600 2> out/cloudflared.log & CFP=$!
for _ in $(seq 1 60); do
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' out/cloudflared.log | head -1 || true)
  [ -n "$URL" ] && break; sleep 1
done
[ -z "${URL:-}" ] && { echo "tunnel didn't come up; see out/cloudflared.log"; exit 1; }
echo
echo "  Create a room:  $URL/?admin=$MEET_ADMIN"
echo "  (then share the room link it gives you; the admin link is only for you)"
echo
wait $SRV
