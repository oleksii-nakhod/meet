#!/usr/bin/env bash
# Meet on a public link: the server on localhost:8600 plus a Cloudflare quick tunnel (https://<random>.trycloudflare.com,
# no account; mic and camera work because it's HTTPS). Anyone who opens the link can create a room or join one by its
# ID or link (+ passcode) and use Higgs on your API key; MEET_BUDGET caps the spend, MEET_ADMIN locks room creation.
#   ./public.sh            (Ctrl-C stops both)
# Env: MEET_ADMIN (default: none; set it to make creating rooms need /?admin=<token>),
#      MEET_BUDGET (Higgs requests per server run, default 30000), MEET_MAX_ROOMS (default 100),
#      CLOUDFLARED (default ~/.local/opt/cloudflared/cloudflared),
#      MEET_ROOM: a default room /r/<id> that exists from the start (default Yrw_p1Ja-3U; empty: none),
#      MEET_HOST: your own hostname (e.g. meet.example.com) via a named tunnel made in the Cloudflare dashboard (Zero Trust
#      -> Networks -> Tunnels; DNS: CNAME <host> -> <tunnel id>.cfargotunnel.com, proxied); its token goes in MEET_TUNNEL_TOKEN_FILE
#      (default ~/.cloudflared/meet.token, chmod 600). Without MEET_HOST: a random trycloudflare.com link.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-$HERE/.venv/bin/python}"; [ -x "$PY" ] || PY=python3
CF="${CLOUDFLARED:-$HOME/.local/opt/cloudflared/cloudflared}"
export MEET_ADMIN="${MEET_ADMIN:-}"
export MEET_ROOM="${MEET_ROOM-Yrw_p1Ja-3U}"
cd "$HERE" && mkdir -p out
CFP=""
"$PY" server.py --port 8600 --host 127.0.0.1 & SRV=$!
trap 'kill $SRV $CFP 2>/dev/null' EXIT
# QUIC (UDP) is often blocked on university networks; http2 runs over TCP 443
if [ -n "${MEET_HOST:-}" ]; then
  TOK="${MEET_TUNNEL_TOKEN_FILE:-$HOME/.cloudflared/meet.token}"
  [ -s "$TOK" ] || { echo "MEET_HOST is set but there's no tunnel token in $TOK"; exit 1; }
  "$CF" tunnel --no-autoupdate --protocol http2 run --url http://localhost:8600 --token-file "$TOK" 2> out/cloudflared.log & CFP=$!
  for _ in $(seq 1 60); do
    grep -q 'Registered tunnel connection' out/cloudflared.log && { URL="https://$MEET_HOST"; break; }
    kill -0 $CFP 2>/dev/null || break; sleep 1
  done
else
  "$CF" tunnel --no-autoupdate --protocol http2 --url http://localhost:8600 2> out/cloudflared.log & CFP=$!
  for _ in $(seq 1 60); do
    URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' out/cloudflared.log | head -1 || true)
    [ -n "$URL" ] && break; sleep 1
  done
fi
[ -z "${URL:-}" ] && { echo "tunnel didn't come up; see out/cloudflared.log"; exit 1; }
kill -0 $SRV 2>/dev/null || { echo "the server didn't start (is another ./public.sh still running on port 8600?)"; exit 1; }
echo
echo "  Meet:          $URL/  (join a room by ID or link, or create one)"
[ -n "$MEET_ROOM" ] && echo "  Default room:  $URL/r/$MEET_ROOM"
[ -n "$MEET_ADMIN" ] && echo "  Create rooms:  $URL/?admin=$MEET_ADMIN  (MEET_ADMIN is set: only for you)"
echo
# Quick tunnels can be dropped by Cloudflare ("Tunnel not found", 2026-10-23 03:37): if either side stops, say so and
# stop the other, rather than leaving a server nobody can reach.
set +e
wait -n $SRV $CFP
if ! kill -0 $CFP 2>/dev/null; then
  echo "the tunnel stopped (see out/cloudflared.log); run ./run.sh public again" >&2
else
  echo "the server stopped" >&2
fi
