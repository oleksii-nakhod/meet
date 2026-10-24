PY ?= .venv/bin/python
PORT ?= 8600

.PHONY: install install-avatar doctor run run-lan stub test live live-avatar public clean

install:            ## create .venv and install the core
	python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements.txt
	@test -f .env || (cp .env.example .env && echo "Created .env: put your BOSON_API_KEY in it")

install-avatar:     ## add the optional live-avatar packages
	$(PY) -m pip install -q -r requirements-avatar.txt

doctor:             ## check Python, packages, the API key (one cheap call each), LiveKit
	$(PY) -m meet.doctor

run:                ## http://localhost:$(PORT)
	$(PY) -m meet.server --port $(PORT)

run-lan:            ## other devices on your network (self-signed https: accept the warning)
	$(PY) -m meet.server --host 0.0.0.0 --port $(PORT) --https

stub:               ## the UI with a stand-in AI (no API calls)
	MEET_AGENT=stub $(PY) -m meet.server --port $(PORT)

test:               ## unit + integration tests, no API calls (~35 s)
	$(PY) -m pytest -q tests

live:               ## live scenarios against real Boson (real API calls, ~3 min)
	$(PY) -m tests.live.run

live-avatar:        ## the live-avatar scenario (needs LiveKit settings)
	$(PY) -m tests.live.run --only s7

public:             ## a public https link via cloudflared (set MEET_ADMIN first)
	./public.sh

clean:
	rm -rf out .pytest_cache
