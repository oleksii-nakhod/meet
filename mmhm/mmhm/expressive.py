"""Expressive voice path: Higgs Realtime writes the reply (text mode) with Higgs TTS 3 inline tags, and TTS 3
speaks it sentence by sentence in the same preset voice. Native Realtime audio never laughs or sighs (probe,
2026-10-12: 0/12 on laugh-bait); TTS 3 does, from tags like `<|sfx:laughter|>Haha`.

`SentenceSpeaker` takes text deltas, cuts sentences, and streams each through TTS 3 into a shared PCM buffer in
order, recording which samples belong to which sentence (so an interruption can trim the stored reply to what
was actually heard)."""
import re
import threading

import requests

from .higgs import BASE, HDR

TAG = re.compile(r"<\|[a-z_]+:[a-z_]+\|>")
LEAD = re.compile(r"^\s*((?:<\|(?:emotion|style|prosody):[a-z_]+\|>\s*)+)")
SENT_END = re.compile(r"(?<=[.!?…])\s+|(?<=[.!?…])$")
FIRST_CUT = re.compile(r"(?<=[,;:—])\s+")
# TTS 3 realizes a sound effect best when the written sound follows the tag (docs: "<|sfx:laughter|>Haha")
SOUNDS = {"laughter": "Haha,", "sigh": "Uh,", "humming": "Hmm,", "cough": "Ahem,", "sniff": "Sff,"}
SFX = re.compile(r"<\|sfx:([a-z_]+)\|>\s*")

EXPRESSIVE_RULES = (
    " Your words are spoken by a voice engine that understands inline tags; use them the way a real voice "
    "behaves, sparingly and never in every reply: <|sfx:laughter|>Haha when something is actually funny, "
    "<|sfx:sigh|>Uh for mild frustration or tiredness, <|sfx:humming|>Hmm when thinking. You may start a reply "
    "with one emotion tag when your mood is clear, e.g. <|emotion:amusement|>, <|emotion:surprise|>, "
    "<|emotion:sadness|>, <|emotion:enthusiasm|>. At most one sound tag per reply. Never mention the tags."
)


def voiced(s):
    """Insert the written sound after a bare sfx tag, unless the model already wrote one."""
    def fix(m):
        sound = SOUNDS.get(m.group(1))
        rest = s[m.end():m.end() + 6].lower()
        if not sound or rest.startswith(sound[:2].lower()):
            return m.group(0).rstrip()
        return f"<|sfx:{m.group(1)}|>{sound} "
    return SFX.sub(fix, s)


def spoken(text):
    """The words without tags (for transcripts and context)."""
    return re.sub(r"\s+", " ", TAG.sub("", text)).strip()


class SentenceSpeaker:
    def __init__(self, voice, audio, on_done, sr=24000):
        self.voice, self.audio, self.on_done, self.sr = voice, audio, on_done, sr
        self.buf = ""
        self.lead = ""          # leading emotion/style tags carry over to every sentence of the reply
        self.segments = []      # (start_sample, end_sample, sentence_text)
        self.queue = []
        self.closed = False
        self.cancelled = False
        self.cv = threading.Condition()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def feed(self, delta):
        self.buf += delta
        if not self.segments and not self.queue and len(spoken(self.buf).split()) >= 5:
            # speak the first clause as soon as it's complete: the first audio arrives sooner
            first = FIRST_CUT.split(self.buf, maxsplit=1)
            if len(first) == 2 and not SENT_END.search(first[0]):
                self._push(first[0])
                self.buf = first[1]
        parts = SENT_END.split(self.buf)
        for s in parts[:-1]:
            self._push(s)
        self.buf = parts[-1]

    def close(self):
        if self.buf.strip():
            self._push(self.buf)
        self.buf = ""
        with self.cv:
            self.closed = True
            self.cv.notify()

    def cancel(self):
        with self.cv:
            self.cancelled, self.closed = True, True
            self.queue.clear()
            self.cv.notify()

    def _push(self, s):
        s = s.strip()
        if not spoken(s):
            return
        m = LEAD.match(s)
        if m and not self.lead:
            self.lead = m.group(1).strip()
        elif self.lead and not m:
            s = self.lead + s
        with self.cv:
            self.queue.append(voiced(s))
            self.cv.notify()

    def _run(self):
        while True:
            with self.cv:
                while not self.queue and not self.closed:
                    self.cv.wait()
                if self.cancelled or (not self.queue and self.closed):
                    break
                s = self.queue.pop(0)
            start = len(self.audio) // 2
            try:
                with requests.post(f"{BASE}/audio/speech", headers=HDR, stream=True, timeout=60, json={
                        "model": "higgs-tts-3", "input": s, "voice": self.voice, "response_format": "pcm",
                        "stream": True}) as r:
                    r.raise_for_status()
                    odd = b""
                    for chunk in r.iter_content(4096):
                        if self.cancelled:
                            break
                        chunk = odd + chunk
                        cut = len(chunk) - len(chunk) % 2
                        self.audio.extend(chunk[:cut])
                        odd = chunk[cut:]
            except requests.RequestException:
                pass
            # a short breath between sentences
            self.audio.extend(bytes(int(0.12 * self.sr) * 2))
            self.segments.append((start, len(self.audio) // 2, spoken(s)))
        self.on_done()

    def heard_text(self, samples_played):
        """Sentences fully or mostly heard by the listener, for trimming the stored reply."""
        return " ".join(t for a, b, t in self.segments if a + 0.6 * (b - a) <= samples_played)
