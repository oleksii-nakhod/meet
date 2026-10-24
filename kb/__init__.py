"""Knowledge base (F5): documents -> chunks -> full-text index, plus a fact sheet the host can edit.

One collection per room by default (a room can reuse a saved collection). Storage: <kb_dir>/<collection>/kb.sqlite
and the original files beside it. Retrieval is SQLite FTS5 (BM25) over chunks; the Boson key used here has no
embedding model, so there is no vector index (the search() interface leaves room for one).
"""
from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from . import parsers

MAX_PAGES = 200
STOP = set("""a an the and or but if so to of in on at for with from by as is are was were be been it its this that these
those i you he she we they me him her us them my your our their do does did not no yes okay ok um uh like just yeah
right well oh what when where who how why which can could would should will about there here then than has have had
am any all some more most also into over under up down out off again only own same too very s t don now get got""".split())

SCHEMA = """
create table if not exists docs (id text primary key, name text, kind text, status text, error text,
  pages integer default 0, chunks integer default 0, facts integer default 0, added_by text, source text,
  created real);
create table if not exists chunks (id integer primary key, doc_id text, page integer, text text);
create virtual table if not exists chunks_fts using fts5(text, content='chunks', content_rowid='id',
  tokenize='porter unicode61');
create table if not exists facts (id integer primary key, text text, doc_id text, page integer, origin text,
  position integer);
"""

FACT_SYS = ("You extract hard facts from documents for a meeting assistant that must catch people stating them "
            "wrongly. Return ONLY a JSON array of objects {\"fact\": \"...\", \"page\": <page number>}. A fact is one "
            "self-contained sentence with a specific amount, date, time, deadline, name, number, place or rule, worded "
            "so it makes sense on its own (name the subject). At most 15 facts, the ones most likely to come up or be "
            "misremembered. No opinions, no summaries.")


def terms(text: str) -> list[str]:
    """Search terms, split the way the FTS5 tokenizer splits ("$2,400" -> "2", "400"); digits always count."""
    seen, out = set(), []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if (len(w) > 1 or w.isdigit()) and w not in STOP and w not in seen:
            seen.add(w)
            out.append(w)
    return out


class KnowledgeBase:
    def __init__(self, directory: Path, cid: str):
        self.dir, self.id = directory, cid
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "files").mkdir(exist_ok=True)
        self.db = sqlite3.connect(directory / "kb.sqlite", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.version = 0

    # ------------------------------------------------------------ adding sources
    def _new_doc(self, name: str, kind: str, added_by: str = "", source: str = "") -> dict:
        pages = self.db.execute("select coalesce(sum(pages), 0) from docs where status != 'error'").fetchone()[0]
        if pages >= MAX_PAGES:
            raise ValueError(f"this knowledge base already holds {MAX_PAGES} pages")
        doc = {"id": uuid.uuid4().hex[:10], "name": name[:200], "kind": kind, "status": "queued", "error": "",
               "added_by": added_by[:40], "source": source, "created": time.time()}
        self.db.execute("insert into docs (id, name, kind, status, error, added_by, source, created) values "
                        "(:id, :name, :kind, :status, :error, :added_by, :source, :created)", doc)
        self.db.commit()
        self.version += 1
        return doc

    def add_file(self, name: str, data: bytes, added_by: str = "") -> dict:
        ext = parsers.kind_of(name)
        if len(data) > parsers.MAX_BYTES:
            raise ValueError("file is larger than 20 MB")
        doc = self._new_doc(name, ext, added_by)
        (self.dir / "files" / f"{doc['id']}{ext}").write_bytes(data)
        return doc

    def add_url(self, url: str, added_by: str = "") -> dict:
        if not re.match(r"^https?://", url):
            raise ValueError("the link must start with http:// or https://")
        return self._new_doc(url, "url", added_by, source=url)

    def add_text(self, title: str, text: str, added_by: str = "") -> dict:
        doc = self._new_doc(title, ".txt", added_by)
        (self.dir / "files" / f"{doc['id']}.txt").write_text(text)
        return doc

    def delete(self, doc_id: str) -> None:
        ids = [r[0] for r in self.db.execute("select id from chunks where doc_id = ?", (doc_id,))]
        for i in ids:
            self.db.execute("insert into chunks_fts(chunks_fts, rowid, text) select 'delete', id, text from chunks "
                            "where id = ?", (i,))
        self.db.execute("delete from chunks where doc_id = ?", (doc_id,))
        self.db.execute("delete from facts where doc_id = ? and origin = 'extracted'", (doc_id,))
        self.db.execute("delete from docs where id = ?", (doc_id,))
        self.db.commit()
        for f in (self.dir / "files").glob(f"{doc_id}.*"):
            f.unlink()
        self.version += 1

    def _set(self, doc_id: str, **kw: Any) -> None:
        cols = ", ".join(f"{k} = :{k}" for k in kw)
        self.db.execute(f"update docs set {cols} where id = :id", {**kw, "id": doc_id})
        self.db.commit()
        self.version += 1

    # ------------------------------------------------------------ ingestion
    async def ingest(self, doc_id: str, models=None) -> None:
        row = self.db.execute("select * from docs where id = ?", (doc_id,)).fetchone()
        if row is None:
            return
        try:
            self._set(doc_id, status="parsing")
            if row["kind"] == "url":
                pages, title = await self._fetch(row["source"])
                if title:
                    self._set(doc_id, name=title[:200])
            else:
                path = next((self.dir / "files").glob(f"{doc_id}.*"))
                pages = await asyncio.to_thread(parsers.parse, path.name, path.read_bytes())
            pieces = parsers.chunk(pages)
            if not pieces:
                raise ValueError("no text found (a scanned PDF needs OCR first)")
            for page, text in pieces:
                cur = self.db.execute("insert into chunks (doc_id, page, text) values (?, ?, ?)", (doc_id, page, text))
                self.db.execute("insert into chunks_fts (rowid, text) values (?, ?)", (cur.lastrowid, text))
            self.db.commit()
            self._set(doc_id, status="indexed", pages=len(pages), chunks=len(pieces))
            if models is not None:
                self._set(doc_id, status="extracting facts")
                n = await self._extract_facts(doc_id, pieces, models)
                self._set(doc_id, facts=n)
            self._set(doc_id, status="ready")
        except Exception as e:  # noqa: BLE001
            self._set(doc_id, status="error", error=str(e)[:300])

    async def _fetch(self, url: str) -> tuple[list[tuple[int, str]], str]:
        import aiohttp
        import ssl
        import certifi
        TLS = ssl.create_default_context(cafile=certifi.where())
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=TLS)) as http:
            async with http.get(url, timeout=aiohttp.ClientTimeout(total=20), headers={"User-Agent": "Meet/2.0"}) as r:
                if r.status != 200:
                    raise ValueError(f"the link answered HTTP {r.status}")
                data = await r.read()
                ctype = r.headers.get("content-type", "")
        if len(data) > parsers.MAX_BYTES:
            raise ValueError("the page is larger than 20 MB")
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            return await asyncio.to_thread(parsers.parse_pdf, data), ""
        if "html" in ctype or not ctype:
            title, pages = parsers.parse_html(data)
            return pages, title
        return [(1, parsers.clean(data.decode("utf-8", "replace")))], ""

    async def _extract_facts(self, doc_id: str, pieces: list[tuple[int, str]], models) -> int:
        """Up to 6 requests per document, ~6000 characters each, run in parallel."""
        batches, cur, size = [], [], 0
        for page, text in pieces:
            if size + len(text) > 6000 and cur:
                batches.append(cur)
                cur, size = [], 0
            cur.append((page, text))
            size += len(text)
        if cur:
            batches.append(cur)

        async def one(batch):
            body = "\n\n".join(f"[page {p}]\n{t}" for p, t in batch)
            try:
                return await models.chat(FACT_SYS, body, max_tokens=900, temperature=0.0, timeout=40)
            except Exception:  # noqa: BLE001
                return ""
        texts = await asyncio.gather(*(one(b) for b in batches[:6]))
        known = {f["text"].lower() for f in self.facts()}
        n = 0
        pos = self.db.execute("select coalesce(max(position), 0) from facts").fetchone()[0]
        for text in texts:
            m = re.search(r"\[.*\]", text, re.S)
            try:
                items = json.loads(m.group(0)) if m else []
            except json.JSONDecodeError:
                items = []
            for it in items:
                fact = str(it.get("fact", "") if isinstance(it, dict) else it).strip()
                if len(fact) < 8 or fact.lower() in known:
                    continue
                known.add(fact.lower())
                pos += 1
                page = it.get("page") if isinstance(it, dict) and isinstance(it.get("page"), int) else None
                self.db.execute("insert into facts (text, doc_id, page, origin, position) values (?, ?, ?, ?, ?)",
                                (fact, doc_id, page, "extracted", pos))
                n += 1
        self.db.commit()
        self.version += 1
        return n

    # ------------------------------------------------------------ fact sheet
    def facts(self) -> list[dict]:
        rows = self.db.execute("select f.id, f.text, f.doc_id, f.page, f.origin, d.name as doc from facts f "
                               "left join docs d on d.id = f.doc_id order by position, f.id").fetchall()
        return [dict(r) for r in rows]

    def fact_texts(self) -> list[str]:
        return [f["text"] for f in self.facts()]

    def set_facts(self, texts: list[str]) -> None:
        """The host's edited list replaces the sheet. Facts kept word for word keep their source."""
        old = {f["text"]: f for f in self.facts()}
        self.db.execute("delete from facts")
        for i, t in enumerate(texts):
            o = old.get(t)
            self.db.execute("insert into facts (text, doc_id, page, origin, position) values (?, ?, ?, ?, ?)",
                            (t, o["doc_id"] if o else None, o["page"] if o else None,
                             o["origin"] if o else "host", i))
        self.db.commit()
        self.version += 1

    # ------------------------------------------------------------ retrieval
    def search(self, query: str, k: int = 5) -> list[dict]:
        ts = terms(query)
        if not ts:
            return []
        q = " OR ".join('"' + t.replace('"', "") + '"' for t in ts[:24])
        try:
            rows = self.db.execute(
                "select c.id, c.doc_id, c.page, c.text, d.name as doc, bm25(chunks_fts) as score from chunks_fts "
                "join chunks c on c.id = chunks_fts.rowid join docs d on d.id = c.doc_id "
                "where chunks_fts match ? order by score limit ?", (q, k)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [{"id": r["id"], "doc_id": r["doc_id"], "doc": r["doc"], "page": r["page"], "text": r["text"],
                 "score": round(-r["score"], 3)} for r in rows]

    def relevant_facts(self, query: str, k: int = 12) -> list[dict]:
        """The whole sheet if it is short, else the k facts sharing the most words with the query."""
        facts = self.facts()
        if len(facts) <= k:
            return facts
        q = set(terms(query))
        return sorted(facts, key=lambda f: -len(q & set(terms(f["text"]))))[:k]

    # ------------------------------------------------------------ state
    def docs(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("select * from docs order by created")]

    def state(self) -> dict:
        return {"collection": self.id, "docs": self.docs(), "facts": self.facts(), "version": self.version}

    def close(self) -> None:
        self.db.close()


class Library:
    """All collections under kb_dir."""

    def __init__(self, directory: Path):
        self.dir = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.open: dict[str, KnowledgeBase] = {}

    def get(self, cid: str) -> KnowledgeBase:
        cid = re.sub(r"[^A-Za-z0-9_-]", "", cid)[:64] or "default"
        if cid not in self.open:
            self.open[cid] = KnowledgeBase(self.dir / cid, cid)
        return self.open[cid]

    def drop(self, cid: str) -> None:
        """Stop holding a collection open (it stays on disk; get() opens it again). Not closed here: an ingest that
        is still running may hold it, and the connection closes when the last user lets go."""
        self.open.pop(re.sub(r"[^A-Za-z0-9_-]", "", cid)[:64] or "default", None)

    def list(self) -> list[dict]:
        out = []
        for d in sorted(self.dir.iterdir()):
            if (d / "kb.sqlite").exists():
                kb = self.get(d.name)
                out.append({"id": d.name, "docs": len(kb.docs()), "facts": len(kb.facts())})
        return out
