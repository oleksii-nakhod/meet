"""Knowledge base (F5): parsing, chunking, search, the fact sheet, ingestion and the upload API."""
import asyncio
import io
import json

import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kb import KnowledgeBase, Library, terms  # noqa: E402
from kb.parsers import chunk, parse  # noqa: E402

LEASE_PAGES = [
    "Residential Lease. Landlord: Mrs. Okafor. Tenants: Alex, Jordan and Mei.",
    "Section 3. Rent is $2,400 a month, due on the 1st of each month. A late fee of $50 applies after the 5th.",
    "Section 7. No pets without the landlord's written approval. Section 9. Guests staying more than 7 nights "
    "in a row need advance notice.",
]


def make_pdf(pages: list[str]) -> bytes:
    """A minimal valid PDF with one line of Helvetica text per page."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 9 Tf 20 700 Td ({safe}) Tj ET"
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        content = len(objs)
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content} 0 R "
                    f"/Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = io.BytesIO(), []
    out.write(b"%PDF-1.4\n")
    for i, o in enumerate(objs):
        offsets.append(out.tell())
        out.write(f"{i + 1} 0 obj\n{o}\nendobj\n".encode())
    xref = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


def make_docx(paras: list[str], table: list[list[str]]) -> bytes:
    import docx
    d = docx.Document()
    for p in paras:
        d.add_paragraph(p)
    t = d.add_table(rows=len(table), cols=len(table[0]))
    for r, row in enumerate(table):
        for c, v in enumerate(row):
            t.cell(r, c).text = v
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def test_parsers():
    pages = parse("lease.pdf", make_pdf(LEASE_PAGES))
    assert [p for p, _ in pages] == [1, 2, 3] and "due on the 1st" in pages[1][1]
    text = parse("notes.docx", make_docx(["Trash goes out Monday."], [["Chore", "Who"], ["Kitchen", "Alex"]]))[0][1]
    assert "Trash goes out Monday." in text and "Kitchen | Alex" in text
    assert "Who: Alex" in parse("chores.csv", b"Chore,Who\nKitchen,Alex\n")[0][1]
    assert parse("a.md", b"# Title\n\nSome   text")[0][1] == "# Title\n\nSome text"
    html = b"<html><head><title>T</title><script>x()</script></head><body><p>Rent: $2,400</p></body></html>"
    assert parse("p.html", html)[0][1] == "Rent: $2,400"
    with pytest.raises(ValueError):
        parse("evil.exe", b"MZ")


def test_chunks_keep_their_page_and_overlap():
    long = " ".join(f"Sentence number {i} about the lease." for i in range(200))
    pieces = chunk([(1, "Intro."), (2, long)], size=600, overlap=100)
    assert pieces[0] == (1, "Intro.") and all(p == 2 for p, _ in pieces[1:]) and len(pieces) > 5
    assert all(len(t) <= 700 for _, t in pieces)


def test_search_and_facts(tmp_path):
    kb = KnowledgeBase(tmp_path / "c1", "c1")
    doc = kb.add_file("lease.pdf", make_pdf(LEASE_PAGES))
    asyncio.run(kb.ingest(doc["id"]))                     # no models: indexed, no fact extraction
    d = kb.docs()[0]
    assert d["status"] == "ready" and d["pages"] == 3 and d["chunks"] == 3
    hits = kb.search("when is the rent due?")
    assert hits and hits[0]["page"] == 2 and hits[0]["doc"] == "lease.pdf"
    assert kb.search("pets allowed?")[0]["page"] == 3
    assert kb.search("") == [] and kb.search('"; drop table') == []
    kb.set_facts(["Rent is due on the 1st.", "Quiet hours start at 11 pm."])
    assert [f["origin"] for f in kb.facts()] == ["host", "host"]
    assert kb.relevant_facts("quiet hours", k=1)[0]["text"] == "Quiet hours start at 11 pm."
    kb.delete(doc["id"])
    assert kb.docs() == [] and kb.search("rent") == []


class FactModels:
    async def chat(self, system, user, **kw):
        assert "extract hard facts" in system and "[page 2]" in user
        return json.dumps([{"fact": "Rent is $2,400 a month, due on the 1st.", "page": 2},
                           {"fact": "No pets without the landlord's written approval.", "page": 3}])


def test_ingestion_extracts_facts_with_sources(tmp_path):
    kb = KnowledgeBase(tmp_path / "c2", "c2")
    doc = kb.add_file("lease.pdf", make_pdf(LEASE_PAGES))
    asyncio.run(kb.ingest(doc["id"], FactModels()))
    facts = kb.facts()
    assert [f["page"] for f in facts] == [2, 3] and all(f["doc"] == "lease.pdf" for f in facts)
    assert kb.docs()[0]["facts"] == 2
    kb.set_facts([facts[0]["text"], "A new host fact."])     # an edit keeps the source of unchanged facts
    assert [(f["origin"], f["page"]) for f in kb.facts()] == [("extracted", 2), ("host", None)]


def test_scanned_pdf_reports_an_error(tmp_path):
    kb = KnowledgeBase(tmp_path / "c3", "c3")
    doc = kb.add_file("scan.pdf", make_pdf([""]))
    asyncio.run(kb.ingest(doc["id"]))
    assert kb.docs()[0]["status"] == "error" and "OCR" in kb.docs()[0]["error"]



def test_library_lists_collections(tmp_path):
    lib = Library(tmp_path / "lib")
    lib.get("team-a").set_facts(["x"])
    lib.get("../../etc").set_facts(["y"])                    # ids are sanitised
    assert {c["id"] for c in lib.list()} == {"team-a", "etc"}
    assert terms("When is the RENT due, $2,400?") == ["rent", "due", "2", "400"]
