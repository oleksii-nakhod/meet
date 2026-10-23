"""Unit tests for the speak/silent rule and the heuristics (no API). Run: python -m pytest -q test_tv.py"""
from tv_eval import dec_rule, heuristic, heuristic2, parse_json


def d(to, ask=False, answered=False, can=False, wrong=False):
    return {"to": to, "ask": ask, "answered": answered, "can": can, "wrong": wrong}


def test_rule_v2():
    assert dec_rule(d("me", ask=True), "Nova", 2)
    assert dec_rule(d("Nova", ask=True), "Nova", 2)  # Higgs sometimes writes the AI's name instead of "me"
    assert not dec_rule(d("me"), "Nova", 2)  # "okay, thanks" needs no reply
    assert dec_rule(d("group", ask=True, can=True), "Nova", 2)
    assert not dec_rule(d("group", ask=True, can=True, answered=True), "Nova", 2)
    assert not dec_rule(d("group", ask=True, can=False), "Nova", 2)
    assert not dec_rule(d("person", ask=True, can=True), "Nova", 2)
    assert not dec_rule(d("self"), "Nova", 2)
    assert dec_rule(d("person", wrong=True), "Nova", 2)  # a correction overrides who it was aimed at
    assert not dec_rule(None, "Nova", 2)


def test_parse_json():
    assert parse_json('Sure: {"to": "me", "ask": true}') == {"to": "me", "ask": True}
    assert parse_json("SPEAK") is None
    assert parse_json("{broken") is None


def row(text, people=("Ana", "Ben"), ai="Nova", speaker="Ana"):
    return {"current_turn": {"speaker": speaker, "text": text}, "target_speaker": ai, "people": list(people)}


def test_heuristics():
    assert heuristic(row("Nova, the agenda"))
    assert heuristic(row("Ben, can you do it?"))
    assert not heuristic2(row("Ben, can you do it?"))  # a question naming a person in the room
    assert heuristic2(row("When does it start?"))


def test_retrieval_keeps_the_relevant_fact():
    from retrieve import top_facts
    facts = [f"Unrelated fact number {i} about gardening." for i in range(40)]
    facts.insert(17, "The walk-in fridge repair technician is scheduled for Monday at 9 am.")
    r = {"facts": facts, "context_turns": [{"speaker": "A", "text": "The walk-in is making that noise again."}],
         "current_turn": {"speaker": "B", "text": "When is the repair guy coming?"}}
    top = top_facts(r, 8)
    assert len(top) == 8 and facts[17] in top


def test_repeat_guard():
    from app import repeats
    said = ["We can't increase the price. The selling price is fixed at 25 euros."]
    assert repeats("We can't increase the price. The selling price is fixed at 25 euros.", said)
    assert repeats("We can't increase the price, the selling price is fixed at 25 euros", said)
    assert not repeats("That's way over our twelve fifty budget.", said)
