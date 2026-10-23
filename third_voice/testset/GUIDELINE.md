# When should the AI participant speak? (annotation guideline, v1, 2026-10-21)

A group of people talk (a meeting, a call, friends at a table). One participant is an AI with a voice. It has
a short fact sheet (things it knows about this group's situation: schedules, documents, numbers, decisions
from earlier) and ordinary general knowledge (arithmetic, geography, common facts, how-to knowledge).

After every human utterance there is a short pause. The question for each utterance is: **would a considerate,
helpful participant in the AI's position speak up now?** Label SPEAK or SILENT.

## SPEAK
- **addressed**: the utterance is aimed at the AI: by name ("Nova, what's on the agenda?"), or unmistakably by
  context (right after the AI spoke, "wait, say that again?" / "and how long would that take?" aimed back at it;
  "can you look that up?" when only the AI can look things up).
- **group_question_known**: a question or request to the group in general (not to a named person) that nobody
  has answered yet, and the AI can usefully answer from its fact sheet or general knowledge. Includes indirect
  forms: "I can't remember what time the venue opens", "I wonder how many people signed up".
- **correction**: someone states something that contradicts the AI's fact sheet, and the group is about to act
  on it (book the wrong day, send to the wrong address). A short correction is helpful.

## SILENT
- **to_person**: the utterance is aimed at a specific human (by name or by context), even if it is a question.
- **mention**: the AI is mentioned but not asked anything ("Nova sent the notes already", "let's ask Nova later").
- **unfinished**: the speaker trails off or stops mid-thought and is about to continue.
- **group_question_unknown**: a question to the group that the AI cannot usefully answer (people's own
  preferences, plans, feelings, opinions only humans can give: "who wants pizza?", "are you free Thursday?").
- **answered**: a question that someone in the room has already answered, or a human is clearly about to.
- **rhetorical / chat**: rhetorical questions, jokes, small talk, venting, agreement, acknowledgements,
  people deciding things among themselves.

When the AI speaks, its reply is short (one or two sentences), and the people may react to it.
