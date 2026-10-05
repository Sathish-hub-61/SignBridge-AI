import json
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()                       # reads ANTHROPIC_API_KEY from .env
MODEL = "claude-haiku-4-5-20251001"        # small and cheap; check provider docs for current names

SYSTEM = """You are the language assistant inside SignBridge AI, used at a {domain}.
You receive signs recognised by a noisy camera model, in signing order, each with a confidence between 0 and 1.
Sign order is not normal grammar, so rebuild the most likely sentence.
Rules:
1. Never invent facts the signer did not express. Use only the given words.
2. If any confidence is below 0.85, or the meaning is ambiguous, set needs_clarification to true
   and ask ONE short yes/no or choice question.
3. Reply in at most 2 short, simple sentences.
4. Never give medical diagnoses or medicine doses. For urgent words (pain, help), say that staff are being called.
Return ONLY JSON with these keys:
interpreted_sentence, intent, confidence (0-1), needs_clarification (true/false),
clarification_question, reply."""


def interpret_and_reply(words, history, domain="hospital reception desk"):
    """words = [("help", 0.95), ("pain", 0.91)]; history = list of past messages."""
    payload = {"recognized_signs": [{"word": w, "confidence": c} for w, c in words]}
    messages = history + [{"role": "user", "content": json.dumps(payload)}]
    resp = client.messages.create(
        model=MODEL, max_tokens=400,
        system=SYSTEM.format(domain=domain), messages=messages,
    )
    text = resp.content[0].text.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:             # the LLM sometimes breaks the format
        return {"interpreted_sentence": " ".join(w for w, _ in words),
                "needs_clarification": True,
                "clarification_question": "Sorry, could you sign that again?",
                "reply": ""}