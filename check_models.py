"""One call per model with the exact settings the experiment uses: checks the key works and answers can be read."""

import time

from analysis import EMBED_MODEL
from telephone import REQUESTS_PER_MINUTE, make_caller, parse_answers, profile_to_text, survey_prompt

PROMPT = survey_prompt(profile_to_text([1, 1, 0, 1, 0, 1, 0, 0, 1]))

if __name__ == "__main__":
    caller = make_caller()
    for model in [m for m in REQUESTS_PER_MINUTE if "embedding" not in m]:
        start = time.monotonic()
        try:
            reply = caller.generate(model, PROMPT, temperature=1.0).text
            status = "ok, answers read" if parse_answers(reply) else f"answered, but unreadable: {reply[:60]!r}"
        except RuntimeError as e:
            status = f"FAILED: {e}"
        print(f"{model:24s} {time.monotonic() - start:5.1f}s  {status}")
    try:
        caller.embed(EMBED_MODEL, ["a calm person", "a quiet person"])
        print(f"{EMBED_MODEL:24s}   ok, embeddings work")
    except RuntimeError as e:
        print(f"{EMBED_MODEL:24s}   FAILED: {e}")
