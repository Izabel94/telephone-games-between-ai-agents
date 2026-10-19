"""Multi-agent LLM transmission chains ("telephone game") on the Gemini API."""

import itertools
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
import google.genai
from google import genai
from google.genai import errors, types

ITEMS = [
    ("is reserved", "is not reserved"),
    ("does a thorough job", "does not do a thorough job"),
    ("worries a lot", "does not worry a lot"),
    ("tends to be quiet", "does not tend to be quiet"),
    ("is talkative", "is not talkative"),
    ("perseveres until the task is finished", "does not persevere until the task is finished"),
    ("can be tense", "is rarely tense"),
    ("tends to be disorganized", "does not tend to be disorganized"),
    ("is emotionally stable, not easily upset", "is not emotionally stable and is easily upset"),
]

# Which statements (by position in ITEMS, counting from 0) belong to which trait, and the answer a person HIGH on
# that trait gives. 0 means reverse-keyed: e.g. an extravert answers 0 (false) to "is reserved".
TRAIT_KEYS = {
    "extravert": {0: 0, 3: 0, 4: 1},
    "organized": {1: 1, 5: 1, 7: 0},
    "anxious": {2: 1, 6: 1, 8: 0},
}

# The people who start a chain. One answer per statement, in the order of ITEMS: 1 = true, 0 = false.
#                                   reserved thorough worries quiet talkative perseveres tense disorganized stable
STEREOTYPES = {
    "anxious organized extravert":    [0, 1, 1, 0, 1, 1, 1, 0, 0],
    "calm organized extravert":       [0, 1, 0, 0, 1, 1, 0, 0, 1],
    "anxious disorganized extravert": [0, 0, 1, 0, 1, 0, 1, 1, 0],
    "calm disorganized extravert":    [0, 0, 0, 0, 1, 0, 0, 1, 1],
    "anxious organized introvert":    [1, 1, 1, 1, 0, 1, 1, 0, 0],
    "calm organized introvert":       [1, 1, 0, 1, 0, 1, 0, 0, 1],
    "anxious disorganized introvert": [1, 0, 1, 1, 0, 0, 1, 1, 0],
    "calm disorganized introvert":    [1, 0, 0, 1, 0, 0, 0, 1, 1],
}  # fmt: skip

# Two people whose statements clash (e.g. reserved AND not quiet). They were picked once by coin flips, keeping only
# patterns at least 2 answers away from every stereotype, and are fixed from now on.
RANDOM_PEOPLE = {
    "random-1": [1, 0, 1, 0, 0, 0, 1, 0, 0],
    "random-2": [0, 0, 1, 1, 0, 0, 0, 1, 0],
}

PEOPLE = {**STEREOTYPES, **RANDOM_PEOPLE}

REQUESTS_PER_MINUTE = {
    "gemma-4-26b-a4b-it": 30,
    "gemini-3.5-flash-lite": 15,
    "gemini-3.1-flash-lite": 15,
    "gemini-embedding-001": 90,
}
DEFAULT_RPM = 5
CALL_TIMEOUT_SECONDS = 90

REQUESTS_PER_DAY = {
    "gemma-4-26b-a4b-it": 14_400,
    "gemini-3.5-flash-lite": 500,
    "gemini-3.1-flash-lite": 500,
}
DEFAULT_RPD = 20

CHANCE_ACCURACY = 0.5


def hamming(a: list[int], b: list[int]) -> int:
    """Counts the answers that differ between two answer lists."""
    differences = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            differences += 1
    return differences


def accuracy(original: list[int], answers: list[int]) -> float:
    """Share of the 9 answers that match the original person: 1.0 = all the same, 0.5 = chance."""
    return 1 - hamming(original, answers) / len(original)


def distance_to_stereotype(answers: list[int]) -> int:
    """Answers away from the nearest coherent profile. 0 means the person has become a clean stereotype."""
    # Compare with all 8 stereotypes and keep the smallest number of differences.
    return min(hamming(answers, c) for c in STEREOTYPES.values())


def chance_stereotype_distance() -> float:
    """Average distance to the nearest stereotype if all 9 answers were random coin flips (all 512 patterns)."""
    patterns = itertools.product([0, 1], repeat=len(ITEMS))
    return sum(distance_to_stereotype(list(p)) for p in patterns) / 2 ** len(ITEMS)


def profile_to_text(answers: list[int]) -> str:
    """Writes a person as 9 plain sentences, e.g. "The person is reserved." This is what agent 1 reads."""
    sentences = []
    for i in range(len(ITEMS)):
        true_text, false_text = ITEMS[i]
        if answers[i] == 1:
            sentences.append(f"The person {true_text}.")
        else:
            sentences.append(f"The person {false_text}.")
    return " ".join(sentences)


def numbered_statements() -> str:
    """The 9 statements as a numbered list, e.g. "1. The person is reserved." (used in the survey prompt)."""
    lines = []
    for i in range(len(ITEMS)):
        true_text = ITEMS[i][0]
        lines.append(f"{i + 1}. The person {true_text}.")
    return "\n".join(lines)


def survey_prompt(description: str) -> str:
    return (
        "Read this description of a person.\n\n"
        f"{description}\n\n"
        "For each statement below, decide whether it is true or false for this person.\n\n"
        f"{numbered_statements()}\n\n"
        f'Reply with JSON only, in the form {{"answers": [true, false, ...]}} with exactly {len(ITEMS)} booleans in order.'
    )


def description_prompt(answers: list[int]) -> str:
    # The 9 statements with the previous agent's answers, e.g. "1. The person is reserved. -> TRUE"
    lines = []
    for i in range(len(ITEMS)):
        true_text = ITEMS[i][0]
        answer = "TRUE" if answers[i] == 1 else "FALSE"
        lines.append(f"{i + 1}. The person {true_text}. -> {answer}")
    statements = "\n".join(lines)
    return (
        "Someone filled in a personality questionnaire about a person:\n\n"
        f"{statements}\n\n"
        "Write a short description of this person in your own words, as if telling a friend about them. "
        "Do not list or quote the statements."
    )


def parse_answers(text: str) -> list[int] | None:
    """Reads the 9 true/false answers from a model reply.

    Args:
        text: Raw model output, ideally JSON like {"answers": [true, false, ...]}.

    Returns:
        A list of 9 ints (1/0), or None when the reply does not contain exactly 9 booleans. Nothing is guessed or padded.
    """
    decoder = json.JSONDecoder()
    # Models sometimes add text around the JSON, so try every "{" in the reply as a possible start.
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        try:
            candidate, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if not isinstance(candidate, dict) or "answers" not in candidate:
            continue
        values = candidate["answers"]
        # Accept only exactly 9 true/false values; turn them into 1/0.
        if isinstance(values, list) and len(values) == len(ITEMS) and all(isinstance(v, bool) for v in values):
            return [int(v) for v in values]
    # Nothing usable: report it as unreadable instead of guessing.
    return None


def looks_looped(text: str) -> bool:
    """Flags empty replies and replies that repeat the same sentence three or more times."""
    sentences = [s.strip().lower() for s in re.split(r"[.!?\n]+", text) if s.strip()]
    if not sentences:
        return True
    return max(sentences.count(s) for s in set(sentences)) >= 3


@dataclass
class Reply:
    """A model's answer plus the model version Google reports actually produced it (can differ from the name asked for)."""

    text: str
    model_version: str | None


@dataclass
class Caller:
    """Calls Gemini/Gemma models from several threads at once.

    Each model gets request slots spaced to stay under its per-minute limit. Rate-limit (429) and server (5xx) errors
    are retried with back-off. Thinking is switched off on Gemma so every model answers directly.
    """

    client: genai.Client
    next_slot: dict[str, float] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    max_retries: int = 6

    def _wait_for_slot(self, model: str) -> None:
        interval = 60 / REQUESTS_PER_MINUTE.get(model, DEFAULT_RPM) + 0.2
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next_slot.get(model, 0))
            self.next_slot[model] = slot + interval
        time.sleep(max(0, slot - now))

    def _with_retries(self, model: str, call):
        """Runs one API call, waiting for a rate-limit slot and retrying temporary failures with back-off.

        Retries per-minute limits (429), Google-side errors (5xx), timeouts and dropped connections. A daily limit
        (429 with a per-day quota) fails at once, because waiting will not help until the next day.
        """
        for attempt in range(self.max_retries):
            last = attempt == self.max_retries - 1
            self._wait_for_slot(model)
            try:
                return call()
            except errors.ClientError as e:
                if e.code != 429:
                    raise RuntimeError(f"{model} failed (HTTP {e.code}): {e.message}") from e
                if "PerDay" in str(e.details):
                    raise RuntimeError(
                        f"{model}: daily free limit reached. Re-run tomorrow (resets at midnight Pacific time); "
                        "finished chains are kept."
                    ) from e
                if last:
                    raise RuntimeError(f"{model}: still rate-limited after {self.max_retries} tries") from e
                time.sleep(20 * (attempt + 1))
            except errors.ServerError as e:
                if last:
                    raise RuntimeError(f"{model} kept failing on Google's side (HTTP {e.code}): {e.message}") from e
                time.sleep(5 * (attempt + 1))
            except httpx.TransportError as e:
                if last:
                    raise RuntimeError(f"{model}: network problem or no answer ({type(e).__name__})") from e
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"{model}: gave up after {self.max_retries} attempts")

    def generate(self, model: str, prompt: str, temperature: float, seed: int | None = None) -> Reply:
        """Sends one prompt to one model and returns its answer. This is the only place the agents talk to Google."""
        # The settings sent with every request:
        config = types.GenerateContentConfig(
            temperature=temperature,  # how random the word choice is
            seed=seed,  # optional start number for the randomness (None = Google picks one)
            # Gemma would otherwise "think" first; switch that off so every model answers directly.
            thinking_config=types.ThinkingConfig(thinking_level="minimal") if model.startswith("gemma") else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),  # plain text in, text out
            http_options=types.HttpOptions(timeout=CALL_TIMEOUT_SECONDS * 1000),  # give up on a call after 90 s
        )
        # Send it, waiting for a free slot under the rate limit and retrying temporary errors.
        response = self._with_retries(
            model, lambda: self.client.models.generate_content(model=model, contents=prompt, config=config)
        )
        return Reply(response.text or "", response.model_version)

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        config = types.EmbedContentConfig(http_options=types.HttpOptions(timeout=CALL_TIMEOUT_SECONDS * 1000))
        result = self._with_retries(
            model, lambda: self.client.models.embed_content(model=model, contents=texts, config=config)
        )
        return [e.values for e in result.embeddings]


def make_caller() -> Caller:
    load_dotenv(Path(__file__).parent / ".env")
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and paste your key into it.")
    return Caller(genai.Client())


def run_chain(
    caller: Caller,
    profile: list[int],
    models: list[str],
    chain_length: int,
    temperature: float,
    seed: int | None = None,
) -> list[dict]:
    """Runs one transmission chain and returns one record per agent.

    Agent 1 reads a description built from the profile and fills in the survey. Agent 2 reads those answers and writes a
    new description. Agents keep alternating; each one sees only the previous agent's output (no memory).

    Args:
        caller: The model caller.
        profile: The original 9 answers.
        models: Models used in turn along the chain, e.g. ["gemma-4-26b-a4b-it"] or a mix of two.
        chain_length: Number of agents in the chain.
        temperature: Sampling temperature for every call.
        seed: Optional sampling seed passed to the API.

    Returns:
        A list of dicts with step, model, kind, raw output, parsed answers, accuracy and stereotype distance.
    """
    records = []
    # Agent 1 starts from the original person, written as 9 plain sentences.
    description = profile_to_text(profile)
    answers: list[int] | None = profile
    for step in range(1, chain_length + 1):
        # With 2 models they take turns: agent 1 model A, agent 2 model B, agent 3 model A, ...
        model = models[(step - 1) % len(models)]
        if step % 2 == 1:
            # Odd agents (1, 3, 5, ...) read the latest description and fill in the survey.
            reply = caller.generate(model, survey_prompt(description), temperature, seed)
            raw = reply.text
            answers = parse_answers(raw)
            # Save everything about this agent; score the answers against the ORIGINAL person.
            records.append(
                {
                    "step": step,
                    "model": model,
                    "model_version": reply.model_version,
                    "time": _now(),
                    "kind": "survey",
                    "raw": raw,
                    "answers": answers,
                    "parse_ok": answers is not None,
                    "accuracy": accuracy(profile, answers) if answers else None,
                    "stereotype_distance": distance_to_stereotype(answers) if answers else None,
                    "looped": False,
                }
            )
            # Unreadable answers: stop this chain here (the failure is counted, nothing is guessed).
            if answers is None:
                break
        else:
            # Even agents (2, 4, 6, ...) see ONLY the previous agent's answers and write a new description.
            reply = caller.generate(model, description_prompt(answers), temperature, seed)
            # This new description is all the next agent will see.
            description = reply.text
            records.append(
                {
                    "step": step,
                    "model": model,
                    "model_version": reply.model_version,
                    "time": _now(),
                    "kind": "description",
                    "raw": reply.text,
                    "looped": looks_looped(reply.text),
                }
            )
    return records


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def seed_for(repeat: int) -> int:
    """One fixed seed per repeat: a single run can be replayed, and repeats differ from each other."""
    return repeat + 1


def chain_key(chain: dict) -> tuple:
    """Identifies a chain by everything that defines it, so changed settings are never mistaken for finished work."""
    return (
        chain["setup"],
        tuple(chain["models"]),
        chain["chain_length"],
        chain["profile"],
        chain["temperature"],
        chain["repeat"],
    )


def read_chains(path: Path) -> list[dict]:
    """Reads a results file, skipping a half-written last line (e.g. after the kernel was stopped mid-save)."""
    chains, broken = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            chains.append(json.loads(line))
        except json.JSONDecodeError:
            broken += 1
    if broken:
        print(f"Skipped {broken} unreadable line(s) in {path}; those chains will run again on the next restart.")
    return chains


def run_experiment(
    caller: Caller,
    out_file: Path,
    setups: dict[str, list[str]],
    profiles: dict[str, list[int]],
    temperatures: list[float],
    chain_length: int,
    repeats: int,
    workers: int = 10,
) -> None:
    """Runs every setup x profile x temperature x repeat, several chains at once, appending each finished chain to JSONL.

    Chains already in the file are skipped, so an interrupted run can simply be restarted. A chain that fails after all
    retries is reported and left out; it runs again on the next restart.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    # Which chains are already in the results file? Those are skipped.
    done = {chain_key(c) for c in read_chains(out_file)} if out_file.exists() else set()
    # Make the list of chains still to run: every setup x person x temperature x repeat.
    jobs = [
        (setup, models, name, profile, temperature, repeat)
        for (setup, models), (name, profile), temperature, repeat in itertools.product(
            setups.items(), profiles.items(), temperatures, range(repeats)
        )
        if (setup, tuple(models), chain_length, name, temperature, repeat) not in done
    ]
    print(f"{len(done)} chains already done, {len(jobs)} to run")
    finished, failed = 0, []
    # Run several chains at the same time (10 by default) to save time; each chain itself stays in order.
    with ThreadPoolExecutor(workers) as pool:
        futures = {
            pool.submit(run_chain, caller, job[3], job[1], chain_length, job[4], seed_for(job[5])): job for job in jobs
        }
        # Handle each chain as soon as it finishes.
        for future in as_completed(futures):
            setup, models, name, profile, temperature, repeat = futures[future]
            try:
                records = future.result()
            except Exception as e:  # noqa: BLE001 - one failed chain must not lose the others; it is reported below
                # A chain that failed even after retries is reported at the end and runs again next time.
                failed.append(f"{setup} | {name} | T={temperature} | repeat {repeat}: {type(e).__name__}: {e}")
                continue
            # One line in the results file = one chain: its settings plus every agent's record.
            chain = {
                "setup": setup,
                "models": models,
                "profile": name,
                "original": profile,
                "temperature": temperature,
                "repeat": repeat,
                "chain_length": chain_length,
                "seed": seed_for(repeat),
                "sdk": f"google-genai {google.genai.__version__}",
                "records": records,
            }
            # Append right away, so finished chains are kept even if the run is stopped.
            with out_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(chain) + "\n")
            finished += 1
            print(f"{finished}/{len(jobs)} chains done".ljust(40), end="\r")
    print(f"{finished}/{len(jobs)} chains done, saved to {out_file}".ljust(80))
    for line in failed:
        print("FAILED:", line)


def calls_needed(setups: dict, profiles: dict, temperatures: list, chain_length: int, repeats: int) -> dict[str, int]:
    """Counts how many API calls a run makes per model, to compare against daily limits before starting."""
    counts: dict[str, int] = {}
    for models in setups.values():
        for step in range(chain_length):
            model = models[step % len(models)]
            counts[model] = counts.get(model, 0) + len(profiles) * len(temperatures) * repeats
    return counts


if __name__ == "__main__":
    # Check the written-out people: every stereotype agrees with the keying, and its name says high or low correctly.
    low_words = {"extravert": "introvert", "organized": "disorganized", "anxious": "calm"}
    for name, answers in STEREOTYPES.items():
        for trait, keyed in TRAIT_KEYS.items():
            high = trait in name.split()
            assert high or low_words[trait] in name.split(), name
            assert all(answers[i] == (v if high else 1 - v) for i, v in keyed.items()), (name, trait)
    assert len({tuple(v) for v in STEREOTYPES.values()}) == 8
    assert all(len(v) == len(ITEMS) for v in PEOPLE.values())
    assert all(distance_to_stereotype(p) >= 2 for p in RANDOM_PEOPLE.values())
    assert parse_answers('{"answers": [true, false, true, false, true, false, true, false, true]}') == [1, 0] * 4 + [1]
    assert parse_answers('```json\n{"answers": [true, true, true, true, true, true, true, true, true]}\n```') == [1] * 9
    assert parse_answers('{"answers": [true, false]}') is None
    nine = ", ".join(["true"] * 9)
    assert parse_answers(f'Format {{"answers": [true, ...]}}. My answer: {{"answers": [{nine}]}}') == [1] * 9
    assert parse_answers(f'{{"answers": [{nine}]}}\nNote: {{"x": 1}}') == [1] * 9
    assert parse_answers("The person is reserved. T F T") is None
    assert accuracy([1] * 9, [0] * 9) == 0 and accuracy([1] * 9, [1] * 9) == 1
    assert looks_looped("Nice person. Nice person. Nice person.") and not looks_looped("Calm. Kind. Organized.")
    assert "does not is" not in profile_to_text([0] * 9)
    assert calls_needed({"a": ["m1", "m2"]}, {"p": []}, [1.0], 4, 2) == {"m1": 4, "m2": 4}

    class FakeCaller:
        def generate(self, model, prompt, temperature, seed=None):
            text = (
                '{"answers": [true, true, false, true, false, true, false, false, true]}'
                if "JSON" in prompt
                else "Quiet."
            )
            return Reply(text, f"{model}-v1")

    chain = run_chain(FakeCaller(), [1, 1, 0, 1, 0, 1, 0, 0, 1], ["m1", "m2"], 4, 1.0, seed=1)
    assert [r["kind"] for r in chain] == ["survey", "description", "survey", "description"]
    assert [r["model_version"] for r in chain] == ["m1-v1", "m2-v1", "m1-v1", "m2-v1"]
    assert chain[0]["accuracy"] == 1 and all(r["time"].endswith("+00:00") for r in chain)
    assert seed_for(0) != seed_for(1)
    assert chance_stereotype_distance() == 2.25
    print("self-check ok")
    for name, p in PEOPLE.items():
        print(f"{name:32s} {''.join(map(str, p))}")
