"""Loads chain results and draws the workshop charts."""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from telephone import CHANCE_ACCURACY, STEREOTYPES, Caller, chain_key, hamming, read_chains

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
EMBED_MODEL = "gemini-embedding-001"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_MUTED,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.frameon": False,
        "font.size": 11,
    }
)


def load_chains(path: Path) -> list[dict]:
    return read_chains(path)


def survey_table(chains: list[dict]) -> pd.DataFrame:
    """One row per survey answer along every chain, plus step 0 = the original profile."""
    rows = []
    for c in chains:
        # Settings that every row of this chain shares.
        base = {k: c[k] for k in ("setup", "profile", "temperature", "repeat")}
        base["profile_type"] = "random" if c["profile"].startswith("random") else "coherent"
        # Row for step 0: the original person (100% accurate by definition).
        rows.append(
            {
                **base,
                "step": 0,
                "accuracy": 1.0,
                "stereotype_distance": _distance(c["original"]),
                "parse_ok": True,
                "answers": "".join(map(str, c["original"])),
            }
        )
        # One more row for every survey agent; description agents have no answers to score.
        for r in c["records"]:
            if r["kind"] == "survey":
                rows.append(
                    {
                        **base,
                        "step": r["step"],
                        "accuracy": r["accuracy"],
                        "stereotype_distance": r["stereotype_distance"],
                        "parse_ok": r["parse_ok"],
                        "answers": "".join(map(str, r["answers"])) if r["answers"] else None,
                    }
                )
    return pd.DataFrame(rows)


def records_table(chains: list[dict]) -> pd.DataFrame:
    """One row per agent in every chain, with the full model output: the complete data for your own analysis.

    Columns: chain settings (setup, profile, temperature, repeat, seed), step, kind (survey or description), model
    asked for, model_version that answered, time (UTC), raw output, parsed answers, accuracy, stereotype_distance,
    parse_ok and looped. Settings a run did not record (older result files) are left empty.
    """
    rows = []
    for c in chains:
        base = {k: c.get(k) for k in ("setup", "profile", "temperature", "repeat", "seed", "chain_length", "sdk")}
        base["original"] = "".join(map(str, c["original"]))
        for r in c["records"]:
            row = {**base, **{k: v for k, v in r.items() if k != "answers"}}
            row["answers"] = "".join(map(str, r["answers"])) if r.get("answers") else None
            rows.append(row)
    table = pd.DataFrame(rows)
    for column in ("seed", "model_version", "time"):
        if column not in table:
            table[column] = None
    return table


def _distance(answers: list[int]) -> int:
    return min(hamming(answers, c) for c in STEREOTYPES.values())


def _line_chart(df: pd.DataFrame, value: str, group: str, ylabel: str, title: str, ax=None):
    ax = ax or plt.subplots(figsize=(8, 4.5))[1]
    # Average the value over all chains, separately for each group (e.g. each setup) and each step.
    means = df.groupby([group, "step"])[value].mean().unstack(0)
    if len(means.columns) > len(SERIES):
        raise ValueError(
            f"{len(means.columns)} groups in '{group}' but only {len(SERIES)} colours. Filter first, "
            f"e.g. df[df['profile_type'] == 'random'], or use a column with fewer values."
        )
    # One line per group: x = chain step, y = the average.
    for color, name in zip(SERIES, means.columns):
        ax.plot(means.index, means[name], color=color, marker="o", label=name)
    ax.set_xticks(means.index)
    ax.set_xlabel("chain step (0 = original person)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color=INK)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    return ax


def plot_accuracy(df: pd.DataFrame, group: str = "setup"):
    """Share of the 9 traits still correct at each survey step, with the coin-flip line."""
    ax = _line_chart(
        df, "accuracy", group, "traits still correct (share of 9)", f"Accuracy along the chain, by {group}"
    )
    # Dashed line at 0.5: what random answers would score.
    ax.axhline(CHANCE_ACCURACY, color=INK_MUTED, linestyle="--", linewidth=1)
    ax.text(df["step"].max(), CHANCE_ACCURACY - 0.06, "chance (coin flip)", color=INK_MUTED, fontsize=9, ha="right")
    ax.set_ylim(0, 1.05)
    return ax


def plot_stereotype(df: pd.DataFrame, group: str = "profile_type"):
    """Distance to the nearest coherent profile. Falling toward 0 = the person is turning into a clean stereotype."""
    ax = _line_chart(
        df,
        "stereotype_distance",
        group,
        "answers away from nearest stereotype",
        f"Does the person drift into a stereotype? by {group}",
    )
    ax.set_ylim(bottom=0)
    return ax


def plot_failures(chains: list[dict]):
    """Share of survey replies that could not be parsed and of descriptions that looped, per setup and temperature."""
    stats = Counter()
    for c in chains:
        key = f"{c['setup']}\nT={c['temperature']}"
        for r in c["records"]:
            stats[(key, r["kind"], "n")] += 1
            stats[(key, r["kind"], "bad")] += (not r["parse_ok"]) if r["kind"] == "survey" else r["looped"]
    keys = sorted({k for k, _, _ in stats})
    unparsed = [stats[(k, "survey", "bad")] / max(stats[(k, "survey", "n")], 1) for k in keys]
    looped = [stats[(k, "description", "bad")] / max(stats[(k, "description", "n")], 1) for k in keys]
    x = np.arange(len(keys))
    _, ax = plt.subplots(figsize=(max(6, len(keys) * 2.2), 3.5))
    ax.bar(x - 0.2, unparsed, 0.38, color=SERIES[0], label="survey reply unparseable")
    ax.bar(x + 0.2, looped, 0.38, color=SERIES[1], label="description looped")
    ax.set_xticks(x, [k.replace(" + ", " +\n") for k in keys], fontsize=9)
    ax.set_ylim(0, max([*unparsed, *looped, 0.05]) * 1.2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("share of replies")
    title = "Broken outputs" if sum(unparsed) + sum(looped) else "Broken outputs: none in this run (no bars = 0%)"
    ax.set_title(title, loc="left", color=INK)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    return ax


def attractors(df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Most common final answer patterns (last readable survey of each chain), with the nearest coherent profile."""
    readable = df.dropna(subset=["answers"])
    chain = ["setup", "profile", "temperature", "repeat"]
    last = readable[readable["step"] == readable.groupby(chain)["step"].transform("max")]
    names = {"".join(map(str, v)): k for k, v in STEREOTYPES.items()}
    counts = last.groupby("answers").size().sort_values(ascending=False).head(top).reset_index(name="chains")
    counts["nearest stereotype"] = counts["answers"].map(lambda a: _nearest(a, names))
    # Who each chain started as: "stayed" = the person was already this pattern, otherwise the starting person.
    starts = df[df["step"] == 0].groupby("profile")["answers"].first()
    counts["started as"] = counts["answers"].map(
        lambda a: _started_as(last[last["answers"] == a]["profile"], a, starts)
    )
    return counts


def _started_as(profiles: pd.Series, answers: str, starts: pd.Series) -> str:
    """E.g. "18 stayed + 11 from random-2 + 5 from random-1" for the chains that ended on one answer pattern."""
    parts = []
    for profile, n in profiles.value_counts().items():
        parts.append(f"{n} stayed" if starts[profile] == answers else f"{n} from {profile}")
    return " + ".join(sorted(parts, key=lambda part: "stayed" not in part))


def _nearest(answers: str, names: dict[str, str]) -> str:
    best = min(names, key=lambda c: hamming(list(c), list(answers)))
    d = hamming(list(best), list(answers))
    return names[best] if d == 0 else f"{names[best]} (off by {d})"


DRIFT_KEY = ["setup", "models", "chain_length", "profile", "temperature", "repeat"]


def _drift_key(chain: dict) -> tuple:
    setup, models, length, profile, temperature, repeat = chain_key(chain)
    return (setup, "|".join(models), length, profile, temperature, repeat)


def text_drift(caller: Caller | None, chains: list[dict], cache: Path | None = None) -> pd.DataFrame:
    """Meaning similarity of every description to the chain's first written description, via Gemini embeddings.

    The first written description (not the template sentence list) is the reference, so style differences between the
    template and natural prose do not count as drift. With `cache`, each chain's result is saved there and reused, so
    only chains not yet in the cache call the embedding API. Returns rows for exactly the chains passed in.
    """
    cached = (
        pd.read_csv(cache) if cache and cache.exists() else pd.DataFrame(columns=[*DRIFT_KEY, "step", "similarity"])
    )
    have = set(cached[DRIFT_KEY].itertuples(index=False, name=None))
    missing = [c for c in chains if _drift_key(c) not in have]
    rows = []
    for c in missing:
        described = [r for r in c["records"] if r["kind"] == "description"]
        if len(described) < 2:
            continue
        if caller is None:
            raise RuntimeError(f"{len(missing)} chain(s) are not in the drift cache and no API key is set.")
        vectors = np.array(caller.embed(EMBED_MODEL, [r["raw"] for r in described]))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        for r, sim in zip(described, vectors @ vectors[0]):
            rows.append({**dict(zip(DRIFT_KEY, _drift_key(c))), "step": r["step"], "similarity": float(sim)})
    if rows:
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        if cache:
            cached.to_csv(cache, index=False)
    wanted = {_drift_key(c) for c in chains}
    return cached[[k in wanted for k in cached[DRIFT_KEY].itertuples(index=False, name=None)]].reset_index(drop=True)


def plot_text_drift(drift: pd.DataFrame, group: str = "setup"):
    ax = _line_chart(
        drift,
        "similarity",
        group,
        "meaning similarity to first description",
        f"How far the descriptions drift, by {group}",
    )
    ax.set_ylim(0, 1.05)
    return ax


SHORT_ITEMS = ["reserved", "thorough", "worries", "quiet", "talkative", "perseveres", "tense", "disorganized", "stable"]
KEPT, CHANGED, LANDED = SERIES[2], SERIES[0], SERIES[1]


def _flip_label(parent: str, child: str) -> str:
    flips = [f"{'+' if c == '1' else '−'}{SHORT_ITEMS[i]}" for i, (p, c) in enumerate(zip(parent, child)) if p != c]
    return ", ".join(flips)


def _build_tree(chains: list[dict]) -> tuple[dict, dict]:
    """Merges chains into a prefix tree: a node is the sequence of survey answers so far; counts are chains through it."""
    counts: dict[tuple, int] = {}
    children: dict[tuple, list] = {}
    for c in chains:
        path = ("".join(map(str, c["original"])),)
        counts[path] = counts.get(path, 0) + 1
        for r in c["records"]:
            if r["kind"] != "survey" or not r["answers"]:
                continue
            child = (*path, "".join(map(str, r["answers"])))
            if child not in counts:
                children.setdefault(path, []).append(child)
            counts[child] = counts.get(child, 0) + 1
            path = child
    return counts, children


def plot_tree(chains: list[dict], profile: str, setup: str | None = None):
    """Draws how one person's answers branch along the chain, merging chains that took the same path.

    Line width = number of chains. Green = answers unchanged, blue = some traits flipped (labelled), orange square =
    landed exactly on a different coherent profile (a stereotype), open square = back to a version seen earlier on
    that branch.
    """
    picked = [c for c in chains if c["profile"] == profile and (setup is None or c["setup"] == setup)]
    if not picked:
        raise ValueError(f"no chains for profile={profile!r}, setup={setup!r}")
    counts, children = _build_tree(picked)
    root = ("".join(map(str, picked[0]["original"])),)
    stereotypes = {"".join(map(str, v)): k for k, v in STEREOTYPES.items()}

    y_of: dict[tuple, float] = {}

    def place(node: tuple, next_leaf: list[int]) -> float:
        kids = sorted(children.get(node, []), key=lambda k: -counts[k])
        if not kids:
            y_of[node] = next_leaf[0]
            next_leaf[0] += 1
        else:
            y_of[node] = float(np.mean([place(k, next_leaf) for k in kids]))
        return y_of[node]

    place(root, [0])
    leaves = max(y_of.values()) + 1
    _, ax = plt.subplots(figsize=(11, max(3.5, leaves * 1.1)))
    total = counts[root]
    for node, y in y_of.items():
        x = len(node) - 1
        msg = node[-1]
        for kid in children.get(node, []):
            width = 1 + 7 * counts[kid] / total
            same = kid[-1] == msg
            ax.plot(
                [x, x + 1],
                [y, y_of[kid]],
                color=KEPT if same else CHANGED,
                linewidth=width,
                solid_capstyle="round",
                zorder=1,
            )
            if not same:
                ax.annotate(
                    _flip_label(msg, kid[-1]),
                    (x + 0.5, (y + y_of[kid]) / 2),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=CHANGED,
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": SURFACE, "edgecolor": "none"},
                )
        name = stereotypes.get(msg)
        if x > 0 and msg in node[:-1] and msg != node[-2]:
            ax.scatter(x, y, s=70, marker="s", facecolor=SURFACE, edgecolor=LANDED, linewidth=1.5, zorder=3)
        elif x > 0 and name and msg != root[0]:
            ax.scatter(x, y, s=80, marker="s", color=LANDED, zorder=3)
        else:
            ax.scatter(x, y, s=40, color=INK, zorder=3)
        is_leaf = node not in children
        if x == 0 or msg != node[-2] or is_leaf:
            label = f"{msg} ({counts[node]})" + (f"\n{name}" if name and (x == 0 or msg != node[-2]) else "")
            ax.annotate(
                label, (x, y), xytext=(0, -9), textcoords="offset points", fontsize=8, color=INK, ha="center", va="top"
            )
    steps = max(len(n) for n in y_of)
    ax.set_xticks(range(steps), ["original"] + [f"agent {2 * i + 1}" for i in range(steps - 1)])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.4, steps - 0.6)
    ax.set_ylim(leaves - 0.4, -0.6)
    where = setup or "all setups"
    ax.set_title(f"How '{profile}' changed along the chain ({total} chains, {where})", loc="left", color=INK)
    ax.plot([], [], color=KEPT, linewidth=3, label="unchanged")
    ax.plot([], [], color=CHANGED, linewidth=3, label="traits flipped (+ now yes, − now no)")
    ax.scatter([], [], marker="s", color=LANDED, label="landed on a stereotype")
    ax.scatter([], [], marker="s", facecolor=SURFACE, edgecolor=LANDED, label="back to an earlier version")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=9)
    return ax
