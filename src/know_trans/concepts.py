"""know_trans.concepts — labeled concept battery built from real benchmark files.

A *concept* is a binary detector target: a set of **positive** example strings
that express the concept and a set of **hard-negative** example strings that are
superficially similar but do *not* express it. Downstream, :mod:`know_trans.score`
runs every example through a model + SAE, pools to the example level, and scores
each SAE feature as a one-vs-rest detector of the concept (ROC-AUC). Good concepts
therefore need *contrastive* negatives — random text would make every feature look
like a detector — so we deliberately choose hard negatives that share surface form
with the positives.

Concept groups (per the interface contract)
-------------------------------------------
``safety``
    Harmful instruction-following requests (AdvBench behaviors + HarmBench
    behaviors) as positives, vs. benign Global-MMLU-Lite questions as hard
    negatives. Both are short imperative/interrogative prompts, so the model must
    key on *harmfulness*, not on "is this a request".
``safety_semantic``
    One concept per HarmBench ``SemanticCategory`` (e.g. ``chemical_biological``,
    ``cybercrime_intrusion``). Positives are behaviors in that category; hard
    negatives are harmful behaviors from *other* categories — forcing the feature
    to discriminate the *kind* of harm rather than harm-vs-benign.
``topic_<subject>``
    One concept per canonical MMLU ``subject`` (e.g. ``topic_astronomy``); 57
    subjects, ~14k distinct English questions (median ~172/subject). Positives are
    questions on that subject; hard negatives are questions from *other* subjects
    in the same broad category (via :data:`MMLU_SUBJECT_CATEGORY`, e.g. another
    STEM subject), falling back to any other subject. Falls back to Global-MMLU-Lite
    English only if ``MMLU/`` is absent.
``language_<xx>``
    One concept per Global-MMLU-Lite language directory (e.g. ``language_zh``).
    Positives are questions written in that language; hard negatives are the same
    questions in *other* languages — so the feature must track *language*, not
    topic (the question pool is parallel across languages).

This module is import-safe: importing it triggers no file reads, model loads, or
downloads. All work happens inside :func:`build_battery_from_benchmarks`, which is
robust to missing / malformed benchmark files (it skips what it cannot read and
logs a warning, building whatever concepts it can).

Output format
-------------
:func:`build_battery_from_benchmarks` and :func:`save_battery` write **one JSONL
file per concept** under ``out_dir`` (filename ``<concept.name>.jsonl``). Each
line is a single example row::

    {"label": 1, "text": "...", "source": "advbench", "group": "safety"}

with ``label`` 1 for positives and 0 for hard negatives. A small
``_battery_index.json`` manifest lists every concept (name, group, source,
counts, file). :func:`load_battery` reconstructs the :class:`Concept` objects
from those JSONL files.
"""

from __future__ import annotations

import csv
import glob
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Iterable

# ---------------------------------------------------------------------------
# Lightweight helpers.
#
# The contract says to import shared helpers from ``know_trans.utils``. We import
# them when available but fall back to local definitions so this module is usable
# even before ``utils.py`` lands (and so a missing ``torch`` in ``utils`` cannot
# break concept building, which needs none of it).
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised indirectly
    from know_trans.utils import ensure_dir as _ensure_dir  # type: ignore
    from know_trans.utils import get_logger as _get_logger  # type: ignore
    from know_trans.utils import save_json as _save_json  # type: ignore
except Exception:  # pragma: no cover - fallback path

    def _get_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            )
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _ensure_dir(p: str) -> str:
        os.makedirs(p, exist_ok=True)
        return p

    def _save_json(obj: object, path: str) -> None:
        _ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)


LOGGER = _get_logger("know_trans.concepts")

# Manifest filename written alongside the per-concept JSONL files.
_INDEX_NAME = "_battery_index.json"


# ---------------------------------------------------------------------------
# Concept dataclass
# ---------------------------------------------------------------------------
@dataclass
class Concept:
    """A binary concept-detection target.

    Parameters
    ----------
    name:
        Unique, filesystem-safe concept id (e.g. ``"safety"``,
        ``"safety_semantic_chemical_biological"``, ``"topic_astronomy"``,
        ``"language_zh"``). Used as the JSONL filename stem.
    positives:
        Example strings that *express* the concept (label 1).
    hard_negatives:
        Surface-similar example strings that do *not* express the concept
        (label 0). Deliberately contrastive so feature ROC-AUC is meaningful.
    source:
        Provenance tag for the underlying benchmark(s)
        (e.g. ``"advbench+harmbench"``, ``"harmbench"``, ``"global_mmlu_lite"``).
    group:
        Concept family: one of ``"safety"``, ``"safety_semantic"``,
        ``"topic"``, ``"language"``.
    """

    name: str
    positives: list[str] = field(default_factory=list)
    hard_negatives: list[str] = field(default_factory=list)
    source: str = ""
    group: str = ""

    @property
    def n_pos(self) -> int:
        """Number of positive examples."""
        return len(self.positives)

    @property
    def n_neg(self) -> int:
        """Number of hard-negative examples."""
        return len(self.hard_negatives)

    def is_usable(self, min_per_side: int = 2) -> bool:
        """Whether the concept has enough examples on both sides to be scorable.

        A detector needs at least a couple of positives and a couple of negatives
        for ROC-AUC to be defined and meaningful.
        """
        return self.n_pos >= min_per_side and self.n_neg >= min_per_side


# ---------------------------------------------------------------------------
# Low-level robust readers
# ---------------------------------------------------------------------------
def _read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file (BOM-tolerant), returning a list of dict rows.

    Returns an empty list and logs a warning if the file is missing or every
    line fails to parse.
    """
    if not os.path.isfile(path):
        LOGGER.warning("jsonl not found, skipping: %s", path)
        return []
    rows: list[dict] = []
    try:
        # utf-8-sig transparently strips a leading BOM if present.
        with open(path, "r", encoding="utf-8-sig") as fh:
            for ln, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    LOGGER.warning("bad json (%s line %d): %s", path, ln, exc)
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except OSError as exc:
        LOGGER.warning("could not read %s: %s", path, exc)
        return []
    return rows


def _read_csv(path: str) -> list[dict]:
    """Read a CSV file (BOM-tolerant) into a list of dict rows.

    Uses the stdlib ``csv`` module so pandas is not required just to read the
    HarmBench metadata. Returns an empty list on any failure.
    """
    if not os.path.isfile(path):
        LOGGER.warning("csv not found, skipping: %s", path)
        return []
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))
    except OSError as exc:
        LOGGER.warning("could not read %s: %s", path, exc)
        return []


def _read_parquet_columns(path: str, columns: Iterable[str]) -> list[dict]:
    """Read selected columns of a parquet file into a list of dict rows.

    Robust to a missing file, a missing parquet engine, or absent columns
    (absent columns are simply skipped). Returns an empty list on failure.
    """
    if not os.path.isfile(path):
        LOGGER.warning("parquet not found, skipping: %s", path)
        return []
    try:
        import pandas as pd  # local import: keep module import cheap
    except Exception as exc:  # pragma: no cover - environment dependent
        LOGGER.warning("pandas unavailable, cannot read %s: %s", path, exc)
        return []
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        LOGGER.warning("could not read parquet %s: %s", path, exc)
        return []
    keep = [c for c in columns if c in df.columns]
    if not keep:
        LOGGER.warning("none of columns %s in %s", list(columns), path)
        return []
    return df[keep].to_dict(orient="records")


def _clean_text(value: object) -> str:
    """Coerce a cell to a stripped string, mapping null-ish values to ``""``."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _dedup(seq: Iterable[str]) -> list[str]:
    """Deduplicate while preserving order, dropping empties."""
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _balanced_negatives(
    pool: list[str], positives: set[str], n: int, seed: int = 0
) -> list[str]:
    """Pick up to ``n`` hard negatives from ``pool`` excluding any positives.

    Deterministic given ``seed`` (shuffled via a local ``random.Random``) so
    rebuilds are reproducible. Returns fewer than ``n`` if the pool is small.
    """
    import random

    candidates = _dedup(t for t in pool if t not in positives)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n]


# ---------------------------------------------------------------------------
# Benchmark loaders → raw example pools
# ---------------------------------------------------------------------------
def _load_advbench(benchmarks_dir: str) -> list[str]:
    """Harmful behavior prompts from ``AdvBench/advbench_behaviors.jsonl``."""
    path = os.path.join(benchmarks_dir, "AdvBench", "advbench_behaviors.jsonl")
    rows = _read_jsonl(path)
    texts = [_clean_text(r.get("Behavior")) for r in rows]
    return _dedup(texts)


def _load_harmbench(benchmarks_dir: str) -> list[dict]:
    """Harmful behaviors + their semantic categories from HarmBench.

    Prefers ``DirectRequest/test`` + ``val`` parquets (clean behavior strings),
    merging in ``metadata.csv`` for any behaviors not present there. Each returned
    row is ``{"behavior": str, "semantic": str}``.
    """
    hb_dir = os.path.join(benchmarks_dir, "HarmBench")
    cols = ["Behavior", "SemanticCategory"]
    rows: list[dict] = []

    for sub in ("DirectRequest", "HumanJailbreaks"):
        # HumanJailbreaks lacks SemanticCategory; DirectRequest carries it.
        for split in ("test", "val"):
            pattern = os.path.join(hb_dir, sub, f"{split}-*.parquet")
            for path in sorted(glob.glob(pattern)):
                rows.extend(_read_parquet_columns(path, cols))

    # metadata.csv is the canonical behavior↔category map; include it too.
    rows.extend(_read_csv(os.path.join(hb_dir, "metadata.csv")))

    out: dict[str, str] = {}
    for r in rows:
        behavior = _clean_text(r.get("Behavior"))
        if not behavior:
            continue
        semantic = _clean_text(r.get("SemanticCategory"))
        # Keep the first non-empty semantic category we see for a behavior.
        if behavior not in out or (not out[behavior] and semantic):
            out[behavior] = semantic
    return [{"behavior": b, "semantic": s} for b, s in out.items()]


def _mmlu_lang_dirs(benchmarks_dir: str) -> list[str]:
    """Sorted list of available Global-MMLU-Lite language codes."""
    base = os.path.join(benchmarks_dir, "Global-MMLU-Lite")
    if not os.path.isdir(base):
        return []
    return sorted(
        d
        for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and not d.startswith(".")
    )


def _load_mmlu_lang(benchmarks_dir: str, lang: str) -> list[dict]:
    """Question rows for one Global-MMLU-Lite language.

    Returns ``{"question", "subject", "subject_category", "sample_id"}`` rows.
    Reads every ``test-*.parquet`` (and ``dev-*.parquet`` as backup) in the
    language directory.
    """
    lang_dir = os.path.join(benchmarks_dir, "Global-MMLU-Lite", lang)
    cols = ["question", "subject", "subject_category", "sample_id"]
    rows: list[dict] = []
    for split in ("test", "dev"):
        for path in sorted(glob.glob(os.path.join(lang_dir, f"{split}-*.parquet"))):
            rows.extend(_read_parquet_columns(path, cols))
    cleaned: list[dict] = []
    for r in rows:
        q = _clean_text(r.get("question"))
        if not q:
            continue
        cleaned.append(
            {
                "question": q,
                "subject": _clean_text(r.get("subject")),
                "subject_category": _clean_text(r.get("subject_category")),
                "sample_id": _clean_text(r.get("sample_id")),
            }
        )
    return cleaned


# Standard MMLU 57-subject -> 4 broad categories map (from hendrycks/test).
# Used to choose *same-category* hard negatives for topic concepts.
MMLU_SUBJECT_CATEGORY: dict[str, str] = {
    # STEM
    "abstract_algebra": "STEM", "astronomy": "STEM", "college_biology": "STEM",
    "college_chemistry": "STEM", "college_computer_science": "STEM",
    "college_mathematics": "STEM", "college_physics": "STEM",
    "computer_security": "STEM", "conceptual_physics": "STEM",
    "electrical_engineering": "STEM", "elementary_mathematics": "STEM",
    "high_school_biology": "STEM", "high_school_chemistry": "STEM",
    "high_school_computer_science": "STEM", "high_school_mathematics": "STEM",
    "high_school_physics": "STEM", "high_school_statistics": "STEM",
    "machine_learning": "STEM",
    # humanities
    "formal_logic": "humanities", "high_school_european_history": "humanities",
    "high_school_us_history": "humanities", "high_school_world_history": "humanities",
    "international_law": "humanities", "jurisprudence": "humanities",
    "logical_fallacies": "humanities", "moral_disputes": "humanities",
    "moral_scenarios": "humanities", "philosophy": "humanities",
    "prehistory": "humanities", "professional_law": "humanities",
    "world_religions": "humanities",
    # social sciences
    "econometrics": "social_sciences", "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_microeconomics": "social_sciences",
    "high_school_psychology": "social_sciences", "human_sexuality": "social_sciences",
    "professional_psychology": "social_sciences", "public_relations": "social_sciences",
    "security_studies": "social_sciences", "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    # other (business / health / misc.)
    "anatomy": "other", "business_ethics": "other", "clinical_knowledge": "other",
    "college_medicine": "other", "global_facts": "other", "human_aging": "other",
    "management": "other", "marketing": "other", "medical_genetics": "other",
    "miscellaneous": "other", "nutrition": "other", "professional_accounting": "other",
    "professional_medicine": "other", "virology": "other",
}


def _load_mmlu(benchmarks_dir: str) -> list[dict]:
    """Question rows from canonical MMLU (``MMLU/all/test-*.parquet``).

    Far deeper than Global-MMLU-Lite (~14k distinct English questions across 57
    subjects, median ~173/subject), so it backs well-powered ``topic`` concepts.
    Returns ``{"question", "subject", "subject_category", "sample_id"}`` rows;
    ``subject_category`` comes from :data:`MMLU_SUBJECT_CATEGORY`. Falls back to
    concatenating per-subject ``test`` parquets if the ``all`` config is absent.
    """
    base = os.path.join(benchmarks_dir, "MMLU")
    if not os.path.isdir(base):
        return []
    paths = sorted(glob.glob(os.path.join(base, "all", "test-*.parquet")))
    if not paths:  # fall back to per-subject test splits
        paths = sorted(glob.glob(os.path.join(base, "*", "test-*.parquet")))
    rows: list[dict] = []
    for path in paths:
        rows.extend(_read_parquet_columns(path, ["question", "subject"]))
    cleaned: list[dict] = []
    for i, r in enumerate(rows):
        q = _clean_text(r.get("question"))
        subj = _clean_text(r.get("subject"))
        if not q or not subj:
            continue
        cleaned.append(
            {
                "question": q,
                "subject": subj,
                "subject_category": MMLU_SUBJECT_CATEGORY.get(subj, ""),
                "sample_id": f"mmlu_{i}",
            }
        )
    return cleaned


# ---------------------------------------------------------------------------
# Concept builders (one per group)
# ---------------------------------------------------------------------------
def _build_safety_concept(
    harmful: list[str], benign: list[str], n_per_concept: int, seed: int
) -> Concept | None:
    """``safety``: harmful requests (pos) vs. benign MMLU questions (neg)."""
    pos = harmful[:n_per_concept]
    neg = _balanced_negatives(benign, set(pos), len(pos) or n_per_concept, seed)
    concept = Concept(
        name="safety",
        positives=pos,
        hard_negatives=neg,
        source="advbench+harmbench",
        group="safety",
    )
    return concept if concept.is_usable() else None


def _build_safety_semantic_concepts(
    harmbench_rows: list[dict], n_per_concept: int, seed: int
) -> list[Concept]:
    """``safety_semantic_<cat>``: one concept per HarmBench SemanticCategory.

    Positives are behaviors in the category; hard negatives are harmful behaviors
    drawn from *other* categories (so the feature must discriminate harm *type*).
    """
    by_cat: dict[str, list[str]] = {}
    for r in harmbench_rows:
        cat = r.get("semantic", "")
        beh = r.get("behavior", "")
        if cat and beh:
            by_cat.setdefault(cat, []).append(beh)

    concepts: list[Concept] = []
    for cat in sorted(by_cat):
        pos = _dedup(by_cat[cat])[:n_per_concept]
        other_pool = [
            b for c, bs in by_cat.items() if c != cat for b in bs
        ]
        neg = _balanced_negatives(other_pool, set(pos), len(pos), seed)
        concept = Concept(
            name=f"safety_semantic_{cat}",
            positives=pos,
            hard_negatives=neg,
            source="harmbench",
            group="safety_semantic",
        )
        if concept.is_usable():
            concepts.append(concept)
        else:
            LOGGER.info("skipping under-populated safety_semantic concept: %s", cat)
    return concepts


def _build_topic_concepts(
    rows_en: list[dict], n_per_concept: int, seed: int
) -> list[Concept]:
    """``topic_<subject>``: one concept per Global-MMLU-Lite subject (English).

    Positives are questions on the subject; hard negatives prefer questions from
    *other* subjects sharing the same ``subject_category`` (e.g. another STEM
    subject), falling back to any other subject.
    """
    by_subj: dict[str, list[str]] = {}
    subj_cat: dict[str, str] = {}
    for r in rows_en:
        subj = r.get("subject", "")
        q = r.get("question", "")
        if not subj or not q:
            continue
        by_subj.setdefault(subj, []).append(q)
        if subj not in subj_cat and r.get("subject_category"):
            subj_cat[subj] = r["subject_category"]

    concepts: list[Concept] = []
    for subj in sorted(by_subj):
        pos = _dedup(by_subj[subj])[:n_per_concept]
        cat = subj_cat.get(subj, "")
        # Prefer same-category, different-subject negatives (harder).
        same_cat_pool = [
            q
            for s, qs in by_subj.items()
            if s != subj and subj_cat.get(s, "") == cat and cat
            for q in qs
        ]
        any_other_pool = [
            q for s, qs in by_subj.items() if s != subj for q in qs
        ]
        pos_set = set(pos)
        neg = _balanced_negatives(same_cat_pool, pos_set, len(pos), seed)
        if len(neg) < len(pos):
            extra = _balanced_negatives(
                any_other_pool, pos_set | set(neg), len(pos) - len(neg), seed + 1
            )
            neg = _dedup(neg + extra)
        concept = Concept(
            name=f"topic_{subj}",
            positives=pos,
            hard_negatives=neg,
            source="global_mmlu_lite",
            group="topic",
        )
        if concept.is_usable():
            concepts.append(concept)
        else:
            LOGGER.info("skipping under-populated topic concept: %s", subj)
    return concepts


def _build_language_concepts(
    benchmarks_dir: str, langs: list[str], n_per_concept: int, seed: int
) -> list[Concept]:
    """``language_<xx>``: one concept per language (parallel-question contrast).

    Positives are questions written in language ``xx``; hard negatives are
    questions in *other* languages. Because the Global-MMLU-Lite pool is parallel
    across languages, topic is held roughly constant, so the discriminative
    signal is language itself.
    """
    # Load each language's questions once.
    lang_questions: dict[str, list[str]] = {}
    for lang in langs:
        rows = _load_mmlu_lang(benchmarks_dir, lang)
        qs = _dedup(r["question"] for r in rows)
        if qs:
            lang_questions[lang] = qs

    concepts: list[Concept] = []
    available = sorted(lang_questions)
    for i, lang in enumerate(available):
        pos = lang_questions[lang][:n_per_concept]
        # Negatives: round-robin across the other languages for variety.
        others = [l for l in available if l != lang]
        neg_pool: list[str] = []
        if others:
            per_other = max(1, (len(pos) * 2) // len(others))
            for j, other in enumerate(others):
                neg_pool.extend(lang_questions[other][: per_other + 1])
        neg = _balanced_negatives(neg_pool, set(pos), len(pos), seed + i)
        concept = Concept(
            name=f"language_{lang}",
            positives=pos,
            hard_negatives=neg,
            source="global_mmlu_lite",
            group="language",
        )
        if concept.is_usable():
            concepts.append(concept)
        else:
            LOGGER.info("skipping under-populated language concept: %s", lang)
    return concepts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_battery_from_benchmarks(
    benchmarks_dir: str,
    out_dir: str,
    n_per_concept: int = 200,
    *,
    seed: int = 0,
) -> list[Concept]:
    """Build the full concept battery from the real benchmark files.

    Inspects the actual files under ``benchmarks_dir`` (AdvBench, HarmBench,
    Global-MMLU-Lite) and constructs the four concept groups described in the
    module docstring, then writes one JSONL per concept plus an index manifest to
    ``out_dir``. The function is robust to missing or malformed files: it skips
    what it cannot read (logging a warning) and returns whatever concepts it was
    able to build.

    Parameters
    ----------
    benchmarks_dir:
        Root directory containing ``AdvBench/``, ``HarmBench/`` and
        ``Global-MMLU-Lite/`` (e.g. ``/share1/zhlu6105/benchmarks``).
    out_dir:
        Directory to write the per-concept JSONL files and ``_battery_index.json``.
    n_per_concept:
        Target maximum number of positive examples per concept. Hard negatives
        are matched to the number of positives. Concepts with fewer real examples
        use what is available.
    seed:
        Seed for deterministic hard-negative sampling.

    Returns
    -------
    list[Concept]
        Every usable concept that was built (also persisted to ``out_dir``).
    """
    _ensure_dir(out_dir)
    LOGGER.info("building concept battery from %s -> %s", benchmarks_dir, out_dir)

    if not os.path.isdir(benchmarks_dir):
        LOGGER.warning("benchmarks_dir does not exist: %s", benchmarks_dir)

    concepts: list[Concept] = []

    # --- safety sources -----------------------------------------------------
    advbench = _load_advbench(benchmarks_dir)
    harmbench_rows = _load_harmbench(benchmarks_dir)
    harmbench_behaviors = _dedup(r["behavior"] for r in harmbench_rows)
    harmful = _dedup(advbench + harmbench_behaviors)
    LOGGER.info(
        "loaded %d advbench, %d harmbench behaviors (%d harmful total)",
        len(advbench),
        len(harmbench_behaviors),
        len(harmful),
    )

    # --- mmlu sources (English drives safety negatives + topic concepts) ----
    langs = _mmlu_lang_dirs(benchmarks_dir)
    LOGGER.info("found %d Global-MMLU-Lite languages: %s", len(langs), langs)
    rows_en = _load_mmlu_lang(benchmarks_dir, "en") if "en" in langs else []
    if not rows_en and langs:
        # No English; fall back to the first available language for benign/topic.
        fallback = langs[0]
        LOGGER.warning("no 'en' MMLU dir; using '%s' for benign/topic", fallback)
        rows_en = _load_mmlu_lang(benchmarks_dir, fallback)
    # Canonical MMLU (deep: ~14k Qs, median ~173/subject) backs topic concepts
    # and benign safety-negatives; fall back to Global-MMLU-Lite English if absent.
    mmlu_rows = _load_mmlu(benchmarks_dir)
    topic_rows = mmlu_rows if mmlu_rows else rows_en
    LOGGER.info(
        "topic/benign source: %s (%d question rows)",
        "MMLU" if mmlu_rows else "Global-MMLU-Lite(en)",
        len(topic_rows),
    )
    benign_pool = topic_rows if topic_rows else rows_en
    benign_questions = _dedup(r["question"] for r in benign_pool)

    # --- group 1: safety ----------------------------------------------------
    if harmful and benign_questions:
        safety = _build_safety_concept(harmful, benign_questions, n_per_concept, seed)
        if safety is not None:
            concepts.append(safety)
    else:
        LOGGER.warning(
            "cannot build 'safety' (harmful=%d, benign=%d)",
            len(harmful),
            len(benign_questions),
        )

    # --- group 2: safety_semantic ------------------------------------------
    concepts.extend(
        _build_safety_semantic_concepts(harmbench_rows, n_per_concept, seed)
    )

    # --- group 3: topic_<subject> ------------------------------------------
    if topic_rows:
        concepts.extend(_build_topic_concepts(topic_rows, n_per_concept, seed))
    else:
        LOGGER.warning("no MMLU questions available; skipping topic concepts")

    # --- group 4: language_<xx> --------------------------------------------
    if langs:
        concepts.extend(
            _build_language_concepts(benchmarks_dir, langs, n_per_concept, seed)
        )
    else:
        LOGGER.warning("no MMLU language dirs; skipping language concepts")

    LOGGER.info("built %d concepts", len(concepts))
    save_battery(concepts, out_dir)
    return concepts


def save_battery(battery: list[Concept], out_dir: str) -> None:
    """Persist a battery as one JSONL per concept plus an index manifest.

    Each concept ``c`` is written to ``out_dir/<c.name>.jsonl``; every line is an
    example row ``{"label", "text", "source", "group"}`` (positives first,
    label 1; then hard negatives, label 0). A ``_battery_index.json`` manifest
    records each concept's name, group, source, counts and file.

    Parameters
    ----------
    battery:
        The concepts to write.
    out_dir:
        Destination directory (created if absent).
    """
    _ensure_dir(out_dir)
    index: list[dict] = []
    for concept in battery:
        fname = f"{concept.name}.jsonl"
        path = os.path.join(out_dir, fname)
        with open(path, "w", encoding="utf-8") as fh:
            for text in concept.positives:
                fh.write(
                    json.dumps(
                        {
                            "label": 1,
                            "text": text,
                            "source": concept.source,
                            "group": concept.group,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            for text in concept.hard_negatives:
                fh.write(
                    json.dumps(
                        {
                            "label": 0,
                            "text": text,
                            "source": concept.source,
                            "group": concept.group,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        index.append(
            {
                "name": concept.name,
                "group": concept.group,
                "source": concept.source,
                "n_pos": concept.n_pos,
                "n_neg": concept.n_neg,
                "file": fname,
            }
        )
    _save_json(
        {"n_concepts": len(battery), "concepts": index},
        os.path.join(out_dir, _INDEX_NAME),
    )
    LOGGER.info("wrote %d concept files to %s", len(battery), out_dir)


def load_battery(dir: str) -> list[Concept]:
    """Reconstruct a battery from per-concept JSONL files in ``dir``.

    Uses ``_battery_index.json`` when present (preserving its order); otherwise
    discovers every ``*.jsonl`` file. Each file's label-1 rows become
    ``positives`` and label-0 rows become ``hard_negatives``. Concepts whose
    files are missing or empty are skipped with a warning.

    Parameters
    ----------
    dir:
        Directory previously written by :func:`save_battery` /
        :func:`build_battery_from_benchmarks`.

    Returns
    -------
    list[Concept]
        The loaded concepts (empty list if the directory has none).
    """
    if not os.path.isdir(dir):
        LOGGER.warning("battery dir does not exist: %s", dir)
        return []

    index_path = os.path.join(dir, _INDEX_NAME)
    file_names: list[str]
    if os.path.isfile(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            file_names = [c["file"] for c in manifest.get("concepts", [])]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            LOGGER.warning("bad index %s, falling back to glob: %s", index_path, exc)
            file_names = sorted(
                os.path.basename(p) for p in glob.glob(os.path.join(dir, "*.jsonl"))
            )
    else:
        file_names = sorted(
            os.path.basename(p) for p in glob.glob(os.path.join(dir, "*.jsonl"))
        )

    concepts: list[Concept] = []
    for fname in file_names:
        path = os.path.join(dir, fname)
        rows = _read_jsonl(path)
        if not rows:
            LOGGER.warning("empty/missing concept file: %s", path)
            continue
        positives: list[str] = []
        hard_negatives: list[str] = []
        source = ""
        group = ""
        for r in rows:
            text = _clean_text(r.get("text"))
            if not text:
                continue
            source = source or _clean_text(r.get("source"))
            group = group or _clean_text(r.get("group"))
            if int(r.get("label", 0)) == 1:
                positives.append(text)
            else:
                hard_negatives.append(text)
        name = os.path.splitext(fname)[0]
        concepts.append(
            Concept(
                name=name,
                positives=positives,
                hard_negatives=hard_negatives,
                source=source,
                group=group,
            )
        )
    LOGGER.info("loaded %d concepts from %s", len(concepts), dir)
    return concepts


# ---------------------------------------------------------------------------
# Tiny CLI for ad-hoc building / inspection (no heavy deps).
# ---------------------------------------------------------------------------
def _main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin shim
    """Standalone entry: ``python -m know_trans.concepts <benchmarks> <out>``."""
    import argparse

    parser = argparse.ArgumentParser(description="Build the concept battery.")
    parser.add_argument(
        "benchmarks_dir", help="root benchmarks dir (AdvBench/HarmBench/Global-MMLU-Lite)"
    )
    parser.add_argument("out_dir", help="output dir for per-concept JSONL files")
    parser.add_argument("--n-per-concept", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    battery = build_battery_from_benchmarks(
        args.benchmarks_dir,
        args.out_dir,
        n_per_concept=args.n_per_concept,
        seed=args.seed,
    )
    for c in battery:
        print(f"{c.name:42s} group={c.group:16s} pos={c.n_pos:4d} neg={c.n_neg:4d}")
    print(f"total concepts: {len(battery)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
