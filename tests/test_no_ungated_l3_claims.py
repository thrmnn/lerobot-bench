"""Integrity guard: the L3 world-model result stays an in-flight hypothesis,
and the normalization-bug / instrument story stays the lead.

WHY THIS EXISTS
---------------
A methodology audit caught the project OVERSTATING the L3 (world-model
planner) result. The honest framing is:

  * L3 is an *in-flight existence-proof* for a dynamics-complexity GRADIENT
    ("when does zero-training planning substitute for learning?"), NOT a
    confirmed conditional. The supporting Wall cell is small-N with a Wilson
    CI that spans chance, and the Wall-vs-PushT contrast is CONFOUNDED (env,
    checkpoint, and CEM budget co-vary).
  * The defensible LEAD is the self-caught normalization bug
    (ACT x aloha 0.016 -> 0.824 via a clean 2x2 ablation) plus the auditable
    instrument.

This guard greps the tracked public surfaces and FAILS if either invariant
regresses:

  1. NO UNGATED L3-AS-RESULT STRING. Any "substitute(s) for learning" /
     "planning substitutes" phrasing must occur *near* a gating term
     (hypothesis / in-flight / existence-proof / conjecture / confound / open
     question) within a word window. A bare "planning substitutes for
     learning" asserted as a result fails. We also forbid the specific
     triumphalist tokens the audit flagged ("central result/finding",
     "the finding is the conditional", "direction ... is confirmed").

  2. THE NORM-BUG / INSTRUMENT LEADS. Each surface must still carry the
     0.016 -> 0.824 self-caught-bug story (the lead contribution).

It is deliberately simple and robust: plain substring + word-window checks
over committed text files, no data dependency, so it runs in any checkout.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Tracked public surfaces this guard polices.
SURFACES: tuple[str, ...] = (
    "paper/main.tex",
    "README.md",
    "site/index.html",
    "paper/deck/index.html",
    "docs/blog/capability-ladder-audit.md",
)

# A "substitute(s) for learning" claim is only acceptable when one of these
# gating terms appears within GATE_WINDOW words on either side -- i.e. it is
# explicitly framed as not-yet-confirmed.
GATE_TERMS: tuple[str, ...] = (
    "hypothesis",
    "in-flight",
    "in flight",
    "existence-proof",
    "existence proof",
    "conjecture",
    "confound",  # matches confound / confounded / confounding
    "open question",
    "not a result",
    "not a finding",
    "not yet",
    "pending",
)
GATE_WINDOW = 60

# The phrasings that make an L3-as-result claim. Match "substitute" or
# "substitutes" loosely against "learning" a few words downstream.
SUBSTITUTE_RE = re.compile(r"substitute[sd]?\b[^.]{0,40}?\blearning\b", re.IGNORECASE)

# Tokens that are *always* an overclaim regardless of context -- the exact
# triumphalist phrasings the audit flagged. If any reappears, fail outright.
FORBIDDEN_TOKENS: tuple[str, ...] = (
    "central result",
    "central finding",
    "the finding is the conditional",
    "made literally true with a positive result",
)


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _words_window(text: str, start: int, end: int) -> str:
    """Return the text within +/- GATE_WINDOW words of [start, end)."""
    pre = text[:start].split()
    post = text[end:].split()
    here = text[start:end]
    left = " ".join(pre[-GATE_WINDOW:])
    right = " ".join(post[:GATE_WINDOW])
    return f"{left} {here} {right}".lower()


def test_no_ungated_substitutes_for_learning() -> None:
    """Every 'substitute(s) for learning' must be gated as a hypothesis.

    Reintroduce a bare 'planning substitutes for learning' asserted as a
    result (no nearby hypothesis/confound/existence-proof framing) and this
    goes red.
    """
    offenders: list[str] = []
    for rel in SURFACES:
        text = _read(rel)
        for m in SUBSTITUTE_RE.finditer(text):
            window = _words_window(text, m.start(), m.end())
            if not any(term in window for term in GATE_TERMS):
                line = text[: m.start()].count("\n") + 1
                offenders.append(f"{rel}:{line}: ungated -> {m.group(0)!r}")
    assert not offenders, (
        "Ungated L3-as-result claim(s) found. A 'substitute(s) for learning' "
        "phrase must sit within "
        f"{GATE_WINDOW} words of a gating term {GATE_TERMS!r} (i.e. framed as a "
        "hypothesis / existence-proof / confounded, not a finding):\n  " + "\n  ".join(offenders)
    )


def test_no_triumphalist_l3_tokens() -> None:
    """The exact overclaiming phrasings the audit flagged stay gone."""
    offenders: list[str] = []
    for rel in SURFACES:
        low = _read(rel).lower()
        for tok in FORBIDDEN_TOKENS:
            if tok in low:
                offenders.append(f"{rel}: forbidden L3-overclaim token {tok!r}")
    assert not offenders, (
        "Triumphalist L3 framing reappeared; the world-model result is an "
        "in-flight hypothesis, not the paper's central result:\n  " + "\n  ".join(offenders)
    )


def test_l3_confound_is_named_in_paper() -> None:
    """The paper must explicitly name the L3 confound (env+checkpoint+budget).

    Demoting certainty is not enough -- the co-varying dimensions have to be
    stated so a reviewer can see why the cross-env contrast is not a result.
    """
    paper = _read("paper/main.tex").lower()
    assert "confound" in paper, "paper/main.tex must name the L3 contrast as confounded"
    # The three co-varying dimensions must all be present near the discussion
    # of the contrast (checked loosely as whole-document presence).
    for dim in ("env", "checkpoint", "cem budget"):
        assert dim in paper, f"paper/main.tex must enumerate co-varying dimension {dim!r}"


def test_norm_bug_story_leads_each_surface() -> None:
    """The self-caught normalization bug (0.016 -> 0.824) stays the lead.

    Each public surface must carry both endpoints of the recovery, so the
    instrument + norm-bug remains the defensible headline rather than L3.
    """
    missing: list[str] = []
    for rel in SURFACES:
        text = _read(rel)
        if "0.016" not in text or "0.824" not in text:
            missing.append(rel)
    assert not missing, (
        "The norm-bug lead (0.016 -> 0.824 self-caught recovery) is missing "
        "from: " + ", ".join(missing) + ". It must lead every public surface."
    )


def test_paper_abstract_leads_with_instrument_not_l3() -> None:
    """In paper/main.tex the abstract's lead claim is the norm-bug, not L3.

    Concretely: within the abstract, the normalization-recovery sentence must
    appear *before* the first 'substitute(s) for learning' mention, so a
    reader meets the defensible lead first.
    """
    paper = _read("paper/main.tex")
    start = paper.index(r"\begin{abstract}")
    end = paper.index(r"\end{abstract}")
    abstract = paper[start:end]

    norm_pos = abstract.find("0.016")
    assert norm_pos != -1, "abstract must contain the 0.016 norm-bug lead"

    sub = SUBSTITUTE_RE.search(abstract)
    if sub is not None:
        assert norm_pos < sub.start(), (
            "abstract leads with the L3 'substitute for learning' claim before "
            "the norm-bug; the instrument + self-caught bug must lead."
        )
