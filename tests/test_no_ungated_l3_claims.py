"""Integrity guard: the L3 world-model result stays a STRENGTHENED-BUT-QUALIFIED
two-endpoint result, never inflated into a full cross-env law / measured curve,
and the normalization-bug / instrument story stays a co-lead.

WHY THIS EXISTS
---------------
A methodology audit once caught the project OVERSTATING the L3 (world-model
planner) result, and this guard was added to fence it. The result has since
been STRENGTHENED and is now a defensible, qualified finding:

  * Under matched RECEDING-HORIZON MPC, zero-training world-model latent-MPC
    planning solves NAVIGATION (Wall: JEPA-WM 4/6 ~ DINO-WM 3/6) but not
    CONTACT (PushT: JEPA-WM 0/6 ~ DINO-WM 0/6), and the nav-not-contact split
    REPLICATES across two independent world-model families (DINO-WM and
    facebookresearch/jepa-wms). The earlier one-shot 0/6 JEPA-WM Wall reading
    was a PLANNING-PROTOCOL artifact, corrected by a protocol control.
  * It is still NOT a cross-env LAW and NOT a measured CURVE: N=6 episodes per
    cell, ONE environment per endpoint (so a two-endpoint contrast, gradient
    middle unmeasured; MetaWorld middle anchor is future work).
  * The self-caught normalization bug (ACT x aloha 0.016 -> 0.824 via a clean
    2x2 ablation) plus the auditable instrument remain a co-lead.

This guard greps the tracked public surfaces and FAILS if either invariant
regresses:

  1. NO L3 OVER-CLAIM. The paper may report the strengthened two-endpoint
     result, but it must NOT claim a full cross-env "law" or a measured
     dynamics-complexity "curve", and (in paper/main.tex) it must KEEP its
     qualifying caveats (the N=6 per-cell sample and the single-environment-
     per-endpoint / two-endpoint framing). We also still forbid the specific
     triumphalist tokens the original audit flagged. Dropping the caveats or
     asserting a law/curve fails the build.

  2. THE NORM-BUG / INSTRUMENT CO-LEADS. Each surface must still carry the
     0.016 -> 0.824 self-caught-bug story.

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

# Soft over-claim phrase. "substitute(s) for learning" is acceptable when
# explicitly hedged/negated nearby (the parked-L3 blog uses it as an open
# question). The hard law/curve assertions are handled by FORBIDDEN_TOKENS
# below, which fire regardless of nearby caveats.
OVERCLAIM_RE = re.compile(
    r"substitute[sd]?\b[^.]{0,40}?\blearning\b",
    re.IGNORECASE,
)

# Hard over-claim assertions: a *confirmed/measured/established* cross-env LAW
# or dynamics-complexity CURVE. These are ALWAYS an over-claim no matter what
# caveats sit nearby -- the result is a two-endpoint contrast, never a law or
# a measured curve. Phrased as adjective+noun so the disavowed forms the paper
# actually uses ("not a cross-env law", "not a measured curve") do NOT match.
HARD_OVERCLAIM_RE = re.compile(
    r"(confirmed|measured|established|prove[sdn]?|demonstrat\w+)\s+"
    r"(?:(?:cross-env(?:ironment)?|dynamics-complexity)\s+)?(?:law|curve)"
    r"|(?:cross-env(?:ironment)?|dynamics-complexity)\s+(?:law|curve)"
    r"\s+(?:is|are)\s+(?:confirmed|established|proven)",
    re.IGNORECASE,
)

# Negators that, within NEGATE_WINDOW words, make an over-claim phrase
# acceptable -- it is being explicitly disavowed or hedged, not asserted as a
# settled cross-env law. Two classes: (a) disavowals of a *law/curve* that the
# strengthened paper uses, and (b) the original gating vocabulary that
# not-yet-measured surfaces (the blog's parked L3) still legitimately use to
# keep "substitutes for learning" an open question rather than a claim.
NEGATE_TERMS: tuple[str, ...] = (
    # (a) law/curve disavowals (the strengthened, paper framing)
    "not a",
    "not yet",
    "not a measured",
    "not a cross-env",
    "no cross-env",
    "rather than",
    "two-endpoint",
    "two endpoints",
    "future work",
    "not something we claim",
    "is not",
    "is neither",
    # (b) original gating vocabulary (still-open / parked surfaces)
    "hypothesis",
    "in-flight",
    "in flight",
    "open question",
    "conjecture",
    "pending",
    "parked",
)
NEGATE_WINDOW = 60

# Tokens that are *always* an overclaim regardless of context -- the exact
# triumphalist phrasings the audit flagged. If any reappears, fail outright.
FORBIDDEN_TOKENS: tuple[str, ...] = (
    "central result",
    "central finding",
    "the finding is the conditional",
    "made literally true with a positive result",
    "a full dynamics-complexity gradient",
    "the gradient is confirmed",
)

# Caveats the paper MUST keep so the strengthened result stays qualified.
# Stated as alternative-spellings groups: at least one spelling per group
# must be present in paper/main.tex.
REQUIRED_CAVEAT_GROUPS: tuple[tuple[str, ...], ...] = (
    # the small per-cell sample
    ("n{=}6", "n=6", "$n{=}6$", "6 episodes per cell"),
    # one env per endpoint / two-endpoint, not a curve
    ("two-endpoint", "two endpoints", "one environment per endpoint", "one env per endpoint"),
    # the gradient middle is unmeasured / future work (metaworld anchor)
    ("gradient middle", "middle is unmeasured", "metaworld"),
)


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _words_window(text: str, start: int, end: int) -> str:
    """Return the text within +/- NEGATE_WINDOW words of [start, end)."""
    pre = text[:start].split()
    post = text[end:].split()
    here = text[start:end]
    left = " ".join(pre[-NEGATE_WINDOW:])
    right = " ".join(post[:NEGATE_WINDOW])
    return f"{left} {here} {right}".lower()


def test_no_ungated_l3_overclaim() -> None:
    """Any 'cross-env law' / 'measured curve' / 'substitutes for learning'
    must be explicitly NEGATED nearby.

    Reintroduce a bare assertion that planning 'substitutes for learning' or
    that the gradient is a measured 'curve' / 'cross-env law' (with no nearby
    'not a ...' / 'two-endpoint' / 'future work' disavowal) and this goes red.
    The strengthened two-endpoint reading (navigation solved, contact not,
    replicated across two families) is allowed -- it does not match these
    over-claim phrases.
    """
    offenders: list[str] = []
    for rel in SURFACES:
        text = _read(rel)
        for m in OVERCLAIM_RE.finditer(text):
            window = _words_window(text, m.start(), m.end())
            if not any(term in window for term in NEGATE_TERMS):
                line = text[: m.start()].count("\n") + 1
                offenders.append(f"{rel}:{line}: ungated overclaim -> {m.group(0)!r}")
    assert not offenders, (
        "Ungated L3 over-claim(s) found. A 'cross-env law' / measured 'curve' "
        "/ 'substitutes for learning' phrase must sit within "
        f"{NEGATE_WINDOW} words of a negating term {NEGATE_TERMS!r} (i.e. "
        "explicitly disavowed as still a two-endpoint result, not a law):\n  "
        + "\n  ".join(offenders)
    )


def test_no_confirmed_law_or_curve_assertion() -> None:
    """A *confirmed/measured* cross-env LAW or dynamics-complexity CURVE is an
    over-claim regardless of nearby caveats.

    The result is a two-endpoint contrast; asserting it as a measured curve or
    confirmed law fails the build even if generic caveat words sit elsewhere in
    the paragraph. The paper's *disavowed* forms ("not a cross-env law", "not a
    measured curve") are exempted only when an explicit negator sits within the
    same local clause (a 12-word lookbehind), so a real assertion cannot hide
    behind a distant caveat.
    """
    offenders: list[str] = []
    for rel in SURFACES:
        text = _read(rel)
        for m in HARD_OVERCLAIM_RE.finditer(text):
            pre = " ".join(text[: m.start()].split()[-12:]).lower()
            # Exempt explicit disavowals ("not a measured curve") and clearly
            # aspirational/future framings ("turn ... into a measured curve",
            # "would ... a measured curve", "becomes a measured curve") -- the
            # over-claim is *asserting* a curve exists, not naming it as a goal.
            if not any(
                neg in pre
                for neg in (
                    "not a",
                    "not yet",
                    "no ",
                    "is not",
                    "neither",
                    "rather than",
                    "not}",
                    "into a",
                    "into the",
                    "turn",
                    "would",
                    "becomes",
                    "become",
                    "to a ",
                    "toward",
                    "future work",
                )
            ):
                line = text[: m.start()].count("\n") + 1
                offenders.append(f"{rel}:{line}: asserted law/curve -> {m.group(0)!r}")
    assert not offenders, (
        "L3 asserted as a confirmed/measured cross-env law or "
        "dynamics-complexity curve; it is a two-endpoint contrast, not a "
        "law/curve:\n  " + "\n  ".join(offenders)
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
        "Triumphalist L3 framing reappeared; the world-model result is a "
        "strengthened two-endpoint contrast, not a confirmed cross-env law:\n  "
        + "\n  ".join(offenders)
    )


def test_l3_caveats_are_kept_in_paper() -> None:
    """The strengthened L3 result must KEEP its qualifying caveats.

    Strengthening the result is not licence to drop the caveats. The paper
    must still state (i) the small per-cell sample (N=6), (ii) that this is a
    two-endpoint / one-env-per-endpoint contrast, and (iii) that the gradient
    middle is unmeasured (MetaWorld anchor = future work). Drop any of these
    and the contrast silently inflates into a curve.
    """
    paper = _read("paper/main.tex").lower()
    missing: list[str] = []
    for group in REQUIRED_CAVEAT_GROUPS:
        if not any(spelling in paper for spelling in group):
            missing.append(f"none of {group!r}")
    assert not missing, (
        "paper/main.tex dropped a required L3 caveat (the result must stay a "
        "qualified two-endpoint contrast, not a curve). Missing caveat "
        "group(s): " + "; ".join(missing)
    )


def test_l3_replication_is_named_in_paper() -> None:
    """The paper must name BOTH world-model families and the protocol control.

    The strengthening rests on (a) the nav-not-contact split replicating
    across two independent WM families, and (b) the receding-horizon protocol
    control that fixed the earlier one-shot artifact. Both must be present so
    a reviewer can see *why* the result is now defensible.
    """
    paper = _read("paper/main.tex").lower()
    assert "jepa" in paper, "paper/main.tex must name the second WM family (jepa-wms)"
    assert "dino-wm" in paper or "dino_wm" in paper, "paper/main.tex must name DINO-WM"
    assert "receding-horizon" in paper or "receding horizon" in paper, (
        "paper/main.tex must name the receding-horizon protocol (the control "
        "that fixed the earlier one-shot artifact)"
    )


def test_norm_bug_story_leads_each_surface() -> None:
    """The self-caught normalization bug (0.016 -> 0.824) stays a co-lead.

    Each public surface must carry both endpoints of the recovery, so the
    instrument + norm-bug remains a defensible co-headline alongside L3.
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


def test_paper_abstract_leads_with_instrument_not_l3_overclaim() -> None:
    """In paper/main.tex the abstract must carry the norm-bug lead, and must
    not slip an ungated over-claim ahead of it.

    The instrument + self-caught bug stays a co-lead: the norm-recovery
    sentence must be present in the abstract, and any over-claim phrase that
    appears must (as elsewhere) be explicitly negated.
    """
    paper = _read("paper/main.tex")
    start = paper.index(r"\begin{abstract}")
    end = paper.index(r"\end{abstract}")
    abstract = paper[start:end]

    assert "0.016" in abstract, "abstract must contain the 0.016 norm-bug lead"

    for m in OVERCLAIM_RE.finditer(abstract):
        window = _words_window(abstract, m.start(), m.end())
        assert any(term in window for term in NEGATE_TERMS), (
            "abstract carries an ungated L3 over-claim "
            f"({m.group(0)!r}); it must be disavowed as a two-endpoint result."
        )
