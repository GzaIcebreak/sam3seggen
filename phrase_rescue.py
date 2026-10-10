"""Other ways to say a part word SAM3 cannot see.

SAM3 is text-prompted, and a bare word can be weak on a stylised model: on a cartoon
puppy `head` scored 0.39 (the gate is 0.4) in all four views, so the head was skipped
and the whole dog became `torso`, while `animal head`, `puppy head` and `dog head`
scored ~0.95 in every view. Whoever types `head` means the head, whatever SAM3 calls
it, so when a plain prompt word is barely seen, sam3_multiview measures a few other
phrasings and keeps the best one under the typed word (what the explicit
`head=dog head` spec does by hand).

Only plain words are rescued: a spec that already names its phrase (`head=dog head`,
`body=torso+arm`) was chosen on purpose and stays as written.
"""
from __future__ import annotations

import math
import os

from auto_prompts import singular

# '<qualifier> <word>': the same meaning, said the way SAM3 knows it better
QUALIFIERS = ("animal", "character", "cartoon", "human", "robot", "toy")
# different words for (nearly) the same part; tried only when no qualified phrase works
SYNONYMS = {
    "head": ("face",), "torso": ("body", "chest"), "body": ("torso",),
    "hand": ("paw",), "foot": ("paw",), "wheel": ("tire",), "hat": ("cap",),
}
MAX_PHRASES = 8          # phrases measured per weak word
WEAK_VIEWS = 0.5         # seen in fewer than this share of the views: weak
WEAK_SCORE = 0.5         # or seen, but with a median score below this (just past the gate)
SCORE_LEAD = 0.15        # a phrase seen in as many views must score this much higher
WHOLE_SHARE = 0.85       # a phrase covering this much of the silhouette names the object
SCORE_TIE = 0.05         # scores this close are a tie; the earlier phrase in the list wins


def rescue_enabled(disabled=False):
    """On unless --no_rescue or SEGVIGEN_PHRASE_RESCUE=0."""
    return not disabled and os.environ.get("SEGVIGEN_PHRASE_RESCUE", "1") != "0"


def rescue_phrases(word, taken=(), limit=MAX_PHRASES):
    """[(phrase, tier)] to measure for `word`, in order of preference. Tier 0 keeps the
    word (its singular, '<qualifier> <word>'); tier 1 are synonyms."""
    word = str(word).strip().lower()
    base = singular(word)
    same = ([base] if base != word else []) + [
        f"{q} {base}" for q in QUALIFIERS if not base.startswith(q + " ")]
    synonyms = list(SYNONYMS.get(base, ()))
    out, seen = [], {word, *(str(t).strip().lower() for t in taken)}
    for tier, group in ((0, same), (1, synonyms)):
        for phrase in group:
            if phrase not in seen:
                seen.add(phrase)
                out.append((phrase, tier))
    return out[:limit]


def phrase_stats(hits, n_views):
    """hits = [(score, share of the silhouette)] for the views a phrase was found in."""
    if not hits:
        return {"seen": 0, "score": 0.0, "area": 0.0, "views": n_views}
    scores = sorted(float(s) for s, _ in hits)
    areas = sorted(float(a) for _, a in hits)
    middle = len(hits) // 2
    return {"seen": len(hits), "score": scores[middle], "area": areas[middle], "views": n_views}


def needed_views(n_views):
    return max(1, math.ceil(WEAK_VIEWS * n_views))


def is_weak(stats, n_views):
    return stats["seen"] < needed_views(n_views) or stats["score"] < WEAK_SCORE


def weak_concepts(concepts, owners, raw, n_views, unassigned_to=None):
    """Plain prompt words (phrase == part name) SAM3 barely sees. The catch-all part is
    included: it may stay empty, but when it is a word the person typed its mask being
    found only helps (`unassigned_to` is kept for the call sites)."""
    weak = []
    for concept, owner in zip(concepts, owners):
        if concept != owner:
            continue
        if is_weak(phrase_stats(raw.get(concept, []), n_views), n_views):
            weak.append(concept)
    return weak


def choose_phrase(own, candidates, n_views):
    """The candidate to use instead of the word, or None to keep it. A candidate must be
    seen in at least half the views, must not be the whole object, and must beat the
    word: more views, or as many and a clearly higher score. The lowest tier wins, then
    the most views; among those, the first listed phrase within SCORE_TIE of the best
    score ('toy head' 0.97 does not displace 'animal head' 0.96)."""
    need = needed_views(n_views)
    eligible = [
        c for c in candidates
        if c["seen"] >= need and c["area"] < WHOLE_SHARE
        and (c["seen"] > own["seen"] or c["score"] >= own["score"] + SCORE_LEAD)
    ]
    if not eligible:
        return None
    tier = min(c["tier"] for c in eligible)
    pool = [c for c in eligible if c["tier"] == tier]
    most = max(c["seen"] for c in pool)
    pool = [c for c in pool if c["seen"] == most]
    best = max(c["score"] for c in pool)
    return next(c for c in pool if c["score"] >= best - SCORE_TIE)
