from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence


PartSpec = tuple[str, list[str]]

# One sentence from the caller: "head, torso, arm". Chinese commas / enumeration
# marks count as the same split. Spaces inside a token stay (mushroom=small mushroom).
_PROMPT_SEP = re.compile(r"[,，、]+")


def split_prompt_entries(entries: str | Sequence[str] | None) -> list[str]:
    """Flatten a prompt field into one token per output part.

    Accepts the one-line form (`"head, torso, arm"`), a list of tokens, or a mix.
    Empty pieces from a trailing comma are dropped.
    """
    if entries is None:
        return []
    raw = [entries] if isinstance(entries, str) else list(entries)
    parts: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise ValueError("component prompt entries must be non-empty strings")
        parts.extend(piece.strip() for piece in _PROMPT_SEP.split(item) if piece.strip())
    return parts


def normalize_part_specs(entries: str | Sequence[str]) -> list[PartSpec]:
    raw_entries = split_prompt_entries(entries)
    if not raw_entries:
        raise ValueError("at least one component prompt is required")

    specs: list[PartSpec] = []
    for raw in raw_entries:
        name, separator, joined = raw.partition("=")
        name = name.strip()
        joined = joined if separator else raw
        raw_concepts = joined.split("+")
        concepts = [concept.strip() for concept in raw_concepts]
        if not name or not concepts or any(not concept for concept in concepts):
            raise ValueError(f"invalid component prompt specification: {raw!r}")
        specs.append((name, concepts))

    duplicates = sorted(
        name for name, count in Counter(name for name, _ in specs).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate component name(s): {duplicates}")
    return specs


def part_names(specs: Sequence[PartSpec]) -> list[str]:
    return [name for name, _ in specs]


# Words naming the part that carries everything else; the catch-all when the requested
# one is not among the prompts. Bases and stands are left out: a statue's plinth is a
# part of its own, not where a stray leg belongs.
BODY_WORDS = ("body", "torso", "main body", "trunk", "主体", "身体", "躯干", "机身", "车身", "船身")


def body_word(expected_names: Sequence[str]) -> str | None:
    """The first prompt that names the main body ('torso', 'robot body', '主体'), or None."""
    for name in expected_names:
        word = str(name).strip().lower()
        if word in BODY_WORDS or word.endswith(" body") or word.endswith(" torso"):
            return name
    return None


def resolve_unassigned_to(target: str | None, expected_names: Sequence[str]) -> str | None:
    """The part that absorbs faces no prompt claimed.

    `target` is kept when it names a requested component. A name that is not among the
    prompts -- the default `body` against Chinese prompts, or a typo -- falls back to the
    body word among the prompts (torso, 主体, robot body), else the first prompt, rather
    than to nothing: on a grey model where SAM3 recognised neither 主体 nor 底座,
    "nothing" meant every face was dropped and the export was empty. The body comes
    before the first prompt because with 'head, torso' the first prompt took the puppy's
    legs and tail. An explicit empty string (or a Swagger leftover like `string`) still
    means "drop them".
    """
    if not target or target == "string":
        return None
    if target not in expected_names:
        if not expected_names:
            return None
        fallback = body_word(expected_names) or expected_names[0]
        print(f"[split] unassigned_to={target!r} is not in {list(expected_names)}; "
              f"using {fallback!r} for the unclaimed faces")
        return fallback
    return target


def validate_target_name(target: str | None, expected_names: Sequence[str]) -> None:
    if target is not None and target not in expected_names:
        raise ValueError(
            f"unassigned_to {target!r} is not one of the requested components "
            f"{list(expected_names)!r}"
        )


def validate_named_rows(
    expected_names: Sequence[str],
    rows: Sequence[Mapping],
    key: str,
) -> None:
    """Ensure each row names a requested component exactly once."""
    actual: list[str] = []
    for index, row in enumerate(rows):
        if key not in row:
            raise ValueError(
                f"missing field {key!r} in component row at index {index}"
            )
        name = row[key]
        if not isinstance(name, str):
            raise ValueError(
                f"component name in row at index {index} must be a string, got {name!r}"
            )
        actual.append(name)

    duplicates = sorted(name for name, count in Counter(actual).items() if count > 1)
    missing = sorted(set(expected_names) - set(actual))
    extra = sorted(set(actual) - set(expected_names))
    if duplicates or missing or extra:
        raise ValueError(
            f"component names do not match request: missing={missing}, "
            f"extra={extra}, duplicates={duplicates}"
        )


def legend_part_name(row: Mapping) -> str | None:
    if "part" in row and row["part"] is not None:
        return row["part"]
    return row.get("prompt")


def validate_part_coverage(
    expected_names: Sequence[str],
    rows: Sequence[Mapping],
) -> None:
    """Allow several concept rows to belong to one requested component."""
    actual: list[str] = []
    for index, row in enumerate(rows):
        name = legend_part_name(row)
        if name is None:
            raise ValueError(
                f"missing field 'part' or 'prompt' in component row at index {index}"
            )
        if not isinstance(name, str):
            raise ValueError(
                f"component name in row at index {index} must be a string, got {name!r}"
            )
        actual.append(name)

    missing = sorted(set(expected_names) - set(actual))
    extra = sorted(set(actual) - set(expected_names))
    if missing or extra:
        raise ValueError(
            f"component names do not match request: missing={missing}, extra={extra}"
        )
