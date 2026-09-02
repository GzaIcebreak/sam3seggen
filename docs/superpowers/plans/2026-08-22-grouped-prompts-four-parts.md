# Dynamic Grouped Prompts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the segmentation API accept arbitrary single/grouped prompt specifications and strictly export one GLB containing exactly the dynamically requested child Mesh nodes.

**Architecture:** Extract prompt parsing and result validation into a small pure-Python module shared by the API and SAM3 frontend. SAM3 still evaluates unique concepts separately, unions concepts by requested output name, and assigns uncovered foreground to a caller-selected component. The orchestrator validates the 2D legend before expensive 3D inference and validates the final manifest before returning.

**Tech Stack:** Python 3.10+, `argparse`, `numpy`, `Pillow`, `torch`, SAM3, SegviGen, Blender `bpy`, `trimesh`, `unittest`.

## Global Constraints

- Component names, count, and concept groups must come exclusively from the current call; production code must not reference monk-specific names.
- Python callers may pass one string or a sequence of strings.
- `name=concept+a+b` creates one output component whose mask is the union of those concepts.
- The output is one GLB file containing one independently selectable Mesh node per requested component.
- Strict mode rejects missing, extra, or duplicate result names.
- `unassigned_to` must name a component defined by the current prompt specification.
- Preserve existing texture rebaking and coordinate-system behavior.
- Do not create a git commit unless the user explicitly requests one.

---

### Task 1: Pure Dynamic Prompt Specification Module

**Files:**
- Create: `prompt_specs.py`
- Create: `tests/test_prompt_specs.py`

**Interfaces:**
- Produces: `normalize_part_specs(entries: str | Sequence[str]) -> list[tuple[str, list[str]]]`
- Produces: `part_names(specs: Sequence[tuple[str, Sequence[str]]]) -> list[str]`
- Produces: `validate_target_name(target: str | None, expected_names: Sequence[str]) -> None`
- Produces: `validate_named_rows(expected_names: Sequence[str], rows: Sequence[Mapping], key: str) -> None`

- [ ] **Step 1: Write failing parser and validation tests**

```python
import unittest

from prompt_specs import (
    normalize_part_specs,
    part_names,
    validate_named_rows,
    validate_target_name,
)


class PromptSpecsTest(unittest.TestCase):
    def test_accepts_one_prompt_string(self):
        self.assertEqual(
            normalize_part_specs("armor"),
            [("armor", ["armor"])],
        )

    def test_accepts_arbitrary_grouped_prompt_list(self):
        self.assertEqual(
            normalize_part_specs([
                "roof",
                "opening=door+window",
                "wall=facade+brick",
            ]),
            [
                ("roof", ["roof"]),
                ("opening", ["door", "window"]),
                ("wall", ["facade", "brick"]),
            ],
        )

    def test_rejects_duplicate_output_names(self):
        with self.assertRaisesRegex(ValueError, "duplicate component name"):
            normalize_part_specs(["body=head", "body=hand+leg"])

    def test_rejects_empty_name_or_concept(self):
        for value in ("", "=head", "body=", "body=head++leg"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_part_specs(value)

    def test_unassigned_target_is_dynamic(self):
        validate_target_name("wall", ["roof", "opening", "wall"])
        with self.assertRaisesRegex(ValueError, "unassigned_to"):
            validate_target_name("body", ["roof", "opening", "wall"])

    def test_named_rows_must_match_dynamic_request_exactly(self):
        validate_named_rows(
            ["roof", "opening"],
            [{"prompt": "opening"}, {"prompt": "roof"}],
            key="prompt",
        )
        with self.assertRaisesRegex(ValueError, "missing=.*opening"):
            validate_named_rows(["roof", "opening"], [{"prompt": "roof"}], key="prompt")
        with self.assertRaisesRegex(ValueError, "extra=.*floor"):
            validate_named_rows(
                ["roof"],
                [{"prompt": "roof"}, {"prompt": "floor"}],
                key="prompt",
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
E:\AI_New\ModelGen\.venv\Scripts\python.exe -m unittest tests.test_prompt_specs -v
```

Expected: import failure because `prompt_specs.py` does not exist.

- [ ] **Step 3: Implement the pure parser and validators**

```python
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence


PartSpec = tuple[str, list[str]]


def normalize_part_specs(entries: str | Sequence[str]) -> list[PartSpec]:
    raw_entries = [entries] if isinstance(entries, str) else list(entries)
    if not raw_entries:
        raise ValueError("at least one component prompt is required")

    specs: list[PartSpec] = []
    for raw in raw_entries:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("component prompt entries must be non-empty strings")
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
    actual = [row.get(key) for row in rows]
    duplicates = sorted(name for name, count in Counter(actual).items() if count > 1)
    missing = sorted(set(expected_names) - set(actual))
    extra = sorted(set(actual) - set(expected_names))
    if duplicates or missing or extra:
        raise ValueError(
            f"component names do not match request: missing={missing}, "
            f"extra={extra}, duplicates={duplicates}"
        )
```

- [ ] **Step 4: Run the tests and verify GREEN**

Run:

```powershell
E:\AI_New\ModelGen\.venv\Scripts\python.exe -m unittest tests.test_prompt_specs -v
```

Expected: 6 tests pass.

---

### Task 2: SAM3 Group Union and Strict Dynamic Mask Validation

**Files:**
- Modify: `sam3_to_2dmap.py:114-155`
- Modify: `sam3_to_2dmap.py:165-218`
- Modify: `sam3_to_2dmap.py:221-248`
- Create: `tests/test_sam3_grouping.py`

**Interfaces:**
- Consumes: `normalize_part_specs`, `part_names`, and `validate_target_name` from Task 1.
- Produces: `segment_parts(...)` that returns exactly one mask per requested spec or raises `ValueError`.
- Produces: `colorize(...)` that can fold leftovers into any valid dynamic target name.

- [ ] **Step 1: Write failing tests for unions, missing groups, and leftovers**

```python
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import sam3_to_2dmap as module


class Sam3GroupingTest(unittest.TestCase):
    @patch.object(module, "segment_prompts")
    def test_unions_multiple_concepts_into_one_dynamic_component(self, segment_prompts):
        segment_prompts.return_value = [
            {"prompt": "door", "mask": np.array([[1, 0], [0, 0]], bool), "score": 0.8},
            {"prompt": "window", "mask": np.array([[0, 1], [0, 0]], bool), "score": 0.9},
        ]
        parts = module.segment_parts(
            object(), object(), Image.new("RGBA", (2, 2)),
            [("opening", ["door", "window"])], 0.3, "cpu",
        )
        self.assertEqual([part["prompt"] for part in parts], ["opening"])
        np.testing.assert_array_equal(
            parts[0]["mask"],
            np.array([[1, 1], [0, 0]], bool),
        )

    @patch.object(module, "segment_prompts")
    def test_rejects_any_requested_component_without_a_mask(self, segment_prompts):
        segment_prompts.return_value = [
            {"prompt": "roof", "mask": np.ones((2, 2), bool), "score": 0.9},
        ]
        with self.assertRaisesRegex(ValueError, "opening"):
            module.segment_parts(
                object(), object(), Image.new("RGBA", (2, 2)),
                [("roof", ["roof"]), ("opening", ["door", "window"])],
                0.3, "cpu",
            )

    def test_unassigned_foreground_is_folded_into_dynamic_target(self):
        image = Image.new("RGBA", (2, 2), (255, 255, 255, 255))
        parts = [{
            "prompt": "roof",
            "mask": np.array([[1, 0], [0, 0]], bool),
            "score": 1.0,
        }]
        colored, legend = module.colorize(
            image, parts, instance=False, unassigned_to="roof",
        )
        self.assertEqual([row["prompt"] for row in legend], ["roof"])
        self.assertEqual(legend[0]["pixels"], 4)
        self.assertEqual(len(np.unique(np.asarray(colored).reshape(-1, 3), axis=0)), 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
E:\AI_New\ModelGen\.venv\Scripts\python.exe -m unittest tests.test_sam3_grouping -v
```

Expected: the missing component test fails because `segment_parts` currently skips groups with no masks.

- [ ] **Step 3: Replace local parsing and enforce complete group masks**

At module imports:

```python
from prompt_specs import normalize_part_specs, part_names, validate_target_name
```

Delete the local `parse_part_specs` function. In `segment_parts`, collect missing names and raise after all groups are examined:

```python
def segment_parts(processor, model, image, specs, threshold: float, device: str):
    unique = list(dict.fromkeys(
        prompt for _, prompts in specs for prompt in prompts
    ))
    found = {
        part["prompt"]: part
        for part in segment_prompts(processor, model, image, unique, threshold, device)
    }

    parts = []
    missing = []
    for name, prompts in specs:
        members = [found[prompt] for prompt in prompts if prompt in found]
        if not members:
            missing.append(name)
            continue
        mask = np.zeros_like(members[0]["mask"], dtype=bool)
        for member in members:
            mask |= member["mask"].astype(bool)
        parts.append({
            "prompt": name,
            "mask": mask,
            "score": max(member["score"] for member in members),
        })
    if missing:
        raise ValueError(f"SAM3 produced no mask for requested component(s): {missing}")
    return parts
```

In `main()`:

```python
specs = normalize_part_specs(args.prompts)
expected_names = part_names(specs)
validate_target_name(args.unassigned_to, expected_names)
```

- [ ] **Step 4: Run Task 1 and Task 2 tests**

Run:

```powershell
E:\AI_New\ModelGen\.venv\Scripts\python.exe -m unittest tests.test_prompt_specs tests.test_sam3_grouping -v
```

Expected: 9 tests pass.

---

### Task 3: Normalize Python API Inputs and Strictly Validate Pipeline Outputs

**Files:**
- Modify: `segment_api.py:47-171`
- Modify: `segment_api.py:174-231`
- Create: `tests/test_segment_api_contract.py`

**Interfaces:**
- Consumes: all helpers from `prompt_specs.py`.
- Produces: `segment(..., prompts: str | Sequence[str], ..., strict_parts: bool = True)`.
- Produces: one GLB whose manifest names equal the current request when `use_sam3=True` and `strict_parts=True`.

- [ ] **Step 1: Write failing API contract tests**

```python
import unittest

from prompt_specs import normalize_part_specs, part_names, validate_named_rows


class SegmentApiContractTest(unittest.TestCase):
    def test_single_string_does_not_split_into_characters(self):
        specs = normalize_part_specs("armor")
        self.assertEqual(part_names(specs), ["armor"])

    def test_arbitrary_names_drive_final_manifest_validation(self):
        expected = part_names(normalize_part_specs([
            "seat",
            "support=leg+frame",
        ]))
        validate_named_rows(
            expected,
            [{"name": "support"}, {"name": "seat"}],
            key="name",
        )

    def test_extra_unassigned_part_is_rejected(self):
        expected = part_names(normalize_part_specs(["seat", "leg"]))
        with self.assertRaisesRegex(ValueError, "<unassigned>"):
            validate_named_rows(
                expected,
                [
                    {"name": "seat"},
                    {"name": "leg"},
                    {"name": "<unassigned>"},
                ],
                key="name",
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify RED for the current API path**

Run:

```powershell
E:\AI_New\ModelGen\.venv\Scripts\python.exe -m unittest tests.test_segment_api_contract -v
```

Expected: helper-level assertions pass once Task 1 is complete; add an API-level assertion before implementation:

```python
import inspect
import segment_api

self.assertIn("strict_parts", inspect.signature(segment_api.segment).parameters)
```

Expected: FAIL because `strict_parts` is absent.

- [ ] **Step 3: Normalize and validate at the API boundary**

Add imports:

```python
from prompt_specs import (
    normalize_part_specs,
    part_names,
    validate_named_rows,
    validate_target_name,
)
```

Add `strict_parts=True` to `segment(...)`, then normalize before any subprocess:

```python
specs = normalize_part_specs(prompts) if use_sam3 else []
expected_names = part_names(specs)
validate_target_name(unassigned_to, expected_names)
canonical_prompts = [
    name if concepts == [name] else f"{name}={'+'.join(concepts)}"
    for name, concepts in specs
]
```

Pass `canonical_prompts` to SAM3 instead of raw `prompts`.

Immediately after SAM3 returns, validate the legend before starting SegviGen:

```python
if strict_parts:
    with open(legend_path, "r", encoding="utf-8") as file:
        legend = json.load(file)
    validate_named_rows(expected_names, legend, key="prompt")
```

After reading the final manifest:

```python
if strict_parts and use_sam3:
    validate_named_rows(expected_names, manifest, key="name")
```

Update CLI help to state that all names are dynamic. Add an opt-out only for diagnostics:

```python
parser.add_argument(
    "--allow_partial",
    action="store_true",
    help="Return partial/extra components instead of enforcing an exact prompt-name match.",
)
```

Pass `strict_parts=not args.allow_partial`.

- [ ] **Step 4: Preserve intermediates on strict validation failure**

Track completion:

```python
succeeded = False
try:
    # pipeline
    succeeded = True
finally:
    if own_tmp and succeeded:
        shutil.rmtree(work_dir, ignore_errors=True)
    elif own_tmp:
        print(f"pipeline failed; kept intermediates at {work_dir}")
```

- [ ] **Step 5: Run all unit tests**

Run:

```powershell
E:\AI_New\ModelGen\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expected: 12 tests pass with no failures.

---

### Task 4: Monk Four-Component End-to-End Verification

**Files:**
- Modify: `run_monk_4parts.bat`
- Verify output: `data_toolkit/assets/monk/parts4/monk_parts.glb`
- Verify manifest: `data_toolkit/assets/monk/parts4/parts.json`
- Verify views: `data_toolkit/assets/monk/parts4/view_135.png`
- Verify views: `data_toolkit/assets/monk/parts4/view_225.png`
- Verify views: `data_toolkit/assets/monk/parts4/view_0.png`

**Interfaces:**
- Consumes: the strict dynamic API from Task 3.
- Produces: a textured GLB containing exactly four Mesh nodes named for this call.

- [ ] **Step 1: Configure the test with dynamic grouped prompts**

The segmentation command in `run_monk_4parts.bat` must be:

```bat
E:\AI_New\ModelGen\.venv\Scripts\python.exe segment_api.py ^
  --glb %ROOT%\monk.glb ^
  --azimuth 135 ^
  --unassigned_to body ^
  --out %OUT%\monk_parts.glb ^
  --work_dir %OUT%\work ^
  --prompts armor staff base body=head+face+hand+boot+leg
```

- [ ] **Step 2: Run the complete pipeline**

Run:

```powershell
cmd /c run_monk_4parts.bat
```

Expected:

```text
saved ...\monk_parts.glb (4 parts)
ALL DONE
```

- [ ] **Step 3: Verify manifest names and non-empty geometry**

Run:

```powershell
E:\AI_New\ModelGen\.venv\Scripts\python.exe -c "import json; p=r'E:\AI_New\ModelGen\SegviGen\data_toolkit\assets\monk\parts4\parts.json'; rows=json.load(open(p, encoding='utf-8')); assert {r['name'] for r in rows} == {'armor','staff','base','body'}; assert len(rows)==4; assert all(r['faces']>0 for r in rows); print([(r['name'],r['faces']) for r in rows])"
```

Expected: four `(name, positive_face_count)` pairs.

- [ ] **Step 4: Verify GLB node structure and packed textures**

Run:

```powershell
E:\AI_New\ModelGen\.venv\Scripts\python.exe -c "import re,trimesh; p=r'E:\AI_New\ModelGen\SegviGen\data_toolkit\assets\monk\parts4\monk_parts.glb'; s=trimesh.load(p, force='scene'); names={re.sub(r'^part_\d+_','',name) for name in s.geometry}; assert names=={'armor','staff','base','body'}, (names,set(s.geometry)); assert len(s.geometry)==4; assert all(len(g.faces)>0 for g in s.geometry.values()); assert all(getattr(g.visual,'material',None) is not None for g in s.geometry.values()); print(list(s.geometry))"
```

Expected: exactly four geometry names and no assertion.

- [ ] **Step 5: Inspect the three rendered views**

Open:

- `data_toolkit/assets/monk/parts4/view_135.png`
- `data_toolkit/assets/monk/parts4/view_225.png`
- `data_toolkit/assets/monk/parts4/view_0.png`

Expected: upright model, clean baked textures, no off-body spikes, and visually coherent armor/staff/base/body coverage.

- [ ] **Step 6: Run final regression checks**

Run:

```powershell
E:\AI_New\ModelGen\.venv\Scripts\python.exe -m unittest discover -s tests -v
E:\AI_New\ModelGen\.venv\Scripts\python.exe segment_api.py --help
E:\AI_New\ModelGen\.venv_holo\Scripts\python.exe sam3_to_2dmap.py --help
```

Expected: all tests pass; both help commands document single prompts, arbitrary grouped prompts, dynamic `unassigned_to`, and strict result validation.

---

### Task 5: SAM3-only Audit Mode

**Files:**
- Modify: `segment_api.py`
- Modify: `tests/test_segment_api_contract.py`
- Create: `run_monk_sam3_audit.bat`
- Verify: `data_toolkit/assets/monk/sam3_audit/render.png`
- Verify: `data_toolkit/assets/monk/sam3_audit/sam3_2d_map.png`
- Verify: `data_toolkit/assets/monk/sam3_audit/sam3_2d_map_legend.json`

**Interfaces:**
- Produces: `segment(..., sam3_only: bool = False)`.
- Produces in audit mode: `{"render": str, "map": str, "legend": list[dict]}`.
- CLI produces the same artifacts through `--sam3_only`.

- [ ] **Step 1: Write failing contract tests**

```python
import inspect
import segment_api


def test_segment_exposes_sam3_only_mode(self):
    parameter = inspect.signature(segment_api.segment).parameters["sam3_only"]
    self.assertIs(parameter.default, False)


def test_build_audit_result_returns_paths_and_legend(self):
    result = segment_api._build_sam3_audit_result(
        "render.png",
        "map.png",
        [{"prompt": "seat"}],
    )
    self.assertEqual(result, {
        "render": "render.png",
        "map": "map.png",
        "legend": [{"prompt": "seat"}],
    })
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
E:\AI_New\ModelGen\.venv\Scripts\python.exe -m unittest tests.test_segment_api_contract -v
```

Expected: failures because `sam3_only` and `_build_sam3_audit_result` do not exist.

- [ ] **Step 3: Add the audit result helper and API parameter**

```python
def _build_sam3_audit_result(render_path, map_path, legend):
    return {
        "render": os.path.abspath(render_path),
        "map": os.path.abspath(map_path),
        "legend": legend,
    }
```

Add `sam3_only=False` to `segment(...)`. Reject incompatible use before subprocesses:

```python
if sam3_only and not use_sam3:
    raise ValueError("sam3_only requires use_sam3=True")
```

After SAM3 returns and strict legend validation passes:

```python
if sam3_only:
    result = _build_sam3_audit_result(render_path, map_path, legend)
    succeeded = True
    return result
```

Audit artifacts must not be deleted. Require a persistent directory:

```python
if sam3_only and work_dir is None:
    raise ValueError("sam3_only requires work_dir so audit artifacts are preserved")
```

- [ ] **Step 4: Add CLI support**

```python
parser.add_argument(
    "--sam3_only",
    action="store_true",
    help="Stop after rendering and SAM3; output render/map/legend for review.",
)
```

Pass `sam3_only=args.sam3_only`. The CLI must continue requiring prompts and reject `--sam3_only --no_sam`.

- [ ] **Step 5: Run tests and verify GREEN**

Run:

```powershell
E:\AI_New\ModelGen\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 6: Run monk SAM3-only audit**

Create `run_monk_sam3_audit.bat` using:

```bat
E:\AI_New\ModelGen\.venv\Scripts\python.exe segment_api.py ^
  --glb %ROOT%\monk.glb ^
  --azimuth 135 ^
  --sam3_only ^
  --unassigned_to body ^
  --out %OUT%\unused.glb ^
  --work_dir %OUT% ^
  --prompts armor staff base body=head+face+hand+boot+leg
```

Run:

```powershell
cmd /c run_monk_sam3_audit.bat
```

Expected: exit code 0; only render, map, and legend are newly produced; logs contain no SegviGen or Blender steps.

- [ ] **Step 7: Verify artifacts and display the map**

Read the three artifact files. Confirm legend names equal the dynamic request and report visually whether armor/body/staff/base boundaries are acceptable before any 3D run.
