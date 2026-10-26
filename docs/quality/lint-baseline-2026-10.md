# Lint and type-check baseline, 2026-10

Measured on `main` at `8eff361` (2026-10-26), after the 0.10.1 release, with the
tool versions pinned in `uv.lock`: ruff 0.15.15 and mypy 2.1.0. The purpose is
to record what the static tools report today, what those findings mean, and
which quality gates we intend to add, before adding any of them.

## Counts

| Command | Repo-wide | In `marqov/` |
| --- | --- | --- |
| `ruff check .` (ruff defaults: E4, E7, E9, F) | 167 | 5 |
| `ruff check . --select E,F,W,I,B,UP` | 552 (310 auto-fixable) | 112 |
| `ruff format --check .` | 120 of 193 files | 29 of 57 files |
| `mypy marqov` (declared `strict = true`) | n/a | 176 errors in 28 files |

Default-rule findings by code, repo-wide: 118 F541 (f-string without
placeholders), 24 F401 (unused import), 10 F841 (unused local), 6 E402, 4 E701,
3 E401, 2 E722 (bare `except`). 162 of the 167 are outside the package: 125 in
`benchmarks/`, 30 in `tests/`, 6 in `docs/qualification/` scripts and 1 in `tools/`.

In `marqov/`, the wider `E,F,W,I,B,UP` selection adds mostly line length
(48 E501), import order (14 I001), exception chaining (13 B904) and
pre-3.12 typing spellings (UP035, UP037, UP047).

mypy's 176 strict errors are dominated by missing annotations and generics
(42 `no-untyped-def`, 29 `type-arg`, 24 `no-untyped-call`, 10
`no-any-return`) and by optional provider SDKs that are untyped or not
installed (33 `import-untyped`, 15 `import-not-found`). `strict = true` has
never been enforced: the CI `mypy marqov` step is advisory.

## The five default-rule findings in `marqov/`

| Finding | Classification |
| --- | --- |
| `marqov/cli/__init__.py:21` F401, `datetime.timedelta` unused | Cosmetic |
| `marqov/platform/job.py:36` F401, `math` unused | Cosmetic |
| `marqov/workflows/graph.py:30` F401, `typing.Callable` unused | Cosmetic |
| `marqov/workflows/runner.py:13` F401, `typing.TYPE_CHECKING` unused | Cosmetic |
| `marqov/executors/azure.py:249` F841, `loop` unused in `execute()` | Cosmetic |

The F841 was read against the surrounding code: `execute()` assigns
`asyncio.get_running_loop()` and never reads it; the helpers it calls
schedule their blocking work themselves. The line has been dead since the initial commit; it hides
no bug. None of the F401s is a re-export (nothing imports these names from
those modules). There are no E722 findings in `marqov/`; the two repo-wide are
in `benchmarks/raw_sdk/` scripts that are not part of the package. The 13 B904
findings in `marqov/` re-raise inside `except` without `from`, which changes
how the traceback describes the exception chain, not behaviour.

Real bugs: none. False positives: none. Cosmetic: all five.

## What has found bugs in this SDK

Adversarial review and independent verification, not lint. The 2026-10 review
cycles produced the fixes in PRs marqov-sdk#247 to marqov-sdk#260 and
marqov-sdk#275 to marqov-sdk#286: lost job identity on cancellation and
timeout, cost and timing read from the wrong Azure object, a phase-damping rate
derived from the wrong quantity, task workdirs left behind, client contract
leaks, unvalidated timeouts. These are contract
and semantics bugs.

As a check, we ran ruff over the pre-fix version of every file those PRs
changed in `marqov/` (21 PRs) and kept findings within one line of a changed
hunk. With `E,F,W,B,UP` selected, 5 lines were flagged, all cosmetic (3 E501,
1 UP035, 1 UP041). With every rule enabled (`--select ALL`), the hits were
style and docstring rules, plus `S110`/`BLE001` (`try`/`except Exception:
pass`) on the block where the Azure cost bug was being swallowed. Those two
rules fire 37 times in `marqov/` today, so they mark a place to look,
not a bug.

## Decided next steps

1. **ruff in `marqov/`**: make `ruff check marqov` clean on the default rules
   (the five findings above).
2. **mypy configuration**: replace the unenforced `strict = true` with
   `check_untyped_defs = true` globally, strict-style checks
   (`disallow_untyped_defs` and related options) on the modules that already
   pass them, and an explicit, commented list of modules temporarily
   exempted until they are burned down. The target is `mypy marqov` exiting 0.
3. **Formatter sweep deferred**: `ruff format` would touch 120 files and
   conflict with every open branch. It lands as its own no-logic PR right after
   a release, when the fewest branches are open.
4. **Future blocking gate**: ruff on `marqov/`, mypy as configured, and
   `diff-cover` on changed lines. No global coverage floor and no complexity
   cap: neither has found a bug here, and both reward the wrong edits.

## Tool-version drift

The advisory CI lint job installs ruff and mypy unpinned, so it already runs
newer versions than `uv.lock` (ruff 0.17.2 and mypy 2.4.0 on 2026-10-23).
ruff 0.17 enables a much wider default rule set: `ruff check .` reports 607
findings and `ruff check marqov` 116, mostly `RUF012`, `BLE001`, `I001` and
`TRY004`. The mypy counts are unchanged. Before any ruff step becomes blocking,
pin the rule selection in `[tool.ruff.lint]` (or the ruff version) so that a
ruff release cannot change what the gate checks.
