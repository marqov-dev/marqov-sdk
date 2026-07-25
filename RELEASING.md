# Releasing the Marqov SDK

Everything before the `v*` tag push is reversible; the tag is the irreversible
step (PyPI never lets a version be reused), so validate hard and tag last.

> **How the tag publishes:** pushing a `v*` tag triggers
> `.github/workflows/release.yml`, which builds the package and publishes it to
> PyPI via **OIDC trusted publishing** (`id-token: write` + the `pypi`
> environment — there is no stored API token). The tag is only the trigger and
> the version source (hatchling reads `marqov/__init__.py`). The manual "Run
> workflow" dispatch runs the same build but publishes to TestPyPI. If publishing
> fails at the auth step, it's the PyPI trusted-publisher config
> (project / repo / workflow / environment), not a token.

## 0. Scope
- [ ] `git log --oneline vLAST..main` — everything merged since the last tag.
- [ ] Confirm what's in vs. still in flight. Unmerged branches / open PRs are NOT
      in the release; don't describe them as if they are.

## 1. Changelog & version
- [ ] CHANGELOG covers every merged user-facing change (cross-check the log).
- [ ] No changelog block without a matching tag (`git ls-remote --tags origin`) —
      a dated entry with no tag never shipped; fold it in.
- [ ] Credit contributors: `git shortlog -sne vLAST..main`.
- [ ] Bump `__version__` in `marqov/__init__.py` (single source; pyproject is `dynamic`).
- [ ] No stale version strings: `git grep '<previous-version>'`.

## 2. Pre-flight
- [ ] Hygiene gate: `uv run pytest tests/test_no_private_references.py`
- [ ] Full suite green, right extras + fresh metadata:
      ```
      uv sync --extra qiskit --extra cirq --extra pennylane --extra pytket \
              --extra pyquil --extra dev --reinstall-package marqov
      uv run pytest -q          # incl. test_version (source == installed)
      ```
      Traps: use SEPARATE `--extra` flags (a comma-list is read as one nonexistent
      extra and prunes the venv); `cudaq` is Linux-only, omit it off Linux;
      `--reinstall-package marqov` refreshes editable metadata so `test_version` matches.
- [ ] **URLs resolve** — `python tools/check_urls.py` (must exit 0).
      A wrong/dead URL in shipped source or docs is caught by no unit test (nothing
      fetches a prose URL). The tool greps every URL and hits it. It **fails only on
      non-resolving hosts** (DNS failure — the class that shipped `platform.marqov.com`
      in 0.3.0; the real platform base is `app.marqov.ai`). A 4xx from an auth-gated
      provider API, or a transient timeout, is a **WARN**, not a failure — eyeball
      each so an outage on release day doesn't block the tag.

## 3. Build, inspect, dry-run to TestPyPI
- [ ] Look inside the artifact before any upload:
      ```
      uv build
      python -m zipfile -l dist/*.whl        # wheel contents
      tar tzf dist/*.tar.gz                   # sdist MANIFEST sweep — no strays
      uvx twine check dist/*
      ```
- [ ] Push a `release/X.Y.Z` branch (version bumped, changelog dated).
- [ ] Dry-run: `gh workflow run release.yml --ref release/X.Y.Z` (TestPyPI only).
- [ ] `gh run watch <id> --exit-status` — build ✅, publish-testpypi ✅, publish-pypi skipped.
- [ ] Validate the PUBLISHED artifact, not just the pipeline — install from TestPyPI
      and import it (do it here, while it's still free):
      ```
      pip install --index-url https://test.pypi.org/simple/ \
                  --extra-index-url https://pypi.org/simple/ marqov==X.Y.Z
      python -c "import marqov; print(marqov.__version__)"
      ```
      (`--extra-index-url` because dependencies aren't on TestPyPI.)
- [ ] CI (full suite) green on the branch.

## 4. Cut the release
- [ ] Fast-forward `main` to the release commit and push.
- [ ] **Wait for CI green on `main`** — the last gate before the irreversible step.

### ⚠️ The tag is the point of no return
Pushing a `v*` tag publishes to real PyPI, and **a PyPI version can never be
reused or unpublished.** Tag the exact SHA you validated on TestPyPI, and only
after CI is green on it:

```
git tag -a vX.Y.Z <validated-sha> -m "marqov X.Y.Z"
git push origin vX.Y.Z
```

- [ ] Watch `release.yml` → `publish-pypi` green.
- [ ] `gh release create vX.Y.Z --title "vX.Y.Z" --notes-file NOTES.md`
- [ ] Verify: real PyPI shows X.Y.Z, and a clean-venv `pip install marqov==X.Y.Z` imports.

## Rollback
- **Before the tag:** nothing has been published — fix forward with a new commit.
  Do **not** force-push the default branch: this repo has forks, and rewriting
  `main` hands everyone who cloned or forked divergent history.
- **After the tag:** a PyPI version cannot be unpublished. Ship `X.Y.(Z+1)` with the fix.

## Secrets
Push-protection secret scanning is enabled on this repository (Settings → Code
security). It blocks known credential formats at push time — a real gate, so
there is no per-release secret grep to remember.
