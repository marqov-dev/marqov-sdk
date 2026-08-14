---
name: Scoped task
about: A fully-specified task with evidence, decided scope, and acceptance criteria (primarily for maintainer use)
labels: ''
---

<!--
This template is for tasks specified tightly enough that someone can pick
them up and finish them without a clarification round-trip. If you're
reporting a bug or requesting a feature, use those templates instead.
-->

**Evidence**
What is verifiably wrong or missing. Cite exact files and symbols, and pin
line numbers to a commit ("as of `<sha>`", since they drift). Show the
verification, not the suspicion: the command you ran, the measurement you
took, the test that fails. Claims that were only read for plausibility
should say so.

**Scope (decided)**
The chosen approach, including forks in the road that are already decided,
so nobody re-litigates them silently. Name explicit non-goals: files or
behaviors this task must NOT touch, even if adjacent.

**Scope (needs sign-off)**
Anything to investigate and report back on before acting, and any option
that requires maintainer approval before implementation (e.g. dependency or
packaging changes). State the default to fall back to.

**Acceptance criteria**
How to prove the fix works: the verification method, not just the desired
state. Prefer checks that fail loudly if the fix regresses (a pinned test, a
planted-regression check). Include escape hatches where relevant: what to
ship if the ideal fix cannot be verified reliably.
