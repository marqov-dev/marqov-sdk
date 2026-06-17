# Execution architecture: current vs. proposed

Companion to [`ARCHITECTURE-dual-execution-paths.md`](ARCHITECTURE-dual-execution-paths.md).
Diagrams of how circuit execution is wired today and a proposed convergence.

---

## Current (as-is)

Two consumers, two **independent** provider-dispatch implementations that both
ultimately drive the same vendor SDKs. Provider routing — the hardest, most
bug-prone logic — exists twice (plus a third vendored copy in marqov-platform).

```mermaid
flowchart TD
    subgraph C1["Consumer: user scripts (SYNC)"]
        U1["platform_worker generates a wrapper script<br/>from marqov import get_device"]
    end
    subgraph C2["Consumer: platform infra (ASYNC)"]
        U2["status polling · QB sim jobs ·<br/>Temporal activities · README quick-start"]
    end

    U1 -->|sync| P1
    U2 -->|async| P2

    subgraph PATH_A["device.py"]
        P1["get_device(params) → MarqovDevice.run()"]
        D1{{"PROVIDER DISPATCH #1<br/>sniff is_braket / is_ibm / is_azure<br/>repeated in 3 methods — can disagree"}}
        P1 --> D1
    end

    subgraph PATH_B["executors/"]
        P2["ExecutorFactory.create_executor(slug, cfg)<br/>→ BaseExecutor.execute()"]
        D2{{"PROVIDER DISPATCH #2<br/>dispatch on provider string<br/>Braket / IBM / Azure / Local / Sim"}}
        P2 --> D2
    end

    D1 --> V["Vendor SDKs<br/>Braket · Qiskit · Azure · QuantumFlow"]
    D2 --> V

    X["⚠ Third copy: vendored device.py in<br/>marqov-platform/sdk/ (must be kept in sync)"]
    X -.duplicate.- PATH_A

    classDef warn fill:#fdecea,stroke:#d33,color:#900;
    class D1,D2,X warn;
```

**Problems:** provider routing duplicated (device sniffing vs. factory strings);
bug fixes (e.g. IBM `DataBin` counts) must land in 2–3 places; both paths
exported as equals with no signpost; required cross-repo investigation to
understand.

---

## Proposed (to-be)

Keep the legitimate **sync/async split as two thin facades**, but collapse the
duplicated routing onto **one** `detect_provider()` decision and **one** set of
per-provider adapters that own build-device · convert-circuit · run ·
extract-counts. Fix bugs once.

```mermaid
flowchart TD
    subgraph C1["user scripts (SYNC)"]
        U1["get_device(params).run(...)"]
    end
    subgraph C2["platform infra (ASYNC)"]
        U2["await executor.execute(...)"]
    end

    U1 --> F1["MarqovDevice.run()<br/><i>thin sync facade</i>"]
    U2 --> F2["BaseExecutor.execute()<br/><i>thin async facade</i>"]

    F1 --> R
    F2 --> R

    R{{"detect_provider(backend, params)<br/>ONE routing decision, shared"}}
    R --> A["Provider adapters — one per provider<br/>Braket · IBM · Azure · Local · Sim<br/><b>build · convert · run · counts</b><br/>ONE source of truth"]
    A --> V["Vendor SDKs<br/>Braket · Qiskit · Azure · QuantumFlow"]

    classDef good fill:#eafaf1,stroke:#2a9d4a,color:#064;
    class R,A good;
```

**Benefits:** single routing + single adapter layer; sync and async are faces
over one core; a provider bug is fixed once; clearer public surface.

### Incremental path (don't big-bang it)
1. **Now (low risk):** land `detect_provider()` and make it the **shared**
   routing used by *both* paths — removes the divergence even while execution
   facades stay separate. (This is the existing 2026-06-12 refactor plan,
   extended to be shared.)
2. **Signpost:** document sync-for-scripts vs. async-for-infra; consider
   demoting one from the top-level public API.
3. **Later (optional, needs feasibility check):** make `MarqovDevice.run()`
   delegate to `asyncio.run(executor.execute(...))` so there is literally one
   execution core. Blocked on reconciling config shapes (sniffed param-dicts
   vs. typed `*ExecutorConfig`) and confirming counts/measurement semantics
   match. `MarqovDevice.run()` is a public contract used by platform-generated
   scripts — treat as a breaking change.
