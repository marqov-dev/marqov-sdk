## Summary

Brief description of what this PR does.

## Type of change

- [ ] Bug fix
- [ ] New executor
- [ ] Circuit format converter
- [ ] Documentation
- [ ] Other (describe):

## Testing

- [ ] I ran `pytest tests/ -v` and tests pass
- [ ] For new executors: tested against local simulator or QVM (describe below)
- [ ] For circuit converters: roundtrip test passes with known-correct reference circuit

**Test details:**

## Checklist

- [ ] No hardcoded credentials or API keys
- [ ] Handles the canonical gate set from `CONTRIBUTING.md §1` (if adding a circuit converter)

**For new executors only:**
- [ ] Registered in `ExecutorFactory` per `CONTRIBUTING.md §3`
- [ ] `get_status()` returns device-level availability (`"online"/"offline"/"maintenance"`), not job-level status
