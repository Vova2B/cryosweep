## What this changes

<!-- One or two sentences. If it changes a fitted number, a threshold, or a detection rule,
     say what you measured: the file, the before and after values, and why the new one is right. -->

## Evidence

<!-- Which tests were added, and the reason each one FAILS without this change.
     Paste the failing-before output if it is short. -->

## Checklist

- [ ] Tests added, and shown to fail before the change
- [ ] `QT_QPA_PLATFORM=offscreen python -m pytest -q` passes from the repository root
- [ ] No measurement data committed (`*.dat` outside the fixtures/examples allowlist)
- [ ] I have read and signed the [CLA](../CLA.md) (a bot will prompt on your first PR)
