## What this changes

<!-- One or two sentences. If it changes a fitted number, a threshold, or a detection rule,
     say what you measured: the file, the before and after values, and why the new one is right. -->

## Evidence

<!-- Which tests were added, and the reason each one FAILS without this change.
     Paste the failing-before output if it is short. -->

## Verification

<!-- Commit and push first, then run these on that commit and paste the block the second
     command prints BELOW this comment, replacing nothing else:

         QT_QPA_PLATFORM=offscreen pytest --junitxml=pytest-results.xml
         python tools/suite_report.py pytest-results.xml --max-skipped 231 --min-total 2000 --verify-block

     CI fails until the block names the newest commit on this pull request, so paste a
     fresh one after every later push. If main has moved, merge it into this branch
     first: CI counts the tests on the merged result. Editing this description re-runs
     the check ("Re-run jobs" replays the old description). -->

## Checklist

- [ ] Tests added, and shown to fail before the change
- [ ] The verification block above is for the newest commit on this branch
- [ ] No measurement data committed (`*.dat` outside the fixtures/examples allowlist)
- [ ] I have read and signed the [CLA](../CLA.md) (a bot will prompt on your first PR)
