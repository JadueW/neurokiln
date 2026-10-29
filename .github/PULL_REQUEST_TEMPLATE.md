## What changed

<!-- One or two sentences. Focus on behavior, not file lists. -->

## Why

<!-- The problem this solves and any design trade-offs. -->

## Verification

- [ ] `python -m pytest engine/tests -q`
- [ ] `bash scripts/ci.sh`
- [ ] `cd web && npm run build`
- [ ] Manual UI check (describe below)

## Leakage and security

- [ ] Fitting still happens only inside training folds.
- [ ] No secrets, databases, logs, private datasets or subject-identifying data are included.
- [ ] Plugin trust behavior is unchanged, or the change is documented.
- [ ] No medical or diagnostic claims were added.

## Screenshots / logs

<!-- For UI changes, include before/after screenshots. Remove personal data. -->
