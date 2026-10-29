# Contributing to NeuroKiln

Thanks for helping make BCI evaluation more reproducible. This guide covers the local workflow, the invariants we protect, and the kinds of contributions that fit the project.

## Ground rules

- **No leakage.** Any change touching preprocessing, feature selection, tuning or splitting must keep fitting inside training folds. Add a test that would fail if leakage is introduced.
- **No medical claims.** NeuroKiln is a research tool. Do not describe results as diagnostic or clinical evidence.
- **No secrets or data.** Never commit `.env`, `data/`, `*.db`, `.auth_secret`, tokens, private datasets or subject-identifying information.
- **Small, focused changes.** One behavior per pull request. Separate refactors from features.

## Development setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -e "./engine[datasets,dev]"
pip install -r server/requirements.txt
pip install httpx

cd web
npm ci
```

Run the API and frontend:

```bash
uvicorn app.main:app --app-dir server --host 127.0.0.1 --port 8100
cd web && npm run dev
```

## Checks before opening a pull request

```bash
# Engine tests
python -m pytest engine/tests -q

# Full local CI: tests, API smoke, web build
bash scripts/ci.sh

# Frontend production build
cd web && npm run build

# Plugin manifest validation
neurokiln plugin validate path/to/plugin
```

If you cannot run a check, say so in the pull request. Do not mark a change as verified without evidence.

## Repository layout

| Path | Purpose |
|---|---|
| `engine/neurokiln/` | Evaluation core: data, protocols, DAG, HPO, reports, bundles, CLI |
| `engine/tests/` | Leakage, determinism, node, protocol and storage tests |
| `server/app/` | FastAPI app, auth, job queue, exports |
| `web/src/` | Vue 3 console |
| `docs/` | User manual and documentation site |
| `server/app/main.py` | HTTP surface; keep request models validated |

## Adding a pipeline node

1. Register the node spec and executor in `engine/neurokiln/graph.py` (or the relevant pipeline module).
2. Declare parameters with types, defaults and valid ranges. Ranges must exclude values the algorithm cannot handle.
3. Make the node deterministic for a fixed seed.
4. Add a test covering shape, metadata and a failure case.
5. Add a short entry to `docs/user-manual.md` and the node library UI if needed.

## Adding a dataset adapter

1. Implement the adapter under `engine/neurokiln/data/`.
2. Record license, citation, paradigm metadata, channel names, sampling rate and a stable data signature.
3. Never bundle restricted data. Download on demand and document the source.
4. Add a test with a tiny synthetic fixture; do not depend on network access in tests.

## Adding an HPO template

Templates live in `engine/neurokiln/hpo.py`. A template must:

- produce a valid graph for every point in its search space;
- keep `n_components` within `[2, n_channels]`;
- keep band edges ordered and below Nyquist;
- avoid search dimensions that change graph topology, such as filter-bank band counts.

Custom templates built from saved pipelines use `custom:<pipeline_id>` and must pass `normalize_search_space`.

## Plugins

- Follow manifest v2 and declare `engine_min_version`, `api_version`, entry point and file hashes.
- Plugins run in-process when `NEUROKILN_PLUGIN_TRUST=1`; there is no sandbox. Treat plugin code as trusted code.
- Run `neurokiln plugin validate <path> --strict` before submitting.
- Bundled plugins belong in `engine/neurokiln/plugin_catalog/`.

## Commit messages

Use a short imperative subject, optionally with a scope:

```text
engine: reject band edges above Nyquist in HPO
web: add share card export
docs: document Docker volume layout
```

## Pull request checklist

- [ ] Tests added or updated for the changed behavior.
- [ ] `python -m pytest engine/tests -q` passes.
- [ ] `cd web && npm run build` passes for UI changes.
- [ ] No secrets, datasets, databases or logs are included.
- [ ] Documentation updated when behavior or configuration changes.
- [ ] The change does not weaken leakage guards or the plugin trust model.

## Reporting security issues

Do not open a public issue. Follow [SECURITY.md](SECURITY.md).
