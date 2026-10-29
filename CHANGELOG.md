# Changelog

All notable changes to this project are documented here. The format follows Keep a Changelog and semantic versioning.

## [0.4.0] - 2026-10-30

First public release under the NeuroKiln name.

### Changed

- Renamed the project from BCI Furnace to **NeuroKiln（脑炉）**.
- Renamed the Python package to `neurokiln`, the CLI to `neurokiln`, and all environment variables to `NEUROKILN_*`.
- Renamed the default database to `data/neurokiln.db`; existing databases can be migrated with a SQLite backup.
- Aligned engine, API and web versions at 0.4.0.

### Added

- Apache-2.0 license, citation metadata, security policy and contribution guide.
- Dockerfile, Docker Compose and single-port static hosting of the built frontend.
- Bilingual README and a VitePress documentation site.
- Five-minute onboarding wizard and built-in golden demo.
- Share cards with QR codes and a read-only precomputed demo report.
- Word, LaTeX and GB/T 7714 report exports.
- Dataset import wizard for EDF, FIF and CSV files with channel and event mapping.
- HPO search-space guards and grouped failure summaries.
- Template import/export and built-in pipeline templates.
- Chinese/English UI locale infrastructure.
- Paper reproduction catalog, plugin compatibility checks and online extension examples.
- CI coverage for Windows, Docker builds and API functional tests.

### Fixed

- HPO full re-evaluation could drift from proxy evaluation because stored vectors were rounded to four decimals; vectors now keep full precision.
- Invalid search candidates (reversed bands, bands above Nyquist, `n_components=1`, `reg=0`) are rejected or reported with a reason instead of appearing as opaque failures.

## [0.3.0] - 2026-10-29

### Added

- Custom HPO templates built from saved DAGs with 1–16 search dimensions.
- Search-space validation and persistence.
- Pipeline editor parameter panel and custom-template selection.
- Plugin manifest v2 with hashes and compatibility checks.
- Named ports, filter-bank parallelism and multi-decoder fusion.

## [0.2.0] - 2026-10-28

### Added

- DAG pipeline engine and graphical editor.
- Authentication, background jobs and SQLite experiment store.
- HTML/Excel reports and reproduction bundles.
- EEGNet and ShallowConvNet baselines.
- Statistical tests and calibration.

## [0.1.0] - 2026-10-28

### Added

- Initial evaluation engine, synthetic and MOABB dataset adapters.
- CSP, Riemann and centroid baselines.
- Protocol splitter and leakage tests.
