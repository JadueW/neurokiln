# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.4.x | Yes |
| < 0.4 | No |

## Reporting a vulnerability

Please report security issues privately through GitHub Security Advisories ("Report a vulnerability") on the repository. If advisories are unavailable, contact the maintainers listed in `CITATION.cff` before opening a public issue.

Include:

- affected version and component;
- reproduction steps or a proof of concept;
- impact assessment;
- whether the issue requires authentication or plugin trust mode.

We aim to acknowledge reports within 7 days and to publish a fix or mitigation within 30 days for confirmed issues.

## Threat model

NeuroKiln is designed as a **single-node research tool**, not a multi-tenant service.

### Assets

- experiment database (`neurokiln.db`) and artifacts;
- local user password hashes;
- token signing secret (`data/.auth_secret`);
- imported datasets and reports;
- plugin code.

### Trust boundaries

- The web UI and API are trusted local clients.
- Any user with filesystem access to the data directory can read or modify experiments and the signing secret.
- Python plugins are **not sandboxed**. With `NEUROKILN_PLUGIN_TRUST=1`, plugin code runs with the API process's privileges.
- MOABB datasets are downloaded from third-party sources; their integrity and licenses are outside this project's control.

### Controls implemented

- Passwords hashed with PBKDF2-HMAC-SHA256 and per-user salts.
- Bearer tokens signed with HMAC-SHA256; the signing key is generated locally and stored in `data/.auth_secret`.
- All `/api/*` routes except health and authentication require a valid token.
- Destructive operations (user management, cache clearing, plugin installation, deleting experiments or custom pipelines, and overwriting imported datasets) require the admin role.
- Plugin manifests are validated and file hashes checked before installation.
- Uploaded NPZ files are size-limited and parsed with `allow_pickle=False`.
- Experiment results and reports exclude raw subject-identifying data.

### Known limitations

- Tokens are stored in browser `localStorage`; any XSS in the frontend could expose them. Keep the UI and dependencies patched.
- There is no token revocation list; tokens expire after the configured TTL (default 12 hours).
- No rate limiting or account lockout is built in. Do not expose the API directly to the public internet.
- The SQLite database is not encrypted at rest. Use full-disk encryption or a protected volume for sensitive data.
- CORS is restricted to local development origins by default; if you deploy behind a different origin, update the allowlist.
- Plugins execute in-process and can read all application data when trust mode is enabled.

## Deployment guidance

- Bind the API to `127.0.0.1` unless you put it behind an authenticated reverse proxy with TLS.
- Keep `NEUROKILN_PLUGIN_TRUST` unset in shared or production environments.
- Protect `data/` with filesystem permissions; back up `neurokiln.db` and `.auth_secret` together.
- Review MOABB dataset licenses before publishing derived results.
- Do not upload subject-identifying data to public issue trackers.
