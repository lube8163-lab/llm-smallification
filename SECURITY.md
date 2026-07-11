# Security policy

## Reporting a vulnerability

Please report security issues privately through GitHub's security advisory
feature. Do not include live API tokens, private model URLs, device logs, or
personal prompts in a public issue.

## Local API

CoreMLProbe contains an optional LAN control API for local automation. The API:

- is disabled by default;
- requires an explicit per-session enable action;
- requires bearer authentication on every endpoint;
- exposes diagnostic logs only after a second explicit opt-in in Debug builds;
- stops and rotates its token when the app enters the background.

The API uses plain HTTP and must not be enabled on an untrusted network. It is
not designed to be exposed through port forwarding, a public IP address, a
reverse proxy, or an internet tunnel.

## Repository contents

Model weights, compiled Core ML bundles, device logs, local environment files,
and Apple signing credentials are intentionally excluded from Git. Before each
public release, scan both the current tree and Git history for credentials and
large binary artifacts.
