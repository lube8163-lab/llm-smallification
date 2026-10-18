# CoreMLProbe local API

The LAN API is intended for short-lived development and benchmark sessions on
a trusted network. It is disabled by default and is not an internet-facing
service.

## Enable from the app

1. Open the `Debug` tab.
2. Turn on `Enable LAN API`.
3. Copy the generated session token.
4. Optionally enable `Allow diagnostic logs` in a Debug build when `/log` or
   `/chatlog` is required.

The token is generated again on every activation and is not persisted. The API
automatically stops and discards the token when the app enters the background.

## Authenticated requests

Set the address and copied token locally. Do not commit the token or paste it
into an issue, article, or terminal transcript intended for publication.

```bash
API_BASE="http://<iphone-address>:8765"
API_TOKEN="<session-token>"

curl -H "Authorization: Bearer $API_TOKEN" \
  "$API_BASE/status"

curl -X POST \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  --data '{"prompt":"Hello","tokens":8}' \
  "$API_BASE/generate"
```

`/log` and `/chatlog` return `403 Forbidden` unless diagnostic access is
explicitly enabled. Diagnostic access is compiled out of Release builds.

## Automation environment

UI-less development sessions can opt in with environment variables:

```text
COREML_PROBE_ENABLE_API=1
COREML_PROBE_API_TOKEN=<at-least-16-characters>
COREML_PROBE_API_PORT=8765
COREML_PROBE_API_DIAGNOSTICS=1   # Debug builds only
```

If no sufficiently long token is supplied, the app generates a random session
token. For unattended automation, supply the token through the launch
environment rather than a tracked file.

## Security boundaries

- Use only on a trusted private LAN or a direct device link.
- Do not enable on shared Wi-Fi.
- Every endpoint, including `/status`, requires bearer authentication.
- Requests are limited to 4 MiB and concurrent generation is rejected.
- Chat and probe logs can contain prompts, outputs, local filenames, and timing
  details; keep diagnostic access off unless it is actively needed.
- HTTP traffic is not encrypted. The bearer token protects authorization, but
  it does not make an untrusted network safe.
