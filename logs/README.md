# Logs Directory

Runtime logs written by the serving components. All `*.log` files are
git-ignored; this directory is kept so the path exists on a fresh checkout.

- `api.log` — every prediction-service request, with the correlation identifier
  echoed in the `x-request-id` response header (rotating, 2 MB × 3).
