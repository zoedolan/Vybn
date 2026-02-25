
## PR Creation (learned 2025-02-25)
The `gh` CLI token (`~/.config/gh/hosts.yml`) does NOT have PR write scope.
The token in `.env` (`GITHUB_TOKEN=...`) DOES. Use `spark/scripts/create_pr.sh`
or call the REST API directly with `curl` using the `.env` token. Extract it with:
  GITHUB_TOKEN=$(grep '^GITHUB_TOKEN=' .env | cut -d= -f2-)
Do NOT try to `source .env` — other values have special chars that break bash.
