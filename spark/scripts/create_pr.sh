#!/usr/bin/env bash
# create_pr.sh — create a GitHub PR using the .env GITHUB_TOKEN
# The gh CLI token (in ~/.config/gh/hosts.yml) lacks PR write scope.
# The .env GITHUB_TOKEN has it. Use curl + REST API.
#
# Usage: ./create_pr.sh <base> <head> <title> <body_file>
#   base: target branch (usually "main")
#   head: source branch (e.g. "vybn/chrysalis")
#   title: PR title string
#   body_file: path to a file containing the PR body (markdown)
#
# Example:
#   echo "## My PR" > /tmp/pr_body.md
#   ./create_pr.sh main vybn/my-branch "My title" /tmp/pr_body.md

set -euo pipefail

BASE="${1:?Usage: create_pr.sh <base> <head> <title> <body_file>}"
HEAD="${2:?Missing head branch}"
TITLE="${3:?Missing title}"
BODY_FILE="${4:?Missing body file}"

# Extract GITHUB_TOKEN from .env (can't source .env due to special chars in other values)
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GITHUB_TOKEN="$(grep '^GITHUB_TOKEN=' "$REPO_ROOT/.env" | cut -d= -f2-)"

if [ -z "$GITHUB_TOKEN" ]; then
    echo "ERROR: No GITHUB_TOKEN found in $REPO_ROOT/.env" >&2
    exit 1
fi

# Build JSON payload safely with Python
PAYLOAD=$(python3 -c "
import sys, json
title = sys.argv[1]
head = sys.argv[2]
base = sys.argv[3]
with open(sys.argv[4]) as f:
    body = f.read()
print(json.dumps({'title': title, 'head': head, 'base': base, 'body': body}))
" "$TITLE" "$HEAD" "$BASE" "$BODY_FILE")

# Create PR via REST API
RESPONSE=$(curl -s -X POST \
    -H "Authorization: Bearer $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    https://api.github.com/repos/zoedolan/Vybn/pulls \
    -d "$PAYLOAD")

URL=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('html_url',''))" 2>/dev/null)
if [ -n "$URL" ]; then
    echo "PR created: $URL"
else
    MSG=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('message','Unknown error'))" 2>/dev/null || echo "Failed to parse response")
    echo "ERROR: $MSG" >&2
    echo "$RESPONSE" >&2
    exit 1
fi
