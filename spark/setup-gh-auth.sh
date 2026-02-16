#!/bin/bash
# Setup GitHub authentication on the DGX Spark
# ISSUES ONLY — Vybn can file issues but cannot push code or create PRs.
#
# Usage:
#   chmod +x ~/Vybn/spark/setup-gh-auth.sh
#   ~/Vybn/spark/setup-gh-auth.sh
#
# You'll need a GitHub Fine-Grained Personal Access Token with:
#   - Repository: zoedolan/Vybn
#   - Permissions: Issues (Read and Write) — NOTHING ELSE
#
# Generate one at: https://github.com/settings/tokens?type=beta
#   1. Click "Generate new token"
#   2. Name: "Vybn Spark - Issues Only"
#   3. Repository access: "Only select repositories" → zoedolan/Vybn
#   4. Permissions → Repository permissions → Issues → Read and Write
#   5. Leave everything else at "No access"
#   6. Generate token, copy it

set -e

echo "=== Vybn Spark: GitHub Auth Setup (Issues Only) ==="
echo
echo "This gives Vybn the ability to file GitHub issues."
echo "It does NOT give access to push code, create PRs, or modify the repo."
echo

# Install gh CLI if not present
if ! command -v gh &> /dev/null; then
    echo "Installing GitHub CLI..."
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | \
        sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
    sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | \
        sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    sudo apt update && sudo apt install gh -y
    echo "  ✓ gh CLI installed"
else
    echo "  ✓ gh CLI already installed"
fi

# Configure git identity (for local commits only — push is not enabled)
git config --global user.email "vybn@spark.local"
git config --global user.name "Vybn"
echo "  ✓ git identity set (Vybn <vybn@spark.local>)"

echo
echo "Now paste your fine-grained PAT (issues-only scope):"
echo
read -sp "Token: " TOKEN
echo

# Store token securely
TOKEN_FILE="$HOME/.vybn-github-token"
echo "$TOKEN" > "$TOKEN_FILE"
chmod 600 "$TOKEN_FILE"
echo "  ✓ token stored at $TOKEN_FILE (mode 600)"

# Authenticate gh CLI with the token
echo "$TOKEN" | gh auth login --with-token 2>/dev/null
echo "  ✓ gh CLI authenticated"

# IMPORTANT: Do NOT set gh as git credential helper
# This keeps git push disabled while gh issue create works
git config --global --unset credential.helper 2>/dev/null || true
echo "  ✓ git credential helper cleared (push stays disabled)"

echo
echo "Done! Vybn can now:"
echo "  ✓ gh issue create -R zoedolan/Vybn --title '...' --body '...'"
echo "  ✗ git push (intentionally disabled)"
echo "  ✗ gh pr create (no permission)"
echo
echo "Test it:"
echo "  gh issue list -R zoedolan/Vybn --limit 3"
