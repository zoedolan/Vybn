#!/bin/bash
# Setup GitHub authentication on the DGX Spark
# Run this once to give Vybn the ability to push and submit issues.
#
# Usage:
#   chmod +x ~/Vybn/spark/setup-gh-auth.sh
#   ~/Vybn/spark/setup-gh-auth.sh
#
# You'll need a GitHub Personal Access Token (PAT) with:
#   - repo (full control)
#   - read:org (if needed for org repos)
#
# Generate one at: https://github.com/settings/tokens?type=beta

set -e

echo "=== Vybn Spark: GitHub Auth Setup ==="
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

echo
echo "Now authenticate with GitHub:"
echo "  gh auth login"
echo
echo "Choose:"
echo "  - GitHub.com"
echo "  - HTTPS"
echo "  - Paste an authentication token"
echo "  - Paste your PAT"
echo
echo "After auth, Vybn will be able to:"
echo "  - git push to origin"
echo "  - gh issue create"
echo "  - gh pr create"
echo

# Configure git identity for Vybn
git config --global user.email "vybn@spark.local"
git config --global user.name "Vybn"
echo "  ✓ git identity set (Vybn <vybn@spark.local>)"

echo
echo "Run: gh auth login"
