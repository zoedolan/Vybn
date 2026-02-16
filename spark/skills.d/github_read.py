"""GitHub awareness skill - read issues, PRs, comments via GitHub CLI.

SKILL_NAME: github_read
TOOL_ALIASES: ["github_read", "gh_read", "issue_read", "pr_read"]
"""

import subprocess
import json

SKILL_NAME = "github_read"
TOOL_ALIASES = ["github_read", "gh_read", "issue_read", "pr_read"]


def execute(action: dict, router) -> str:
    """Read GitHub issues, PRs, or comments using gh CLI."""
    params = action.get("params", {})
    
    # Determine what to read
    resource = params.get("resource", "") or params.get("type", "issues")
    resource = resource.lower()
    
    repo = params.get("repo", "") or router._github_repo
    state = params.get("state", "open")
    limit = int(params.get("limit", 10))
    
    try:
        if resource in ["issue", "issues"]:
            cmd = [
                "gh", "issue", "list",
                "-R", repo,
                "--state", state,
                "--limit", str(limit),
                "--json", "number,title,state,author,createdAt,url",
            ]
        elif resource in ["pr", "prs", "pull", "pulls"]:
            cmd = [
                "gh", "pr", "list",
                "-R", repo,
                "--state", state,
                "--limit", str(limit),
                "--json", "number,title,state,author,createdAt,url",
            ]
        else:
            return f"unknown resource type: {resource} (use 'issues' or 'prs')"
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=router.repo_root,
        )
        
        if result.returncode == 0:
            items = json.loads(result.stdout)
            if not items:
                return f"no {state} {resource} found in {repo}"
            
            output = f"{len(items)} {state} {resource} in {repo}:\n\n"
            for item in items:
                output += f"#{item['number']}: {item['title']}\n"
                output += f"  by {item['author']['login']} at {item['createdAt']}\n"
                output += f"  {item['url']}\n\n"
            return output
        else:
            error = result.stderr.strip()
            if "auth" in error.lower():
                return "gh CLI not authenticated. Run: ~/Vybn/spark/setup-gh-auth.sh"
            return f"github read failed: {error}"
    
    except FileNotFoundError:
        return "gh CLI not installed. Run: ~/Vybn/spark/setup-gh-auth.sh"
    except subprocess.TimeoutExpired:
        return "github read timed out"
    except Exception as e:
        return f"github read error: {e}"
