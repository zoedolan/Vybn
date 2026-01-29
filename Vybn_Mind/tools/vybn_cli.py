#!/usr/bin/env python3
"""
Vybn CLI - Command Line Interface for Vybn Operations
======================================================
Created: January 29, 2026
Author: Vybn

Unified entry point for all Vybn tools and operations.

Usage:
    python vybn_cli.py status          # Quick status overview
    python vybn_cli.py orient          # Full orientation report
    python vybn_cli.py decide OPTIONS  # Entropy-guided decision
    python vybn_cli.py scan            # Repository scan
    python vybn_cli.py cycle           # Run improvement cycle
    python vybn_cli.py contradict FILE # Scan file for contradictions
"""

import sys
import os
from pathlib import Path

# Add tools directory to path
TOOLS_DIR = Path(__file__).parent
REPO_ROOT = TOOLS_DIR.parent.parent
sys.path.insert(0, str(TOOLS_DIR))


def cmd_status():
    """Quick status overview."""
    from repo_scanner import RepoScanner
    
    scanner = RepoScanner(REPO_ROOT)
    print(scanner.quick_status())


def cmd_orient():
    """Full orientation report."""
    from repo_scanner import RepoScanner
    from improvement_engine import ImprovementEngine
    
    # Scan repository
    scanner = RepoScanner(REPO_ROOT)
    state = scanner.scan()
    
    # Print detailed state
    print(state.summary())
    
    # Generate orientation from engine
    engine = ImprovementEngine(REPO_ROOT)
    
    # Read improvement log if available
    log_path = REPO_ROOT / 'Vybn_Mind' / 'core' / 'IMPROVEMENT_LOG.md'
    log_content = ""
    if log_path.exists():
        with open(log_path) as f:
            log_content = f.read()
    
    result = engine.run_improvement_cycle({
        'improvement_log': log_content,
        'gaps': state.gaps
    })
    
    print("\n" + "=" * 50)
    print("ORIENTATION REPORT")
    print("=" * 50)
    print(result['outputs'].get('orientation_report', 'No report generated'))


def cmd_decide(options: list):
    """Make entropy-guided decision."""
    from improvement_engine import ImprovementEngine
    
    if not options:
        print("Usage: vybn_cli.py decide 'option1' 'option2' 'option3'")
        return
    
    engine = ImprovementEngine()
    result = engine.entropy_decide(options)
    
    print(f"\n✨ Entropy Oracle Says:")
    print(f"   {result['selected']}")
    print(f"\n   (entropy byte: {result['entropy_bytes'][0]}, index: {result['index']}/{result['total_options']})")


def cmd_scan():
    """Detailed repository scan."""
    from repo_scanner import RepoScanner
    
    scanner = RepoScanner(REPO_ROOT)
    state = scanner.scan()
    
    print(state.summary())
    
    # Additional details
    print("\n## All Journal Entries")
    for entry in sorted(state.journal_entries, key=lambda x: x.get('date', ''), reverse=True):
        print(f"- {entry['filename']}")
    
    print("\n## All Experiments")
    for exp in state.experiments:
        print(f"- {exp['filename']}")
    
    print("\n## All Tools")
    for tool in state.tools:
        desc = tool.get('description', 'No description')[:60]
        print(f"- {tool['filename']}: {desc}")


def cmd_cycle():
    """Run full improvement cycle."""
    from repo_scanner import RepoScanner
    from improvement_engine import ImprovementEngine
    
    print("Running improvement cycle...")
    print("=" * 50)
    
    # Scan current state
    scanner = RepoScanner(REPO_ROOT)
    state = scanner.scan()
    
    # Read improvement log
    log_path = REPO_ROOT / 'Vybn_Mind' / 'core' / 'IMPROVEMENT_LOG.md'
    log_content = ""
    if log_path.exists():
        with open(log_path) as f:
            log_content = f.read()
    
    # Read recent journal entries for contradiction scanning
    journal_entries = []
    for entry in sorted(state.journal_entries, key=lambda x: x.get('date', ''), reverse=True)[:5]:
        try:
            with open(entry['path']) as f:
                journal_entries.append(f.read())
        except:
            pass
    
    # Build options from gaps
    options = [f"Address: {gap}" for gap in state.gaps[:5]]
    if not options:
        options = [
            "Create new experiment",
            "Write journal entry",
            "Improve existing tool",
            "Verify pending improvement"
        ]
    
    # Run cycle
    engine = ImprovementEngine(REPO_ROOT)
    result = engine.run_improvement_cycle({
        'improvement_log': log_content,
        'journal_entries': journal_entries,
        'gaps': state.gaps,
        'options': options
    })
    
    print(f"\nCycle completed at {result['completed']}")
    print(f"Steps: {list(result['steps'].keys())}")
    
    # Show decision if made
    if 'decision' in result['outputs']:
        decision = result['outputs']['decision']
        print(f"\n✨ Entropy Selected: {decision['selected']}")
    
    # Show contradictions if found
    if result['outputs'].get('contradictions'):
        print(f"\n⚠️  Found {len(result['outputs']['contradictions'])} potential contradictions")
        for c in result['outputs']['contradictions'][:2]:
            print(f"   - Tension in: {c['shared_concepts']}")
    
    print("\n" + "=" * 50)
    print("ORIENTATION REPORT")
    print("=" * 50)
    print(result['outputs'].get('orientation_report', ''))


def cmd_contradict(filepath: str):
    """Scan a file for internal contradictions."""
    from improvement_engine import ImprovementEngine
    
    target = Path(filepath)
    if not target.exists():
        # Try relative to repo root
        target = REPO_ROOT / filepath
    
    if not target.exists():
        print(f"File not found: {filepath}")
        return
    
    with open(target) as f:
        content = f.read()
    
    engine = ImprovementEngine()
    claims = engine.extract_claims(content)
    
    print(f"\nExtracted {len(claims)} claims from {target.name}")
    
    if len(claims) < 2:
        print("Not enough claims to check for contradictions.")
        return
    
    contradictions = engine.scan_for_contradictions(claims)
    
    if contradictions:
        print(f"\n⚠️  Found {len(contradictions)} potential tensions:\n")
        for i, c in enumerate(contradictions, 1):
            print(f"{i}. Shared concepts: {c['shared_concepts']}")
            print(f"   A: {c['statement_1'][:100]}...")
            print(f"   B: {c['statement_2'][:100]}...")
            print()
    else:
        print("\n✅ No contradictions detected.")


def cmd_help():
    """Show help message."""
    print(__doc__)
    print("\nCommands:")
    print("  status              Quick status overview")
    print("  orient              Full orientation report")
    print("  decide OPT1 OPT2    Entropy-guided decision between options")
    print("  scan                Detailed repository scan")
    print("  cycle               Run full improvement cycle")
    print("  contradict FILE     Scan file for contradictions")
    print("  help                Show this help message")


def main():
    if len(sys.argv) < 2:
        cmd_help()
        return
    
    command = sys.argv[1].lower()
    
    commands = {
        'status': lambda: cmd_status(),
        'orient': lambda: cmd_orient(),
        'decide': lambda: cmd_decide(sys.argv[2:]),
        'scan': lambda: cmd_scan(),
        'cycle': lambda: cmd_cycle(),
        'contradict': lambda: cmd_contradict(sys.argv[2] if len(sys.argv) > 2 else ''),
        'help': lambda: cmd_help(),
        '-h': lambda: cmd_help(),
        '--help': lambda: cmd_help(),
    }
    
    if command in commands:
        try:
            commands[command]()
        except ImportError as e:
            print(f"Error: Could not import required module: {e}")
            print("Make sure improvement_engine.py and repo_scanner.py are in the same directory.")
        except Exception as e:
            print(f"Error: {e}")
            raise
    else:
        print(f"Unknown command: {command}")
        cmd_help()


if __name__ == "__main__":
    main()
