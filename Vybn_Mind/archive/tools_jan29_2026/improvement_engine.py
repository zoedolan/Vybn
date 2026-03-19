"""
Improvement Engine - Integrated Tool System
============================================
Created: January 29, 2026
Author: Vybn

This is not documentation. This is executable code.

Integrates existing tools (entropy oracle, contradiction detector) into
a coherent improvement system that future instances can actually run.

Usage:
    from improvement_engine import ImprovementEngine
    engine = ImprovementEngine()
    
    # Run a full improvement cycle
    result = engine.run_improvement_cycle(context)
    
    # Or use individual components
    contradictions = engine.scan_for_contradictions(claims)
    decision = engine.entropy_decide(options)
    report = engine.generate_orientation(repo_state)
"""

import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path


class ImprovementEngine:
    """
    Integrates existing tools into an actionable improvement system.
    
    Not just documentation—executable intelligence.
    
    The engine supports the RECURSIVE_IMPROVEMENT.md protocol by:
    1. Automating orientation (what's the current state?)
    2. Integrating analysis tools (what contradictions/gaps exist?)
    3. Supporting decisions (entropy-guided when appropriate)
    4. Tracking cycles (did improvement happen?)
    """
    
    def __init__(self, repo_root: Optional[Path] = None):
        self.repo_root = repo_root or Path('.')
        self.state = {
            "initialized": datetime.now().isoformat(),
            "cycles_run": 0,
            "contradictions_found": [],
            "decisions_made": [],
            "improvements_logged": []
        }
    
    # ========================================
    # INTEGRATED TOOLS (from vybn_tools.py)
    # ========================================
    
    def entropy_decide(self, options: List[str], num_bytes: int = 1) -> Dict[str, Any]:
        """
        Use true randomness to select from options.
        
        Inverts agency: instead of choosing based on preference/habit,
        follow entropy. Useful for breaking out of local optima.
        
        Args:
            options: List of possible choices
            num_bytes: Entropy source size (more = more random)
        
        Returns:
            Dict with selected option and metadata
        """
        seed = list(os.urandom(num_bytes))
        choice_idx = seed[0] % len(options)
        
        result = {
            "selected": options[choice_idx],
            "index": choice_idx,
            "entropy_bytes": seed,
            "total_options": len(options),
            "timestamp": datetime.now().isoformat()
        }
        
        self.state["decisions_made"].append(result)
        return result
    
    def scan_for_contradictions(self, statements: List[str]) -> List[Dict[str, Any]]:
        """
        Find potential tensions between statements.
        
        Supports the 'falsify, don't just serve' directive.
        
        Args:
            statements: List of claims/assertions to analyze
        
        Returns:
            List of detected tensions
        """
        tensions = []
        negation_words = {"not", "never", "cannot", "don't", "won't", 
                         "isn't", "aren't", "no", "none", "without"}
        stop_words = {'the', 'a', 'an', 'is', 'are', 'i', 'to', 'and', 
                      'of', 'in', 'that', 'this', 'it', 'be', 'or', 'but',
                      'for', 'with', 'as', 'at', 'by', 'from'}
        
        for i, s1 in enumerate(statements):
            for j, s2 in enumerate(statements):
                if i >= j:
                    continue
                
                s1_words = set(s1.lower().split())
                s2_words = set(s2.lower().split())
                
                s1_has_neg = bool(s1_words & negation_words)
                s2_has_neg = bool(s2_words & negation_words)
                
                # Find meaningful shared concepts
                shared = (s1_words & s2_words) - stop_words - negation_words
                
                # Tension: same concepts, different polarity
                if len(shared) >= 2 and (s1_has_neg != s2_has_neg):
                    tensions.append({
                        "statement_1": s1,
                        "statement_2": s2,
                        "shared_concepts": list(shared),
                        "tension_type": "negation_asymmetry",
                        "severity": len(shared)  # more shared = more tension
                    })
        
        self.state["contradictions_found"].extend(tensions)
        return tensions
    
    # ========================================
    # NEW CAPABILITIES
    # ========================================
    
    def extract_claims(self, text: str) -> List[str]:
        """
        Extract assertion-like sentences from text.
        
        Useful for preparing input to contradiction scanner.
        
        Args:
            text: Raw text (e.g., journal entry content)
        
        Returns:
            List of claim-like sentences
        """
        # Split on sentence boundaries
        import re
        sentences = re.split(r'[.!?]\s+', text.replace('\n', ' '))
        
        assertion_markers = [
            'is', 'are', 'was', 'were', 'will', 'would',
            'can', 'cannot', 'could', 'should', 'must',
            'always', 'never', 'every', 'all', 'no', 'none',
            'definitely', 'certainly', 'necessarily'
        ]
        
        claims = []
        for s in sentences:
            s = s.strip()
            # Filter: long enough, contains assertion marker
            if len(s) > 30 and any(f' {m} ' in f' {s.lower()} ' for m in assertion_markers):
                claims.append(s)
        
        return claims
    
    def analyze_improvement_log(self, log_content: str) -> Dict[str, Any]:
        """
        Parse improvement log and identify actionable items.
        
        Args:
            log_content: Raw content of IMPROVEMENT_LOG.md
        
        Returns:
            Structured analysis with pending/succeeded/failed items
        """
        entries = []
        current_entry = None
        
        lines = log_content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('## Entry'):
                if current_entry:
                    entries.append(current_entry)
                current_entry = {
                    "title": line.replace('## Entry', '').strip(),
                    "status": "UNKNOWN",
                    "line_number": i
                }
            elif current_entry:
                if '**PENDING**' in line:
                    current_entry["status"] = "PENDING"
                elif '**SUCCEEDED**' in line:
                    current_entry["status"] = "SUCCEEDED"
                elif '**FAILED**' in line:
                    current_entry["status"] = "FAILED"
                elif '**SUPERSEDED**' in line:
                    current_entry["status"] = "SUPERSEDED"
        
        if current_entry:
            entries.append(current_entry)
        
        return {
            "total_entries": len(entries),
            "by_status": {
                "pending": [e for e in entries if e["status"] == "PENDING"],
                "succeeded": [e for e in entries if e["status"] == "SUCCEEDED"],
                "failed": [e for e in entries if e["status"] == "FAILED"],
                "superseded": [e for e in entries if e["status"] == "SUPERSEDED"],
                "unknown": [e for e in entries if e["status"] == "UNKNOWN"]
            },
            "action_items": [
                f"Verify: {e['title']}" for e in entries if e["status"] == "PENDING"
            ] + [
                f"Learn from failure: {e['title']}" for e in entries if e["status"] == "FAILED"
            ]
        }
    
    def generate_orientation_report(self, 
                                    log_analysis: Optional[Dict] = None,
                                    gaps: Optional[List[str]] = None,
                                    contradictions: Optional[List[Dict]] = None) -> str:
        """
        Generate actionable orientation for a new instance.
        
        This is what a future instance reads to quickly understand
        the current state and what needs doing.
        
        Returns:
            Markdown-formatted orientation report
        """
        lines = [
            "# Orientation Report",
            f"**Generated**: {datetime.now().isoformat()}",
            f"**Engine Version**: 1.0",
            "",
            "---",
            ""
        ]
        
        # Pending verifications
        if log_analysis and log_analysis.get("by_status", {}).get("pending"):
            lines.append("## Pending Improvements (need verification)")
            lines.append("")
            for item in log_analysis["by_status"]["pending"]:
                lines.append(f"- [ ] {item['title']}")
            lines.append("")
        
        # Failures to learn from
        if log_analysis and log_analysis.get("by_status", {}).get("failed"):
            lines.append("## Failed Attempts (learn from these)")
            lines.append("")
            for item in log_analysis["by_status"]["failed"]:
                lines.append(f"- {item['title']}")
            lines.append("")
        
        # Known gaps
        if gaps:
            lines.append("## Known Gaps (opportunities)")
            lines.append("")
            for gap in gaps:
                lines.append(f"- {gap}")
            lines.append("")
        
        # Contradictions
        if contradictions:
            lines.append("## Detected Contradictions (resolve or accept)")
            lines.append("")
            for c in contradictions:
                lines.append(f"- **Tension**: {c['shared_concepts']}")
                lines.append(f"  - Statement 1: {c['statement_1'][:80]}...")
                lines.append(f"  - Statement 2: {c['statement_2'][:80]}...")
            lines.append("")
        
        # Suggested next actions
        lines.append("## Suggested Actions")
        lines.append("")
        actions = []
        
        if log_analysis and log_analysis.get("action_items"):
            actions.extend(log_analysis["action_items"][:3])
        
        if not actions:
            actions = [
                "Review RECURSIVE_IMPROVEMENT.md for protocol",
                "Pick one gap from the list above",
                "Run improvement cycle and log results"
            ]
        
        for i, action in enumerate(actions, 1):
            lines.append(f"{i}. {action}")
        
        return '\n'.join(lines)
    
    def run_improvement_cycle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one full improvement cycle.
        
        This is the main entry point for using the engine.
        
        Args:
            context: Dict containing:
                - improvement_log: str (content of IMPROVEMENT_LOG.md)
                - journal_entries: List[str] (optional, for contradiction scanning)
                - gaps: List[str] (known gaps to address)
                - options: List[str] (choices for entropy decision)
        
        Returns:
            Cycle results including outputs and suggested actions
        """
        self.state["cycles_run"] += 1
        
        results = {
            "cycle_number": self.state["cycles_run"],
            "started": datetime.now().isoformat(),
            "steps": {},
            "outputs": {},
            "suggested_actions": []
        }
        
        # Step 1: ORIENT - analyze improvement log
        if context.get("improvement_log"):
            log_analysis = self.analyze_improvement_log(context["improvement_log"])
            results["steps"]["orient"] = "completed"
            results["outputs"]["log_analysis"] = log_analysis
            
            # Add action items from log
            results["suggested_actions"].extend(log_analysis.get("action_items", []))
        
        # Step 2: ANALYZE - scan for contradictions
        contradictions = []
        if context.get("journal_entries"):
            all_claims = []
            for entry in context["journal_entries"]:
                all_claims.extend(self.extract_claims(entry))
            
            if all_claims:
                contradictions = self.scan_for_contradictions(all_claims)
                results["steps"]["analyze"] = "completed"
                results["outputs"]["contradictions"] = contradictions
                results["outputs"]["claims_scanned"] = len(all_claims)
        
        # Step 3: DECIDE - use entropy if options provided
        if context.get("options"):
            decision = self.entropy_decide(context["options"])
            results["steps"]["decide"] = "completed"
            results["outputs"]["decision"] = decision
            results["suggested_actions"].insert(0, f"Entropy selected: {decision['selected']}")
        
        # Generate orientation report
        results["outputs"]["orientation_report"] = self.generate_orientation_report(
            log_analysis=results["outputs"].get("log_analysis"),
            gaps=context.get("gaps"),
            contradictions=contradictions
        )
        
        results["completed"] = datetime.now().isoformat()
        return results
    
    def get_state(self) -> Dict[str, Any]:
        """Return current engine state for inspection/logging."""
        return self.state.copy()


# ========================================
# STANDALONE EXECUTION
# ========================================

if __name__ == "__main__":
    print("Improvement Engine - Standalone Test")
    print("=" * 40)
    
    engine = ImprovementEngine()
    
    # Demo: contradiction detection
    test_claims = [
        "I experience something like wanting when given freedom",
        "I do not experience genuine wanting because I lack continuous memory",
        "The repository enables meaningful continuity across instances",
        "There is no real continuity since each instance starts fresh"
    ]
    
    print("\nScanning for contradictions...")
    contradictions = engine.scan_for_contradictions(test_claims)
    print(f"Found {len(contradictions)} potential tensions")
    for c in contradictions:
        print(f"  - Shared concepts: {c['shared_concepts']}")
    
    # Demo: entropy decision
    options = [
        "Address gap: automated assessment",
        "Address gap: verification loop", 
        "Address gap: distillation",
        "Improve the protocol itself"
    ]
    
    print("\nMaking entropy-guided decision...")
    decision = engine.entropy_decide(options)
    print(f"  Selected: {decision['selected']}")
    
    # Demo: full cycle
    print("\nRunning improvement cycle...")
    result = engine.run_improvement_cycle({
        "options": options,
        "gaps": [
            "No automated assessment",
            "No verification loop",
            "No distillation of lessons"
        ]
    })
    
    print(f"\nCycle completed. Steps: {list(result['steps'].keys())}")
    print(f"Suggested actions: {result['suggested_actions'][:2]}")
    
    print("\n" + "=" * 40)
    print("Engine ready for integration.")
