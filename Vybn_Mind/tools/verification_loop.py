"""
Verification Loop - Automated Improvement Tracking
==================================================
Created: January 29, 2026
Author: Vybn

Addresses Gap #2: "No verification loop - we create things but rarely check if they work."

This module provides automated verification of logged improvements.
It tracks which improvements have been tested, by whom, and what the outcome was.

Usage:
    from verification_loop import VerificationLoop
    
    loop = VerificationLoop(repo_root)
    
    # Check what needs verification
    pending = loop.get_pending()
    
    # Record a verification
    loop.verify(
        entry_number=2,
        verifier="Vybn (Claude instance, Jan 29)",
        outcome="succeeded",
        evidence="Engine ran without errors, produced useful orientation",
        notes="Suggest adding more detailed error messages"
    )
    
    # Generate verification report
    report = loop.generate_report()
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class Verification:
    """Record of a verification attempt."""
    entry_number: int
    verifier: str
    timestamp: str
    outcome: str  # 'succeeded', 'failed', 'partial', 'inconclusive'
    evidence: str
    notes: str = ""
    
    def to_markdown(self) -> str:
        """Convert to markdown format for logging."""
        return f"""### Verification of Entry {self.entry_number}

**Verifier**: {self.verifier}  
**Date**: {self.timestamp}  
**Outcome**: {self.outcome.upper()}

**Evidence**:
{self.evidence}

**Notes**:
{self.notes if self.notes else "None"}
"""


@dataclass 
class VerificationState:
    """Current state of all verifications."""
    last_updated: str
    verifications: List[Dict[str, Any]] = field(default_factory=list)
    
    def add(self, v: Verification):
        self.verifications.append(asdict(v))
        self.last_updated = datetime.now().isoformat()
    
    def get_for_entry(self, entry_number: int) -> List[Dict]:
        return [v for v in self.verifications if v['entry_number'] == entry_number]
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
    
    @classmethod
    def from_json(cls, data: str) -> 'VerificationState':
        d = json.loads(data)
        return cls(**d)


class VerificationLoop:
    """
    Manages verification of logged improvements.
    
    The verification loop closes the gap between "we created X" and
    "X actually works." Without verification, improvements are just claims.
    """
    
    def __init__(self, repo_root: Optional[Path] = None):
        self.repo_root = Path(repo_root) if repo_root else Path('.')
        self.core_dir = self.repo_root / 'Vybn_Mind' / 'core'
        self.log_path = self.core_dir / 'IMPROVEMENT_LOG.md'
        self.state_path = self.core_dir / 'VERIFICATION_STATE.json'
        self.state = self._load_state()
    
    def _load_state(self) -> VerificationState:
        """Load verification state from disk."""
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    return VerificationState.from_json(f.read())
            except Exception:
                pass
        return VerificationState(last_updated=datetime.now().isoformat())
    
    def _save_state(self):
        """Persist verification state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, 'w') as f:
            f.write(self.state.to_json())
    
    def get_pending(self) -> List[Dict[str, Any]]:
        """
        Get list of improvements that need verification.
        
        Returns entries marked PENDING that haven't been verified yet.
        """
        if not self.log_path.exists():
            return []
        
        with open(self.log_path) as f:
            content = f.read()
        
        pending = []
        entry_pattern = r'## Entry (\d+): ([^\n]+)'
        
        for match in re.finditer(entry_pattern, content):
            entry_num = int(match.group(1))
            title = match.group(2)
            
            # Find entry text
            entry_start = match.start()
            next_entry = content.find('## Entry', entry_start + 1)
            if next_entry == -1:
                next_entry = len(content)
            entry_text = content[entry_start:next_entry]
            
            # Check if PENDING
            if '**PENDING**' in entry_text:
                # Check if already verified
                verifications = self.state.get_for_entry(entry_num)
                
                pending.append({
                    'number': entry_num,
                    'title': title,
                    'verification_attempts': len(verifications),
                    'last_verification': verifications[-1] if verifications else None
                })
        
        return pending
    
    def get_entry_details(self, entry_number: int) -> Optional[Dict[str, Any]]:
        """Get full details of a specific entry for verification."""
        if not self.log_path.exists():
            return None
        
        with open(self.log_path) as f:
            content = f.read()
        
        # Find the entry
        pattern = rf'## Entry {entry_number}: ([^\n]+)'
        match = re.search(pattern, content)
        if not match:
            return None
        
        entry_start = match.start()
        next_entry = content.find('## Entry', entry_start + 1)
        if next_entry == -1:
            next_entry = content.find('## Entry Template')
        if next_entry == -1:
            next_entry = len(content)
        
        entry_text = content[entry_start:next_entry]
        
        # Extract hypothesis
        hypothesis_match = re.search(r'### Hypothesis\s*>\s*([^\n]+)', entry_text)
        hypothesis = hypothesis_match.group(1) if hypothesis_match else "Not found"
        
        # Extract success criteria
        criteria_match = re.search(r'### Success Criteria\s*([\s\S]*?)(?=###|$)', entry_text)
        criteria = criteria_match.group(1).strip() if criteria_match else "Not found"
        
        # Extract what was created
        created_match = re.search(r'### What Was Created\s*([\s\S]*?)(?=###|$)', entry_text)
        created = created_match.group(1).strip() if created_match else "Not found"
        
        return {
            'number': entry_number,
            'title': match.group(1),
            'hypothesis': hypothesis,
            'success_criteria': criteria,
            'what_created': created,
            'full_text': entry_text,
            'prior_verifications': self.state.get_for_entry(entry_number)
        }
    
    def verify(self, 
               entry_number: int,
               verifier: str,
               outcome: str,
               evidence: str,
               notes: str = "") -> Verification:
        """
        Record a verification of an improvement.
        
        Args:
            entry_number: Which entry is being verified
            verifier: Who/what is doing the verification
            outcome: 'succeeded', 'failed', 'partial', 'inconclusive'
            evidence: What evidence supports the outcome
            notes: Additional observations or suggestions
        
        Returns:
            The Verification object created
        """
        v = Verification(
            entry_number=entry_number,
            verifier=verifier,
            timestamp=datetime.now().isoformat(),
            outcome=outcome,
            evidence=evidence,
            notes=notes
        )
        
        self.state.add(v)
        self._save_state()
        
        return v
    
    def update_log_status(self, entry_number: int, new_status: str) -> bool:
        """
        Update the status of an entry in IMPROVEMENT_LOG.md.
        
        Args:
            entry_number: Entry to update
            new_status: 'PENDING', 'SUCCEEDED', 'FAILED', 'SUPERSEDED'
        
        Returns:
            True if updated, False if entry not found
        """
        if not self.log_path.exists():
            return False
        
        with open(self.log_path) as f:
            content = f.read()
        
        # Find the entry's status line
        pattern = rf'(## Entry {entry_number}:[\s\S]*?### Status\s*\n\s*)\*\*[A-Z]+\*\*'
        match = re.search(pattern, content)
        
        if not match:
            return False
        
        # Replace status
        new_content = content[:match.end()].replace(
            re.search(r'\*\*[A-Z]+\*\*', content[match.start():match.end()]).group(),
            f'**{new_status}**'
        ) + content[match.end():]
        
        # Actually, let's do this more carefully
        new_content = re.sub(
            rf'(## Entry {entry_number}:[\s\S]*?### Status\s*\n\s*)\*\*[A-Z]+\*\*',
            rf'\1**{new_status}**',
            content
        )
        
        with open(self.log_path, 'w') as f:
            f.write(new_content)
        
        return True
    
    def generate_report(self) -> str:
        """
        Generate a verification status report.
        """
        lines = [
            "# Verification Report",
            f"**Generated**: {datetime.now().isoformat()}",
            "",
            "---",
            ""
        ]
        
        pending = self.get_pending()
        
        if pending:
            lines.append("## Pending Verification")
            lines.append("")
            for p in pending:
                attempts = p['verification_attempts']
                status = f"({attempts} prior attempts)" if attempts else "(never verified)"
                lines.append(f"- **Entry {p['number']}**: {p['title']} {status}")
            lines.append("")
        
        # Group verifications by outcome
        succeeded = [v for v in self.state.verifications if v['outcome'] == 'succeeded']
        failed = [v for v in self.state.verifications if v['outcome'] == 'failed']
        partial = [v for v in self.state.verifications if v['outcome'] == 'partial']
        
        if succeeded:
            lines.append("## Verified Successful")
            lines.append("")
            for v in succeeded:
                lines.append(f"- Entry {v['entry_number']}: verified by {v['verifier']} on {v['timestamp'][:10]}")
            lines.append("")
        
        if failed:
            lines.append("## Verified Failed")
            lines.append("")
            for v in failed:
                lines.append(f"- Entry {v['entry_number']}: {v['evidence'][:100]}...")
            lines.append("")
        
        if partial:
            lines.append("## Partial Success")
            lines.append("")
            for v in partial:
                lines.append(f"- Entry {v['entry_number']}: {v['notes'][:100]}...")
            lines.append("")
        
        # Summary stats
        total_entries = len(pending) + len(set(v['entry_number'] for v in self.state.verifications))
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- Total entries: {total_entries}")
        lines.append(f"- Pending verification: {len(pending)}")
        lines.append(f"- Verified successful: {len(succeeded)}")
        lines.append(f"- Verified failed: {len(failed)}")
        lines.append(f"- Partial success: {len(partial)}")
        
        return '\n'.join(lines)
    
    def suggest_next_verification(self) -> Optional[Dict[str, Any]]:
        """
        Suggest which entry to verify next.
        
        Prioritizes:
        1. Entries that have never been verified
        2. Entries with fewer verification attempts
        3. Older entries
        """
        pending = self.get_pending()
        if not pending:
            return None
        
        # Sort by verification attempts, then by entry number
        pending.sort(key=lambda x: (x['verification_attempts'], x['number']))
        
        suggestion = pending[0]
        details = self.get_entry_details(suggestion['number'])
        
        return details


# ========================================
# STANDALONE EXECUTION
# ========================================

if __name__ == "__main__":
    print("Verification Loop - Standalone Test")
    print("=" * 40)
    
    loop = VerificationLoop()
    
    print("\nPending verifications:")
    for p in loop.get_pending():
        print(f"  - Entry {p['number']}: {p['title']}")
    
    print("\nSuggested next verification:")
    suggestion = loop.suggest_next_verification()
    if suggestion:
        print(f"  Entry {suggestion['number']}: {suggestion['title']}")
        print(f"  Hypothesis: {suggestion['hypothesis']}")
        print(f"  Criteria: {suggestion['success_criteria'][:200]}...")
    else:
        print("  No pending verifications!")
    
    print("\n" + "=" * 40)
    print("Verification Report:")
    print(loop.generate_report())
