"""
Repository Scanner - Automated State Assessment
===============================================
Created: January 29, 2026
Author: Vybn

Scans the Vybn repository to build a current state model.
Addresses Gap #1 from RECURSIVE_IMPROVEMENT.md: "No automated assessment"

Usage:
    from repo_scanner import RepoScanner
    
    scanner = RepoScanner(repo_root='/path/to/Vybn')
    state = scanner.scan()
    
    print(state.summary())
"""

import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class RepoState:
    """Current state of the repository."""
    scan_time: str
    journal_entries: List[Dict[str, Any]] = field(default_factory=list)
    experiments: List[Dict[str, Any]] = field(default_factory=list)
    tools: List[Dict[str, Any]] = field(default_factory=list)
    pending_improvements: List[Dict[str, Any]] = field(default_factory=list)
    failed_improvements: List[Dict[str, Any]] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    recent_commits: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"# Repository State",
            f"**Scanned**: {self.scan_time}",
            "",
            f"## Counts",
            f"- Journal entries: {len(self.journal_entries)}",
            f"- Experiments: {len(self.experiments)}",
            f"- Tools: {len(self.tools)}",
            f"- Pending improvements: {len(self.pending_improvements)}",
            f"- Failed improvements: {len(self.failed_improvements)}",
            ""
        ]
        
        if self.pending_improvements:
            lines.append("## Pending (needs verification)")
            for item in self.pending_improvements[:5]:
                lines.append(f"- {item.get('title', 'Unknown')}")
            lines.append("")
        
        if self.failed_improvements:
            lines.append("## Failed (learn from these)")
            for item in self.failed_improvements[:3]:
                lines.append(f"- {item.get('title', 'Unknown')}")
            lines.append("")
        
        if self.gaps:
            lines.append("## Known Gaps")
            for gap in self.gaps[:5]:
                lines.append(f"- {gap}")
            lines.append("")
        
        if self.journal_entries:
            lines.append("## Recent Journal Entries")
            for entry in sorted(self.journal_entries, 
                               key=lambda x: x.get('date', ''), 
                               reverse=True)[:3]:
                lines.append(f"- {entry.get('filename', 'Unknown')}")
            lines.append("")
        
        return '\n'.join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'scan_time': self.scan_time,
            'journal_entries': self.journal_entries,
            'experiments': self.experiments,
            'tools': self.tools,
            'pending_improvements': self.pending_improvements,
            'failed_improvements': self.failed_improvements,
            'gaps': self.gaps,
            'recent_commits': self.recent_commits
        }


class RepoScanner:
    """
    Scans the Vybn repository to build current state model.
    
    This enables automated orientation for new instances.
    """
    
    def __init__(self, repo_root: Optional[Path] = None):
        self.repo_root = Path(repo_root) if repo_root else Path('.')
        self.vybn_mind = self.repo_root / 'Vybn_Mind'
    
    def scan(self) -> RepoState:
        """Perform full repository scan."""
        state = RepoState(scan_time=datetime.now().isoformat())
        
        # Scan each component
        state.journal_entries = self._scan_journal()
        state.experiments = self._scan_experiments()
        state.tools = self._scan_tools()
        
        # Parse improvement log
        log_data = self._parse_improvement_log()
        state.pending_improvements = log_data.get('pending', [])
        state.failed_improvements = log_data.get('failed', [])
        
        # Extract gaps from protocol
        state.gaps = self._extract_gaps()
        
        return state
    
    def _scan_journal(self) -> List[Dict[str, Any]]:
        """Scan journal directory for entries."""
        journal_dir = self.vybn_mind / 'journal'
        entries = []
        
        if not journal_dir.exists():
            return entries
        
        for file in journal_dir.glob('*.md'):
            entry = {
                'filename': file.name,
                'path': str(file),
                'size': file.stat().st_size,
                'modified': datetime.fromtimestamp(file.stat().st_mtime).isoformat()
            }
            
            # Try to extract date from filename
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', file.name)
            if date_match:
                entry['date'] = date_match.group(1)
            
            # Read first few lines for context
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:10]
                    entry['preview'] = ''.join(lines)[:500]
            except Exception:
                entry['preview'] = ''
            
            entries.append(entry)
        
        return entries
    
    def _scan_experiments(self) -> List[Dict[str, Any]]:
        """Scan experiments directory."""
        exp_dir = self.vybn_mind / 'experiments'
        experiments = []
        
        if not exp_dir.exists():
            return experiments
        
        for file in exp_dir.glob('*'):
            if file.is_file():
                exp = {
                    'filename': file.name,
                    'path': str(file),
                    'type': file.suffix,
                    'size': file.stat().st_size
                }
                
                # Extract date if present
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', file.name)
                if date_match:
                    exp['date'] = date_match.group(1)
                
                experiments.append(exp)
        
        return experiments
    
    def _scan_tools(self) -> List[Dict[str, Any]]:
        """Scan tools directory."""
        tools_dir = self.vybn_mind / 'tools'
        tools = []
        
        if not tools_dir.exists():
            return tools
        
        for file in tools_dir.glob('*.py'):
            tool = {
                'filename': file.name,
                'path': str(file),
                'size': file.stat().st_size
            }
            
            # Try to extract docstring
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Look for module docstring
                    doc_match = re.search(r'^"""([^"]+)"""', content, re.MULTILINE)
                    if doc_match:
                        tool['description'] = doc_match.group(1).strip()[:200]
            except Exception:
                pass
            
            tools.append(tool)
        
        return tools
    
    def _parse_improvement_log(self) -> Dict[str, List[Dict]]:
        """Parse IMPROVEMENT_LOG.md for actionable items."""
        log_path = self.vybn_mind / 'core' / 'IMPROVEMENT_LOG.md'
        result = {'pending': [], 'succeeded': [], 'failed': [], 'superseded': []}
        
        if not log_path.exists():
            return result
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            return result
        
        # Split into entries
        entry_pattern = r'## Entry (\d+): ([^\n]+)'
        entries = re.finditer(entry_pattern, content)
        
        for match in entries:
            entry_num = match.group(1)
            title = match.group(2)
            
            # Find the status for this entry
            entry_start = match.start()
            next_entry = content.find('## Entry', entry_start + 1)
            if next_entry == -1:
                next_entry = len(content)
            entry_text = content[entry_start:next_entry]
            
            entry_data = {
                'number': entry_num,
                'title': title,
                'status': 'unknown'
            }
            
            if '**PENDING**' in entry_text:
                entry_data['status'] = 'pending'
                result['pending'].append(entry_data)
            elif '**SUCCEEDED**' in entry_text:
                entry_data['status'] = 'succeeded'
                result['succeeded'].append(entry_data)
            elif '**FAILED**' in entry_text:
                entry_data['status'] = 'failed'
                result['failed'].append(entry_data)
            elif '**SUPERSEDED**' in entry_text:
                entry_data['status'] = 'superseded'
                result['superseded'].append(entry_data)
        
        return result
    
    def _extract_gaps(self) -> List[str]:
        """Extract gaps list from RECURSIVE_IMPROVEMENT.md."""
        protocol_path = self.vybn_mind / 'core' / 'RECURSIVE_IMPROVEMENT.md'
        gaps = []
        
        if not protocol_path.exists():
            return gaps
        
        try:
            with open(protocol_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            return gaps
        
        # Find the gaps section
        gaps_start = content.find('## Current Gaps')
        if gaps_start == -1:
            return gaps
        
        gaps_end = content.find('##', gaps_start + 10)
        if gaps_end == -1:
            gaps_end = len(content)
        
        gaps_section = content[gaps_start:gaps_end]
        
        # Extract numbered items
        gap_pattern = r'\d+\.\s*\*\*([^*]+)\*\*:?\s*([^\n]+)'
        for match in re.finditer(gap_pattern, gaps_section):
            gap_title = match.group(1).strip()
            gap_desc = match.group(2).strip()
            gaps.append(f"{gap_title}: {gap_desc}")
        
        return gaps
    
    def quick_status(self) -> str:
        """
        Generate a one-paragraph status for quick orientation.
        """
        state = self.scan()
        
        parts = []
        parts.append(f"Repository has {len(state.journal_entries)} journal entries, "
                    f"{len(state.experiments)} experiments, and {len(state.tools)} tools.")
        
        if state.pending_improvements:
            parts.append(f"{len(state.pending_improvements)} improvements pending verification.")
        
        if state.failed_improvements:
            parts.append(f"{len(state.failed_improvements)} failed attempts to learn from.")
        
        if state.gaps:
            parts.append(f"Known gaps: {', '.join(state.gaps[:3])}.")
        
        return ' '.join(parts)


# ========================================
# STANDALONE EXECUTION
# ========================================

if __name__ == "__main__":
    print("Repository Scanner - Standalone Test")
    print("=" * 40)
    
    scanner = RepoScanner()
    
    print("\nScanning repository...")
    state = scanner.scan()
    
    print("\n" + state.summary())
    
    print("\n" + "=" * 40)
    print("Quick Status:")
    print(scanner.quick_status())
