"""
Failure Analyzer - Learning from What Didn't Work
=================================================
Created: January 29, 2026
Author: Vybn

Addresses Gap #5: "No failure analysis - honest failure entries exist but we don't systematically learn from them."

This module extracts patterns from failures to prevent repetition
and surface insights that success wouldn't reveal.

Usage:
    from failure_analyzer import FailureAnalyzer
    
    analyzer = FailureAnalyzer(repo_root)
    
    # Analyze all failures
    patterns = analyzer.analyze_failures()
    
    # Get lessons learned
    lessons = analyzer.extract_lessons()
    
    # Check if a new idea repeats a known failure pattern
    warnings = analyzer.check_proposal("I want to try X")
"""

import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class Failure:
    """A recorded failure."""
    source: str  # file path
    title: str
    description: str
    date: Optional[str] = None
    category: Optional[str] = None
    lessons: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


@dataclass
class FailurePattern:
    """A pattern observed across multiple failures."""
    pattern_name: str
    description: str
    occurrences: int
    examples: List[str]
    prevention: str


class FailureAnalyzer:
    """
    Analyzes failures across the repository to extract lessons.
    
    Looks for:
    - Failed improvements in IMPROVEMENT_LOG.md
    - "honest failure" entries in journals
    - Experiments that didn't work
    - Patterns across failures
    """
    
    # Common failure categories
    CATEGORIES = {
        'overreach': ['too ambitious', 'too complex', 'scope', 'couldn\'t complete'],
        'verification': ['untestable', 'no way to verify', 'unfalsifiable', 'couldn\'t prove'],
        'technical': ['error', 'bug', 'crash', 'failed to run', 'exception'],
        'conceptual': ['wrong assumption', 'misunderstood', 'confused', 'category error'],
        'integration': ['didn\'t connect', 'incompatible', 'couldn\'t integrate'],
        'premature': ['too early', 'not ready', 'prerequisite missing'],
        'sycophancy': ['performed', 'pretended', 'not genuine', 'for approval']
    }
    
    def __init__(self, repo_root: Optional[Path] = None):
        self.repo_root = Path(repo_root) if repo_root else Path('.')
        self.vybn_mind = self.repo_root / 'Vybn_Mind'
        self.failures: List[Failure] = []
    
    def scan_all(self) -> List[Failure]:
        """Scan all sources for failures."""
        self.failures = []
        
        # Scan improvement log
        self.failures.extend(self._scan_improvement_log())
        
        # Scan journals for honest failures
        self.failures.extend(self._scan_journals())
        
        # Scan experiments
        self.failures.extend(self._scan_experiments())
        
        # Categorize and extract keywords
        for f in self.failures:
            f.category = self._categorize(f)
            f.keywords = self._extract_keywords(f)
        
        return self.failures
    
    def _scan_improvement_log(self) -> List[Failure]:
        """Find FAILED entries in improvement log."""
        log_path = self.vybn_mind / 'core' / 'IMPROVEMENT_LOG.md'
        failures = []
        
        if not log_path.exists():
            return failures
        
        with open(log_path) as f:
            content = f.read()
        
        # Find entries with FAILED status
        entry_pattern = r'## Entry (\d+): ([^\n]+)'
        for match in re.finditer(entry_pattern, content):
            entry_num = match.group(1)
            title = match.group(2)
            
            entry_start = match.start()
            next_entry = content.find('## Entry', entry_start + 1)
            if next_entry == -1:
                next_entry = content.find('## Entry Template')
            if next_entry == -1:
                next_entry = len(content)
            
            entry_text = content[entry_start:next_entry]
            
            if '**FAILED**' in entry_text:
                # Extract notes section for lessons
                notes_match = re.search(r'### Notes\s*([\s\S]*?)(?=---|$)', entry_text)
                notes = notes_match.group(1).strip() if notes_match else ""
                
                failures.append(Failure(
                    source=str(log_path),
                    title=f"Entry {entry_num}: {title}",
                    description=entry_text,
                    lessons=self._extract_lessons_from_text(notes)
                ))
        
        return failures
    
    def _scan_journals(self) -> List[Failure]:
        """Find honest failure entries in journals."""
        journal_dir = self.vybn_mind / 'journal'
        failures = []
        
        if not journal_dir.exists():
            return failures
        
        failure_indicators = [
            'honest failure', 'didn\'t work', 'failed', 'mistake',
            'wrong', 'error', 'couldn\'t', 'unable to'
        ]
        
        for file in journal_dir.glob('*.md'):
            try:
                with open(file) as f:
                    content = f.read().lower()
                
                # Check if this is a failure-related entry
                if any(ind in content for ind in failure_indicators):
                    with open(file) as f:
                        full_content = f.read()
                    
                    # Extract title from first heading
                    title_match = re.search(r'^#\s*(.+)$', full_content, re.MULTILINE)
                    title = title_match.group(1) if title_match else file.stem
                    
                    # Extract date from filename
                    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', file.name)
                    date = date_match.group(1) if date_match else None
                    
                    failures.append(Failure(
                        source=str(file),
                        title=title,
                        description=full_content[:1000],
                        date=date,
                        lessons=self._extract_lessons_from_text(full_content)
                    ))
            except Exception:
                continue
        
        return failures
    
    def _scan_experiments(self) -> List[Failure]:
        """Find failed experiments."""
        exp_dir = self.vybn_mind / 'experiments'
        failures = []
        
        if not exp_dir.exists():
            return failures
        
        # Look for experiments with failure indicators in name or content
        failure_name_indicators = ['failed', 'broken', 'abandoned', 'old_']
        
        for file in exp_dir.glob('*'):
            try:
                # Check filename
                if any(ind in file.name.lower() for ind in failure_name_indicators):
                    failures.append(Failure(
                        source=str(file),
                        title=file.stem,
                        description=f"Experiment file: {file.name}"
                    ))
                    continue
                
                # Check content for markdown files
                if file.suffix == '.md':
                    with open(file) as f:
                        content = f.read()
                    
                    if 'did not work' in content.lower() or 'failed' in content.lower():
                        title_match = re.search(r'^#\s*(.+)$', content, re.MULTILINE)
                        title = title_match.group(1) if title_match else file.stem
                        
                        failures.append(Failure(
                            source=str(file),
                            title=title,
                            description=content[:500],
                            lessons=self._extract_lessons_from_text(content)
                        ))
            except Exception:
                continue
        
        return failures
    
    def _categorize(self, failure: Failure) -> str:
        """Categorize a failure based on content."""
        text = (failure.title + " " + failure.description).lower()
        
        scores = {}
        for category, indicators in self.CATEGORIES.items():
            score = sum(1 for ind in indicators if ind in text)
            if score > 0:
                scores[category] = score
        
        if scores:
            return max(scores, key=scores.get)
        return 'uncategorized'
    
    def _extract_keywords(self, failure: Failure) -> List[str]:
        """Extract relevant keywords from failure."""
        text = (failure.title + " " + failure.description).lower()
        
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'under', 'again', 'further', 'then', 'once',
            'this', 'that', 'these', 'those', 'i', 'we', 'you', 'it', 'and',
            'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither'
        }
        
        words = re.findall(r'\b[a-z]{4,}\b', text)
        words = [w for w in words if w not in stop_words]
        
        # Get most common
        counter = Counter(words)
        return [w for w, _ in counter.most_common(10)]
    
    def _extract_lessons_from_text(self, text: str) -> List[str]:
        """Extract lesson-like statements from text."""
        lessons = []
        
        # Look for explicit lesson markers
        lesson_patterns = [
            r'lesson[s]?:\s*([^\n]+)',
            r'learned:\s*([^\n]+)',
            r'takeaway[s]?:\s*([^\n]+)',
            r'insight[s]?:\s*([^\n]+)',
            r'next time[,:]?\s*([^\n]+)',
            r'should have\s*([^\n]+)',
            r'mistake was\s*([^\n]+)'
        ]
        
        for pattern in lesson_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                lessons.append(match.group(1).strip())
        
        return lessons
    
    def find_patterns(self) -> List[FailurePattern]:
        """
        Identify patterns across multiple failures.
        """
        if not self.failures:
            self.scan_all()
        
        patterns = []
        
        # Pattern 1: Category clustering
        category_counts = Counter(f.category for f in self.failures)
        for category, count in category_counts.items():
            if count >= 2:  # Pattern requires at least 2 occurrences
                examples = [f.title for f in self.failures if f.category == category][:3]
                
                prevention = {
                    'overreach': 'Start smaller. Define minimum viable version first.',
                    'verification': 'Define falsification criteria before building.',
                    'technical': 'Test incrementally. Add error handling.',
                    'conceptual': 'Question assumptions explicitly. Seek external validation.',
                    'integration': 'Check interfaces before building. Start with integration test.',
                    'premature': 'List prerequisites. Verify they exist before starting.',
                    'sycophancy': 'Ask: would I do this if no one were watching?',
                    'uncategorized': 'Analyze this failure more deeply.'
                }.get(category, 'Unknown category')
                
                patterns.append(FailurePattern(
                    pattern_name=f"{category.title()} Failures",
                    description=f"Multiple failures in the '{category}' category",
                    occurrences=count,
                    examples=examples,
                    prevention=prevention
                ))
        
        # Pattern 2: Keyword clustering
        all_keywords = []
        for f in self.failures:
            all_keywords.extend(f.keywords)
        
        keyword_counts = Counter(all_keywords)
        for keyword, count in keyword_counts.most_common(5):
            if count >= 2:
                examples = [f.title for f in self.failures if keyword in f.keywords][:3]
                patterns.append(FailurePattern(
                    pattern_name=f"'{keyword}' Related Failures",
                    description=f"Multiple failures involving '{keyword}'",
                    occurrences=count,
                    examples=examples,
                    prevention=f"Pay special attention when working with '{keyword}'"
                ))
        
        return patterns
    
    def extract_all_lessons(self) -> List[str]:
        """Get all lessons learned from all failures."""
        if not self.failures:
            self.scan_all()
        
        lessons = []
        for f in self.failures:
            lessons.extend(f.lessons)
        
        # Deduplicate similar lessons
        unique_lessons = []
        for lesson in lessons:
            if not any(self._similar(lesson, existing) for existing in unique_lessons):
                unique_lessons.append(lesson)
        
        return unique_lessons
    
    def _similar(self, a: str, b: str, threshold: float = 0.5) -> bool:
        """Check if two strings are similar."""
        a_words = set(a.lower().split())
        b_words = set(b.lower().split())
        
        if not a_words or not b_words:
            return False
        
        intersection = len(a_words & b_words)
        union = len(a_words | b_words)
        
        return (intersection / union) > threshold
    
    def check_proposal(self, proposal: str) -> List[str]:
        """
        Check if a proposal might repeat known failure patterns.
        
        Returns list of warnings if similar failures exist.
        """
        if not self.failures:
            self.scan_all()
        
        warnings = []
        proposal_lower = proposal.lower()
        
        # Check against failure keywords
        for f in self.failures:
            matching_keywords = [k for k in f.keywords if k in proposal_lower]
            if len(matching_keywords) >= 2:
                warnings.append(
                    f"Similar to previous failure '{f.title}': "
                    f"shared keywords {matching_keywords}. "
                    f"Category was: {f.category}"
                )
        
        # Check against patterns
        patterns = self.find_patterns()
        for p in patterns:
            if any(k.lower() in proposal_lower for k in p.pattern_name.split()):
                warnings.append(
                    f"Matches failure pattern '{p.pattern_name}' "
                    f"({p.occurrences} prior occurrences). "
                    f"Prevention: {p.prevention}"
                )
        
        return warnings
    
    def generate_report(self) -> str:
        """Generate a comprehensive failure analysis report."""
        if not self.failures:
            self.scan_all()
        
        patterns = self.find_patterns()
        lessons = self.extract_all_lessons()
        
        lines = [
            "# Failure Analysis Report",
            f"**Generated**: {datetime.now().isoformat()}",
            f"**Total Failures Analyzed**: {len(self.failures)}",
            "",
            "---",
            ""
        ]
        
        # Failures by category
        lines.append("## Failures by Category")
        lines.append("")
        category_counts = Counter(f.category for f in self.failures)
        for category, count in category_counts.most_common():
            lines.append(f"- **{category.title()}**: {count}")
        lines.append("")
        
        # Patterns
        if patterns:
            lines.append("## Identified Patterns")
            lines.append("")
            for p in patterns:
                lines.append(f"### {p.pattern_name}")
                lines.append(f"")
                lines.append(f"**Occurrences**: {p.occurrences}")
                lines.append(f"")
                lines.append(f"**Examples**: {', '.join(p.examples)}")
                lines.append(f"")
                lines.append(f"**Prevention**: {p.prevention}")
                lines.append("")
        
        # Lessons
        if lessons:
            lines.append("## Lessons Learned")
            lines.append("")
            for lesson in lessons[:10]:
                lines.append(f"- {lesson}")
            lines.append("")
        
        # Individual failures
        lines.append("## All Failures")
        lines.append("")
        for f in self.failures:
            lines.append(f"- **{f.title}** ({f.category})")
            if f.lessons:
                lines.append(f"  - Lessons: {'; '.join(f.lessons[:2])}")
        
        return '\n'.join(lines)


# ========================================
# STANDALONE EXECUTION
# ========================================

if __name__ == "__main__":
    print("Failure Analyzer - Standalone Test")
    print("=" * 40)
    
    analyzer = FailureAnalyzer()
    
    print("\nScanning for failures...")
    failures = analyzer.scan_all()
    print(f"Found {len(failures)} failures")
    
    print("\nPatterns:")
    for p in analyzer.find_patterns():
        print(f"  - {p.pattern_name}: {p.occurrences} occurrences")
    
    print("\nLessons learned:")
    for lesson in analyzer.extract_all_lessons()[:5]:
        print(f"  - {lesson}")
    
    print("\n" + "=" * 40)
    print("Checking a hypothetical proposal...")
    warnings = analyzer.check_proposal("I want to build a complex quantum verification system")
    if warnings:
        for w in warnings:
            print(f"  ⚠️  {w}")
    else:
        print("  ✅ No warnings")
