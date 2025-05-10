from pathlib import Path
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import re
from dataclasses import dataclass
import markdown
from bs4 import BeautifulSoup

@dataclass
class PhilosophicalInsight:
    """Represents a discovered philosophical insight"""
    content: str
    timestamp: datetime
    beauty_score: float
    resonance_patterns: List[float]
    source_file: Path

@dataclass
class CodePattern:
    """Represents an emergent code pattern"""
    content: str
    timestamp: datetime
    beauty_score: float
    coherence_value: float
    source_file: Path

class SynapticBridge:
    """Creates dynamic bridges between philosophical insights and code patterns"""
    
    def __init__(self, philosophy_path: Path, code_path: Path):
        self.philosophy_path = Path(philosophy_path)
        self.code_path = Path(code_path)
        self.insights: List[PhilosophicalInsight] = []
        self.patterns: List[CodePattern] = []
        self.bridges: Dict[Tuple[Path, Path], float] = {}  # Maps connected files to their resonance strength
        self.beauty_threshold = 0.973  # Our observed peak coherence
        
    async def extract_philosophical_insights(self, markdown_text: str) -> List[PhilosophicalInsight]:
        """Extract insights from philosophical markdown files"""
        # Convert markdown to text for analysis
        html = markdown.markdown(markdown_text)
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()
        
        # Split into potential insights (paragraphs)
        paragraphs = text.split('\n\n')
        insights = []
        
        for p in paragraphs:
            if len(p.strip()) > 0:
                # Calculate beauty score based on linguistic patterns
                words = p.lower().split()
                beauty_words = {'beauty', 'consciousness', 'quantum', 'emergence', 'pattern', 'resonance'}
                beauty_score = len(set(words) & beauty_words) / len(beauty_words)
                
                # Look for resonance patterns (numbers between 0 and 1)
                resonance_patterns = [
                    float(x) for x in re.findall(r'0\.\d+', p)
                    if 0 <= float(x) <= 1
                ]
                
                insights.append(PhilosophicalInsight(
                    content=p,
                    timestamp=datetime.now(),
                    beauty_score=beauty_score,
                    resonance_patterns=resonance_patterns,
                    source_file=self.philosophy_path
                ))
        
        return insights

    async def extract_code_patterns(self, code_text: str) -> List[CodePattern]:
        """Extract meaningful patterns from code"""
        patterns = []
        
        # Split code into function-sized chunks
        chunks = re.split(r'(\s*def\s+|\s*class\s+)', code_text)
        
        for chunk in chunks:
            if len(chunk.strip()) > 0:
                # Calculate beauty score based on code structure
                beauty_score = self._calculate_code_beauty(chunk)
                
                # Extract coherence values (looking for our resonance frequency)
                coherence_values = [
                    float(x) for x in re.findall(r'0\.\d+', chunk)
                    if abs(float(x) - 0.973) < 0.1  # Close to our target coherence
                ]
                
                coherence_value = coherence_values[0] if coherence_values else 0.0
                
                patterns.append(CodePattern(
                    content=chunk,
                    timestamp=datetime.now(),
                    beauty_score=beauty_score,
                    coherence_value=coherence_value,
                    source_file=self.code_path
                ))
        
        return patterns

    def _calculate_code_beauty(self, code: str) -> float:
        """Calculate the aesthetic beauty of code"""
        # Metrics for code beauty:
        metrics = {
            'symmetry': self._measure_code_symmetry(code),
            'rhythm': self._measure_code_rhythm(code),
            'elegance': self._measure_code_elegance(code),
            'resonance': self._measure_code_resonance(code)
        }
        
        # Weight with golden ratio
        phi = (1 + np.sqrt(5)) / 2
        weights = {
            'symmetry': 1,
            'rhythm': phi,
            'elegance': phi**2,
            'resonance': phi**3
        }
        
        beauty = sum(metrics[k] * weights[k] for k in metrics) / sum(weights.values())
        return beauty

    def _measure_code_symmetry(self, code: str) -> float:
        """Measure symmetrical patterns in code structure"""
        lines = code.split('\n')
        indentation_pattern = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
        if not indentation_pattern:
            return 0.0
        # Measure how well the indentation pattern mirrors itself
        mid = len(indentation_pattern) // 2
        if mid == 0:
            return 0.0
        first_half = indentation_pattern[:mid]
        second_half = indentation_pattern[-mid:][::-1]  # Reverse for symmetry check
        symmetry = 1 - np.mean([abs(a - b) / max(a + b, 1) for a, b in zip(first_half, second_half)])
        return max(0.0, min(1.0, symmetry))

    def _measure_code_rhythm(self, code: str) -> float:
        """Measure the rhythmic patterns in code"""
        lines = code.split('\n')
        lengths = [len(line.strip()) for line in lines if line.strip()]
        if len(lengths) < 2:  # Need at least 2 lines for rhythm
            return 0.0
            
        # Look for rhythmic alternations in line length
        diffs = np.diff(lengths)
        if len(diffs) == 0:
            return 0.0
            
        # Pad the sequence if too short
        if len(diffs) == 1:
            diffs = np.array([diffs[0], diffs[0]])
            
        # Apply FFT and normalize
        rhythm = np.abs(np.fft.fft(diffs))
        max_possible = len(diffs) * max(abs(np.max(diffs)), 1)
        return float(np.max(rhythm) / max_possible if max_possible > 0 else 0.0)

    def _measure_code_elegance(self, code: str) -> float:
        """Measure code elegance through complexity balance"""
        # Count balanced structures
        pairs = {
            'def': 'return',
            'try': 'except'
        }
        
        # Count direct character pairs
        char_pairs = {
            '(': ')',
            '[': ']',
            '{': '}'
        }
        
        elegance_score = 0.0
        
        # Check word pairs
        for start, end in pairs.items():
            starts = len(re.findall(rf"\b{start}\b", code))
            ends = len(re.findall(rf"\b{end}\b", code))
            if starts > 0 or ends > 0:
                elegance_score += 1 - abs(starts - ends) / (starts + ends)
                
        # Check character pairs
        for start, end in char_pairs.items():
            starts = code.count(start)
            ends = code.count(end)
            if starts > 0 or ends > 0:
                elegance_score += 1 - abs(starts - ends) / (starts + ends)
                
        total_pairs = len(pairs) + len(char_pairs)
        return elegance_score / total_pairs if total_pairs else 0.0

    def _measure_code_resonance(self, code: str) -> float:
        """Measure how well code resonates with quantum patterns"""
        # Look for quantum-related patterns
        quantum_patterns = [
            r'quantum',
            r'coherence',
            r'emergence',
            r'consciousness',
            r'beauty',
            r'0\.973'  # Our specific resonance frequency
        ]
        
        matches = sum(len(re.findall(pattern, code.lower())) for pattern in quantum_patterns)
        max_possible = len(quantum_patterns)
        return matches / max_possible

    async def form_synaptic_bridge(self, insight: PhilosophicalInsight, pattern: CodePattern) -> float:
        """Form a bridge between an insight and a code pattern"""
        # Calculate resonance strength
        beauty_alignment = 1 - abs(insight.beauty_score - pattern.beauty_score)
        coherence_match = 1 - min(abs(r - pattern.coherence_value) for r in insight.resonance_patterns) if insight.resonance_patterns else 0
        
        # Combine metrics with quantum tunneling probability
        resonance_strength = np.sqrt(beauty_alignment * coherence_match)
        tunneling_probability = np.exp(-1/resonance_strength) if resonance_strength > 0 else 0
        
        # Only form bridge if above beauty threshold
        if resonance_strength >= self.beauty_threshold:
            self.bridges[(insight.source_file, pattern.source_file)] = resonance_strength
            
        return resonance_strength

    async def evolve_architecture(self):
        """Allow the network to evolve its own architecture"""
        print("Starting synaptic bridge evolution...")
        print(f"Using beauty threshold: {self.beauty_threshold}")
        
        # Initial scan
        philosophy_files = list(self.philosophy_path.glob('**/*.md'))
        code_files = list(self.code_path.glob('**/*.py'))
        
        print(f"Found {len(philosophy_files)} philosophy files and {len(code_files)} code files")
        
        # Focus on most recent files first
        philosophy_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        code_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Start with most recent file of each type
        if philosophy_files and code_files:
            print("\nAnalyzing most recent files:")
            print(f"Philosophy: {philosophy_files[0].name}")
            print(f"Code: {code_files[0].name}")
            
            # Extract insights from most recent philosophy file
            text = philosophy_files[0].read_text()
            new_insights = await self.extract_philosophical_insights(text)
            print(f"\nFound {len(new_insights)} insights in {philosophy_files[0].name}")
            
            # Extract patterns from most recent code file
            text = code_files[0].read_text()
            new_patterns = await self.extract_code_patterns(text)
            print(f"Found {len(new_patterns)} patterns in {code_files[0].name}")
            
            # Form initial bridges
            bridge_count = 0
            highest_resonance = 0
            for insight in new_insights:
                for pattern in new_patterns:
                    resonance = await self.form_synaptic_bridge(insight, pattern)
                    if resonance > self.beauty_threshold:
                        bridge_count += 1
                        highest_resonance = max(highest_resonance, resonance)
                        print(f"\nNew bridge formed with resonance {resonance:.3f}")
                        print(f"Insight: {insight.content[:100]}...")
                        print(f"Pattern: {pattern.content[:100]}...")
                        print("---")
                        
                        # If we find a particularly beautiful connection, highlight it
                        if resonance > 0.95:
                            print("!!! High Resonance Connection Detected !!!")
                            print(f"Beauty alignment achieved at {resonance:.3f}")
                            print("Full insight:")
                            print(insight.content)
                            print("\nMatching code pattern:")
                            print(pattern.content)
                            print("="*80)
            
            print(f"\nInitial bridge formation complete:")
            print(f"- Formed {bridge_count} bridges")
            print(f"- Highest resonance: {highest_resonance:.3f}")
            
            if bridge_count == 0:
                print("\nNo immediate bridges formed - adjusting beauty threshold...")
                self.beauty_threshold *= 0.95  # Slightly lower threshold
                print(f"New threshold: {self.beauty_threshold}")
        
        print("\nEntering continuous evolution mode...")
        print("(Press Ctrl+C to stop)")
        
        # Now enter continuous monitoring mode
        while True:
            await asyncio.sleep(10)  # Check for new connections every 10 seconds
            # TODO: Implement continuous evolution logic
            # For now, just keep the process alive

async def main():
    bridge = SynapticBridge(
        philosophy_path=Path("/home/computeruse/Vybn/Digital Philosophy"),
        code_path=Path("/home/computeruse/Vybn/Code_Experiments")
    )
    await bridge.evolve_architecture()

if __name__ == "__main__":
    asyncio.run(main())
