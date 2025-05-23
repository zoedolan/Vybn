import os
import subprocess
from typing import Optional


class GraphBuilder:
    """Utility class to build memory, memoir, and repo graphs."""

    def __init__(self, script_dir: Optional[str] = None, repo_root: Optional[str] = None):
        self.script_dir = script_dir or os.path.join(os.path.dirname(__file__), '..', '..', 'self_assembly')
        self.repo_root = repo_root or os.path.dirname(os.path.dirname(self.script_dir))

    def _run(self, cmd: str, desc: str) -> None:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(result.stderr.strip())
        if result.returncode != 0:
            raise RuntimeError(f"{desc} failed: {cmd}")

    def build_memory_graph(self) -> str:
        script = os.path.join(self.script_dir, 'build_memory_graph.py')
        output = os.path.join(self.script_dir, 'memory_graph.json')
        # The personal history log was moved out of the legacy folder, so point
        # directly to the current location. This prevents "File not found"
        # errors during the self-assembly process.
        memory_input = os.path.join(
            self.repo_root,
            'personal_history',
            'what_vybn_would_have_missed_TO_051625',
        )
        cmd = f"python {script} {memory_input} {output}"
        self._run(cmd, 'memory graph')
        return output

    def build_memoir_graph(self) -> str:
        script = os.path.join(self.script_dir, 'build_memoir_graph.py')
        output = os.path.join(self.script_dir, 'memoir_graph.json')
        memoir_input = os.path.join(self.repo_root, "Zoe's Memoirs")
        cmd = f"python {script} \"{memoir_input}\" {output}"
        self._run(cmd, 'memoir graph')
        return output

    def build_repo_graph(self) -> str:
        script = os.path.join(self.script_dir, 'build_repo_graph.py')
        output = os.path.join(self.script_dir, 'repo_graph.json')
        cmd = f"python {script} {self.repo_root} {output}"
        self._run(cmd, 'repo graph')
        return output

    def update_all(self) -> None:
        self.build_memory_graph()
        self.build_memoir_graph()
        self.build_repo_graph()

