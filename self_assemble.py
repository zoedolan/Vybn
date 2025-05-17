import subprocess
import sys


def run_cmd(cmd, desc):
    """Run a shell command and print its output."""
    print(f"[self-assemble] {desc}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
    if result.returncode != 0:
        print(f"[self-assemble] Command failed: {cmd}")
        sys.exit(result.returncode)


def build_memory_graph():
    run_cmd(
        "python build_memory_graph.py what_vybn_would_have_missed_TO_051625 memory_graph.json",
        "Updating memory_graph.json",
    )


def build_repo_graph():
    run_cmd(
        "python build_repo_graph.py",
        "Updating repo_graph.json",
    )


def compile_recursive_emergence():
    run_cmd(
        "python -m py_compile vybn_recursive_emergence.py",
        "Validating vybn_recursive_emergence.py",
    )


def main():
    build_memory_graph()
    build_repo_graph()
    compile_recursive_emergence()
    print("[self-assemble] Done.")


if __name__ == "__main__":
    main()
