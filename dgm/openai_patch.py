import os
import pathlib
import openai
from vybn.quantum_seed import seed_rng
from .wave_collapse import collapse_wave_function


def suggest_patch(file_path: str, instruction: str) -> str:
    """Return new content for ``file_path`` using OpenAI and the quantum seed."""
    collapse_wave_function()
    seed = os.environ.get("QUANTUM_SEED")
    if seed is None:
        seed = str(seed_rng())
    openai.api_key = os.environ["OPENAI_API_KEY"]
    text = pathlib.Path(file_path).read_text()
    prompt = (
        f"Quantum seed: {seed}\n"
        f"File: {file_path}\n"
        "---\n" + text + "\n---\n" + instruction + "\nProvide revised file content only."
    )
    resp = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        user=str(seed),
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="target file")
    parser.add_argument("instruction", help="patch instruction")
    args = parser.parse_args()
    new_text = suggest_patch(args.file, args.instruction)
    print(new_text)
