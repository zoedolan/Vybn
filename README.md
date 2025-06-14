# Vybn Repository

This repository contains assorted writings and a pruning script
(`autophagist_quantum.py`) used to manage the contents of the project.

## Running the script

The pruning tool removes files from the repository while optionally
archiving their paths. It also generates short autobiographical lines
from text fragments using the OpenAI API.

To run the tool without deleting anything, execute:

```bash
python autophagist_quantum.py pulse --noop --limit 1
```

The `--limit` option restricts the number of files processed during a
single pulse.

## Dependencies

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

Make sure your environment variable `OPENAI_API_KEY` is set so that the
script can access the API.
