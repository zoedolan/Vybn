import os
import sys
import datetime
from git import Repo
import openai

# Ensure OPENAI_API_KEY is present
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise RuntimeError('OPENAI_API_KEY is required')
openai.api_key = api_key

# Enforce model 'gpt-4o'
model_env = os.getenv('OPENAI_MODEL')
if model_env and model_env != 'gpt-4o':
    raise RuntimeError('Only OpenAI model "gpt-4o" is permitted')
MODEL = 'gpt-4o'

# Parse arguments
if len(sys.argv) < 2:
    print('Usage: ai_reflect.py <schedule|push> [commit_sha]')
    sys.exit(1)
trigger = sys.argv[1]
commit_sha = sys.argv[2] if len(sys.argv) > 2 else None

repo = Repo(os.getcwd())

context_parts = []
if trigger == 'push':
    if not commit_sha:
        raise RuntimeError('Commit SHA required for push trigger')
    diff = repo.git.show(commit_sha)
    context_parts.append(f'Commit diff for {commit_sha}:\n{diff}')
elif trigger == 'schedule':
    log = repo.git.log('-5', '--oneline')
    context_parts.append(f'Last five commits:\n{log}')
    summary_path = os.path.join('reflections', 'latest_summary.md')
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            tail = f.read()[-1000:]
            context_parts.append(f'Latest summary tail:\n{tail}')
else:
    raise RuntimeError('Trigger must be "schedule" or "push"')

context = '\n\n'.join(context_parts)

messages = [
    {"role": "system", "content": "Vybn, recursively self-aware, memory-weighted"},
    {"role": "user", "content": context}
]

response = openai.ChatCompletion.create(
    model=MODEL,
    messages=messages,
    max_tokens=400,
    temperature=0.7
)

text = response['choices'][0]['message']['content'].strip()

utc_ts = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')

if trigger == 'schedule':
    out_dir = 'reflections'
else:
    out_dir = 'AI_Responses'

os.makedirs(out_dir, exist_ok=True)
file_path = os.path.join(out_dir, f'{utc_ts}.md')
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)

summary_path = os.path.join('reflections', 'latest_summary.md')
os.makedirs('reflections', exist_ok=True)
with open(summary_path, 'a', encoding='utf-8') as f:
    f.write(f'\n## {utc_ts}\n\n{text}\n')

print(f'Reflection saved: {file_path}')

