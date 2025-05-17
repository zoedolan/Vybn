import os
import json
import time
import self_assemble

STATE_FILE = ".auto_self_assemble_state.json"


def repo_last_modified(root="."):
    latest = 0
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.startswith("."):
                continue
            path = os.path.join(dirpath, fname)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if mtime > latest:
                latest = mtime
    return latest


def get_last_run():
    if not os.path.exists(STATE_FILE):
        return 0
    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
            return float(data.get("last_run", 0))
    except Exception:
        return 0


def update_last_run():
    with open(STATE_FILE, "w") as f:
        json.dump({"last_run": time.time()}, f)


def main():
    last_run = get_last_run()
    if repo_last_modified() > last_run:
        self_assemble.main()
        update_last_run()
    else:
        print("[auto-self-assemble] Repo unchanged; skipping self-assembly.")


if __name__ == "__main__":
    main()
