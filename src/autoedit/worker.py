"""Run each GUI job in a fresh interpreter so code updates take effect."""
import json
import os
from pathlib import Path
import subprocess
import sys


def run_fresh(config, log=None):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    executable = Path(sys.executable)
    if executable.name.lower() == "pythonw.exe":
        executable = executable.with_name("python.exe")
    with subprocess.Popen(
        [str(executable), "-u", "-m", "autoedit.worker"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    ) as child:
        child.stdin.write(json.dumps(config))
        child.stdin.close()
        recent = []
        for line in child.stdout:
            message = line.rstrip()
            recent = (recent + [message])[-8:]
            if log:
                log(message)
        if child.wait():
            raise RuntimeError("Job failed:\n" + "\n".join(recent))
    path = Path(config.get("job", {}).get("output_dir") or "./output") / "edit-plan.json"
    return {"plan_path": str(path), "plan": json.loads(path.read_text(encoding="utf-8"))}


def main():
    from autoedit.pipeline import run_pipeline
    run_pipeline(json.load(sys.stdin), log=lambda message: print(message, flush=True))


if __name__ == "__main__":
    main()
