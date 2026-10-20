"""Run each GUI job in a fresh interpreter so code updates take effect."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid


def _stop_process_tree(child):
    if child.poll() is not None:
        return
    if os.name == 'nt':
        try:
            subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except (OSError, subprocess.TimeoutExpired):
            pass
        if child.poll() is None:
            child.kill()
    else:
        import signal
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_fresh(config, log=None, cancel_event=None):
    if cancel_event is not None and cancel_event.is_set():
        from autoedit.runtime import JobCancelled
        raise JobCancelled('Job cancelled before launch.')
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2"}
    video = config.get('video') or {}
    characters = video.get('characters') or {}
    objects = video.get('object_detection') or {}
    if (video.get('speech_device') == 'cpu'
            and (characters.get('enabled') is False or characters.get('device') == 'cpu')
            and (objects.get('enabled') is False or objects.get('device') == 'cpu')):
        env['CUDA_VISIBLE_DEVICES'] = '-1'
    out = Path(config.get('job', {}).get('output_dir') or './output').resolve()
    out.mkdir(parents=True, exist_ok=True)
    cancel_file = out / ('.cancel-' + uuid.uuid4().hex)
    config = {**config, '_cancel_file': str(cancel_file)}
    executable = Path(sys.executable)
    if executable.name.lower() == "pythonw.exe":
        executable = executable.with_name("python.exe")
    with subprocess.Popen(
        [str(executable), "-u", "-m", "autoedit.worker"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=env,
        start_new_session=os.name != 'nt',
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0),
    ) as child:
        done = threading.Event()
        def watch():
            while not done.wait(.2):
                if cancel_event is not None and cancel_event.is_set():
                    cancel_file.touch()
                    try:
                        if log:
                            log('Cancelling job; saving completed checkpoints and releasing resources...')
                    finally:
                        # A wedged native decoder/model cannot check a cooperative token.
                        if not done.wait(10):
                            _stop_process_tree(child)
                    return
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        recent = []
        try:
            child.stdin.write(json.dumps(config))
            child.stdin.close()
            for line in child.stdout:
                message = line.rstrip()
                recent = (recent + [message])[-8:]
                if log:
                    log(message)
            exit_code = child.wait()
        except BaseException:
            _stop_process_tree(child)
            raise
        finally:
            done.set()
            watcher.join(timeout=2)
            cancel_file.unlink(missing_ok=True)
        if cancel_event is not None and cancel_event.is_set():
            from autoedit.runtime import JobCancelled
            raise JobCancelled('Job cancelled; verified checkpoints are saved for the next run.')
        if exit_code:
            raise RuntimeError("Job failed:\n" + "\n".join(recent))
    path = Path(config.get("job", {}).get("output_dir") or "./output") / "edit-plan.json"
    return {"plan_path": str(path), "plan": json.loads(path.read_text(encoding="utf-8"))}


def main():
    import cv2
    cv2.setNumThreads(2)
    from autoedit.pipeline import run_pipeline
    run_pipeline(json.load(sys.stdin), log=lambda message: print(message, flush=True))


if __name__ == "__main__":
    main()
