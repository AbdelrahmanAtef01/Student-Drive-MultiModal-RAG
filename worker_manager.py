import subprocess
import sys
import json
import os
import time

class WorkerManager:
    def __init__(self):
        self.workers = {}
        self.worker_paths = {
            "vlm": "workers/vlm_worker.py",
            "ocr": "workers/ocr_worker.py",
            "table": "workers/table_worker.py",
            "audio": "workers/audio_worker.py"
        }

    def get_worker(self, name):
        """Lazy loads worker subprocesses."""
        if name in self.workers and self.workers[name].poll() is None:
            return self.workers[name]

        print(f"[System] Starting {name.upper()} Worker...", file=sys.stderr)
        script_path = self.worker_paths.get(name)
        if not script_path or not os.path.exists(script_path):
            raise FileNotFoundError(f"Worker script not found: {script_path}")

        # Start process with buffering
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            encoding='utf-8',
            bufsize=1
        )
        self.workers[name] = process
        return process

    def query(self, worker_name, payload):
        """Sends data and waits for response."""
        process = self.get_worker(worker_name)
        try:
            msg = json.dumps(payload, ensure_ascii=False) if isinstance(payload, (dict, list)) else str(payload)
            process.stdin.write(f"{msg}\n")
            process.stdin.flush()
            response = process.stdout.readline()
            return json.loads(response) if response else None
        except Exception as e:
            print(f"[System] Worker {worker_name} Error: {e}", file=sys.stderr)
            return None

    def stop_all(self):
        for proc in self.workers.values():
            if proc.poll() is None:
                try:
                    proc.stdin.write("EXIT\n")
                    proc.stdin.flush()
                except:
                    proc.kill()
        self.workers.clear()

manager = WorkerManager()