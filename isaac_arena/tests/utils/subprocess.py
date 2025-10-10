# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing as mp
import subprocess
import sys
import traceback
import uuid
from collections.abc import Callable

from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
from isaac_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


def run_subprocess(cmd, env=None):
    print(f"Running command: {cmd}")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            env=env,
            # Don't capture output, let it flow through in real-time
            capture_output=False,
            text=True,
            # Explicitly set stdout and stderr to None to use parent process's pipes
            stdout=None,
            stderr=None,
        )
        print(f"Command completed with return code: {result.returncode}")
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"Command failed with return code {e.returncode}: {e}\n")
        raise


def _worker_main(task_q: mp.Queue, result_q: mp.Queue, headless: bool, enable_cameras: bool):
    """Lives in a separate process. Creates SimulationAppContext once and serves tasks."""
    parser = get_isaac_arena_cli_parser()
    args = parser.parse_args([])
    args.headless = headless
    args.enable_cameras = enable_cameras

    # Create the simulation app ONCE
    with SimulationAppContext(args) as simulation_app:
        while True:
            task = task_q.get()
            if task is None:  # sentinel for clean shutdown
                break
            task_id = task["id"]
            func = task["func"]  # must be a top-level callable (pickleable)
            kwargs = task.get("kwargs", {})
            import omni.usd

            omni.usd.get_context().new_stage()
            try:
                ok = bool(func(simulation_app, **kwargs))
            except Exception as e:
                print(f"[sim-worker] Exception in task {task_id}: {e}", file=sys.stderr)
                traceback.print_exc()
                ok = False
            # return result to caller
            result_q.put((task_id, ok))


class SimulationAppWorker:
    """Client-side controller for the persistent simulation process."""

    def __init__(self, headless: bool = True, enable_cameras: bool = False):
        self.headless = headless
        self.enable_cameras = enable_cameras
        self._proc: mp.Process | None = None
        self._task_q: mp.Queue | None = None
        self._result_q: mp.Queue | None = None
        self._pending = {}

    def start(self):
        if self._proc and self._proc.is_alive():
            return
        mp.set_start_method("spawn", force=True)  # keep your CUDA note
        self._task_q = mp.Queue()
        self._result_q = mp.Queue()
        self._proc = mp.Process(
            target=_worker_main,
            args=(self._task_q, self._result_q, self.headless, self.enable_cameras),
            daemon=True,
        )
        self._proc.start()

    def stop(self, timeout: float = 30.0):
        if not self._proc:
            return
        if self._proc.is_alive():
            # ask worker to exit
            self._task_q.put(None)
            self._proc.join(timeout=timeout)
            if self._proc.is_alive():
                self._proc.terminate()
        self._proc = None
        self._task_q = None
        self._result_q = None
        self._pending.clear()

    def ensure_config(self, headless: bool, enable_cameras: bool):
        """Restart the worker if flags changed between tests."""
        if self._proc is None:
            self.headless = headless
            self.enable_cameras = enable_cameras
            self.start()
            return
        if self.headless != headless or self.enable_cameras != enable_cameras:
            self.stop()
            self.headless = headless
            self.enable_cameras = enable_cameras
            self.start()

    def run(self, func: Callable[..., bool], timeout: float | None = None, **kwargs) -> bool:
        if not (self._proc and self._proc.is_alive()):
            self.start()
        task_id = str(uuid.uuid4())
        self._task_q.put({"id": task_id, "func": func, "kwargs": kwargs})

        # If you only run tests serially, the next get() will be ours.
        # To be robust, handle out-of-order (e.g., if you add concurrency later).
        while True:
            rid, ok = self._result_q.get(timeout=timeout)
            if rid == task_id:
                return ok
            self._pending[rid] = ok  # stash unexpected (out-of-order) results


# ---- A simple module-level singleton and helper for easy use in tests ----
_global_worker: SimulationAppWorker | None = None


def run_in_persistent_sim(
    function: Callable[..., bool],
    headless: bool = True,
    enable_cameras: bool = False,
    timeout: float | None = None,
    **kwargs,
) -> bool:
    """Submit a function to the persistent SimulationApp process."""
    global _global_worker
    if _global_worker is None:
        _global_worker = SimulationAppWorker(headless=headless, enable_cameras=enable_cameras)
        _global_worker.start()
    else:
        _global_worker.ensure_config(headless, enable_cameras)
    return _global_worker.run(function, timeout=timeout, **kwargs)


def shutdown_persistent_sim():
    global _global_worker
    if _global_worker is not None:
        _global_worker.stop()
        _global_worker = None
