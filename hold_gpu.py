"""Hold a fixed amount of GPU memory; no model and no continuous GPU workload.

Examples (run inside an environment with CUDA-enabled PyTorch):
    python hold_gpu.py --gpu 5 --memory-gb 8
    python hold_gpu.py --gpu 5 --memory-gb 8 --seconds 3600

--gpu sets CUDA_VISIBLE_DEVICES before importing torch. The selected GPU is
cuda:0 inside this process. Use only on GPUs you are authorized to reserve.
Memory is released when the process exits, including Ctrl+C / SIGTERM.
"""

import argparse
import math
import os
import signal
import sys
import threading
import time
from contextlib import suppress


GIB = 1024 ** 3
CHUNK_BYTES = 256 * 1024 ** 2
HEADROOM_BYTES = 512 * 1024 ** 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", type=int, required=True,
                        help="GPU index to expose via CUDA_VISIBLE_DEVICES (e.g. 5).")
    parser.add_argument("--memory-gb", type=float, default=1.0,
                        help="Memory to allocate in GiB, excluding CUDA context (default: 1).")
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="Hold duration after allocation; 0 = until Ctrl+C (default: 0).")
    args = parser.parse_args(argv)
    if args.gpu < 0:
        parser.error("--gpu must be nonnegative.")
    if not math.isfinite(args.memory_gb) or args.memory_gb <= 0:
        parser.error("--memory-gb must be finite and positive.")
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error("--seconds must be finite and nonnegative.")
    requested_bytes = int(args.memory_gb * GIB)
    if requested_bytes < 1:
        parser.error("--memory-gb must request at least one byte.")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    try:
        import torch
    except ImportError:
        print("ERROR: Install/use an environment with CUDA-enabled PyTorch.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print(f"ERROR: CUDA unavailable for GPU {args.gpu}; check the index and PyTorch.",
              file=sys.stderr)
        return 1

    stop = threading.Event()
    previous_handlers = {}
    tensors = []
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, lambda *_: stop.set())
        torch.cuda.set_device(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if requested_bytes > max(0, free_bytes - HEADROOM_BYTES):
            print(
                f"ERROR: Requested {args.memory_gb:.3f} GiB, but only "
                f"{free_bytes / GIB:.3f} GiB is free. Keep at least 0.5 GiB headroom; "
                "reduce --memory-gb. No holder tensors were allocated.", file=sys.stderr)
            return 1
        print(f"PID={os.getpid()} | GPU {args.gpu}: {torch.cuda.get_device_name(0)} | "
              f"free={free_bytes / GIB:.3f}/{total_bytes / GIB:.3f} GiB", flush=True)

        remaining = requested_bytes
        while remaining > 0 and not stop.is_set():
            chunk_bytes = min(CHUNK_BYTES, remaining)
            # uint8 has one byte per element. Touch memory once, then remain idle.
            tensors.append(torch.empty(chunk_bytes, dtype=torch.uint8, device="cuda:0"))
            tensors[-1].zero_()
            remaining -= chunk_bytes
        torch.cuda.synchronize()
        if stop.is_set():
            return 0

        duration = "until Ctrl+C" if args.seconds == 0 else f"{args.seconds:g} seconds"
        print(f"Holding {requested_bytes / GIB:.3f} GiB for {duration}. "
              "Memory only; GPU compute utilization may be near 0%. "
              "Ctrl+C to release.", flush=True)
        deadline = time.monotonic() + args.seconds if args.seconds else None
        while not stop.is_set():
            wait_seconds = 1.0
            if deadline is not None:
                wait_seconds = min(wait_seconds, max(0.0, deadline - time.monotonic()))
                if wait_seconds == 0:
                    break
            stop.wait(wait_seconds)
        return 0
    except RuntimeError as exc:
        print(f"ERROR: GPU allocation failed: {exc}\n"
              "Free memory may have changed; reduce --memory-gb and retry.", file=sys.stderr)
        return 1
    finally:
        tensors.clear()
        with suppress(Exception):
            torch.cuda.empty_cache()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        print("Holder stopped; allocated tensors released. CUDA context releases on exit.",
              flush=True)


if __name__ == "__main__":
    sys.exit(main())
