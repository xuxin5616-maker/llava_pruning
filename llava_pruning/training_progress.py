"""Wall-clock training ETA, with separate train/validation speeds.

Pure reporting: no CUDA synchronization or changes to optimization/timing.
Recent sample costs adapt after cache warm-up; unseen phases use the current
speed provisionally. Estimates exclude unknown future checkpoint-save overhead.
"""

from collections import deque
import math
import time


def format_duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "--:--:--"
    seconds = max(0, math.ceil(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class TrainingETA:
    def __init__(self, epochs, phase_sizes, *, clock=None, window=20):
        if type(epochs) is not int or epochs < 1 or type(window) is not int or window < 1:
            raise ValueError("ETA epochs/window must be positive integers")
        if not phase_sizes or any(type(n) is not int or n < 1 for n in phase_sizes.values()):
            raise ValueError("ETA needs nonempty phases with positive sample counts")
        self.epochs, self.phase_sizes = epochs, dict(phase_sizes)
        self.phases = list(phase_sizes)
        self.clock, self.window = clock or time.perf_counter, window
        self.started = self.clock()
        self.speeds = {}
        self.epoch = self.phase = None

    def begin(self, epoch, phase):
        if not 1 <= epoch <= self.epochs or phase not in self.phase_sizes:
            raise ValueError("Unknown ETA epoch or phase")
        self.epoch, self.phase = epoch, phase
        self.last_index = 0
        self.last_time = self.clock()
        self.costs = deque(maxlen=self.window)

    def update(self, index):
        if self.phase is None or not self.last_index < index <= self.phase_sizes[self.phase]:
            raise ValueError("ETA sample indices must increase within the active phase")
        now = self.clock()
        self.costs.append(max(0., now - self.last_time) / (index - self.last_index))
        self.last_time, self.last_index = now, index
        rate = sum(self.costs) / len(self.costs)
        self.speeds[self.phase] = rate
        phase_left = (self.phase_sizes[self.phase] - index) * rate
        phase_position = self.phases.index(self.phase)
        remaining = phase_left
        provisional = False
        # Include every future training AND validation pass, not just this epoch.
        for epoch in range(self.epoch, self.epochs + 1):
            for position, phase in enumerate(self.phases):
                if epoch == self.epoch and position <= phase_position:
                    continue
                provisional |= phase not in self.speeds
                remaining += self.phase_sizes[phase] * self.speeds.get(phase, rate)
        epoch_samples = sum(self.phase_sizes.values())
        done = ((self.epoch - 1) * epoch_samples
                + sum(self.phase_sizes[p] for p in self.phases[:phase_position]) + index)
        return dict(elapsed=max(0., now - self.started), phase_eta=phase_left, total_eta=remaining,
                    phase_percent=100 * index / self.phase_sizes[self.phase],
                    overall_percent=100 * done / (self.epochs * epoch_samples), provisional=provisional)
