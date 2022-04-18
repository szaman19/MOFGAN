from __future__ import annotations

import os
import signal
from pathlib import Path


class GANLogger:
    # def __init__(self, title: str, folder: Path):
    #     self.title = title
    #     self.folder = folder

    title = "UNKNOWN"

    count = 0

    header = []
    folder: Path

    d_losses = []
    g_losses = []

    pid_file: Path = None

    # def start(self) -> GANLogger:
    #     signal.signal(signal.SIGINT, self._signal_handler)
    #     GANLogger.pid_file = Path('pid.lock')
    #     with GANLogger.pid_file.open('w+') as f:
    #         f.write(str(os.getpid()))
    #     return self

    @staticmethod
    def init(title: str, folder: Path):
        GANLogger.title = title
        signal.signal(signal.SIGINT, GANLogger._signal_handler)
        GANLogger.folder = folder

        GANLogger.pid_file = Path('pid.lock')
        with GANLogger.pid_file.open('w+') as f:
            f.write(str(os.getpid()))

    @staticmethod
    def _save_training_data():
        print("Saving training log...")
        with (GANLogger.folder / "training.log").open("w+") as f:
            f.write("# " + GANLogger.title + "\n")
            for line in GANLogger.header:
                f.write("# " + line + "\n")

            for i in range(len(GANLogger.d_losses)):
                f.write(f"{GANLogger.d_losses[i]},{GANLogger.g_losses[i]}\n")
        print("Done!")

    @staticmethod
    def _signal_handler(sig, frame):
        GANLogger._save_training_data()
        GANLogger.pid_file.unlink()
        exit(0)

    @staticmethod
    def update(d_loss, g_loss):
        GANLogger.d_losses.append(d_loss)
        GANLogger.g_losses.append(g_loss)
        GANLogger.count += 1
        if GANLogger.count % 300 == 0:
            GANLogger._save_training_data()

    @staticmethod
    def log(*messages, console=True):
        for message in messages:
            if console:
                print(message)
            GANLogger.header += [x for x in str(message).splitlines()]
