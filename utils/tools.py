from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str
from time import time
import subprocess


def print_debug(*args):
    print()
    print('----- DEBUG -----')
    for a in args:
        print(a)
    print('----- DEBUG -----')
    print()


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


class Chrono:
    def __init__(self):
        self.start_time = 0.0
        self.duration = 0.0

    def start(self):
        self.start_time = time()

    def stop(self, message: str = None):
        self.duration = time() - self.start_time
        if message is not None:
            print(message, self.duration, 'secs', '\n')

    def reset(self):
        self.start_time = 0.0
        self.duration = 0.0


def print_flops(model, input, rank):
    flops = FlopCountAnalysis(model, input)
    print("Flops table", rank)
    print(rank, flop_count_table(flops))
    print(rank, flop_count_str(flops))
    print(" ", rank)
