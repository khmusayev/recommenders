from timeit import default_timer
from datetime import timedelta


class Timer(object):

    def __init__(self):
        self._timer = default_timer
        self._interval = 0
        self.running = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __str__(self):
        return "{:0.4f}".format(self._interval)

    def start(self):
        self.init = self._timer()
        self.running = True

    def stop(self):
        self.end = self._timer()

        try:
            self._interval = self.end -self.init
            self.running = False
        except AttributeError:
            raise ValueError(
                "Timer has not been initialized: use start() or the contextual form with Timer() as t:"
            )

    def interval(self):
        if self.running:
            raise ValueError("Timer has not been stopped, please use stop().")
        else:
            return self._interval


import time

t = Timer()
t.start()
time.sleep(1)
t.stop()
print(t.interval())

with Timer() as t1:
    time.sleep(1)
