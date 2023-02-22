import time

import torch

from emote.extras.system_logger import SystemLogger


def test_records_all_metrics():
    logger = SystemLogger()
    logger.end_cycle(bp_step=0, bp_samples=0)
    data = list(range(2 * 1024 * 1024))
    start = time.perf_counter()
    i = 0
    while (time.perf_counter() - start) < 1:
        data[i % (2 * 1024 * 1024)] += 1
        i += 1
    logger.end_cycle(bp_step=1, bp_samples=1000)
    del data

    assert logger.scalar_logs is None
