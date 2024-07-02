import time

from emote.extra.system_logger import SystemLogger


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

    # we allocated 2 M ints, each 28 bytes large.
    assert logger.scalar_logs["system/ram_usage_growth_mb_step"] > (2 * 28)
    # we just pinned the thread for 1 second so this should be close to 100
    assert logger.scalar_logs["system/cpu_load"] > 10.0
