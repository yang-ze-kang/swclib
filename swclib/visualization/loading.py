"""Bounded CPU-parallel SWC decoding, with read-only workers and binary IPC."""

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from collections import OrderedDict
import multiprocessing
import os
from pathlib import Path
import signal
import threading
import time

import numpy as np

from .data import MIB, available_memory, swc_packet
from .readonly import ReadOnlyGuard, prepare_scipy_memory_stream
from .plugins import soma_candidates
from .hemisphere import (candidate_counts, neuron_plane_summary, neuron_refined_classes,
                         neuron_statistics, plane_parameters, summary_plane_classes)


def _initialize_worker(started):
    # The console parent owns Ctrl+C and drains its pool during shutdown.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    prepare_scipy_memory_stream()
    ReadOnlyGuard().install()
    # Warm the existing parser once, under the same protection as source reads.
    from swclib.data.swc import Swc  # noqa: F401
    started.wait()


def _ready():
    return os.getpid()


def _read_packet(path, crop):
    # Transfer packed arrays, never a pickled graph of per-node dictionaries.
    return b"".join(swc_packet(path, crop))


def _read_packet_with_summary(path, crop):
    from swclib.data.swc import Swc
    swc = Swc(str(path))
    try:
        summary = neuron_plane_summary(path, swc=swc)
    except ValueError:
        # The viewer permits missing parents; hemisphere tools report these
        # errors when invoked. Preparing a summary must not change display I/O.
        summary = None
    return b"".join(swc_packet(path, crop, swc=swc)), summary


def _source_signature(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


def worker_count(requested=None):
    if requested is not None:
        if not isinstance(requested, int) or not 1 <= requested <= 8:
            raise ValueError("swc-workers 必须为 1..8 的整数")
        return requested
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    return max(1, min(4, cpus, available_memory() // (256 * MIB)))


class SWCReaders:
    def __init__(self, workers=None):
        self.workers = worker_count(workers)
        self.active = 0
        self.reserved = 0
        self.lock = threading.Lock()
        self._summary_cache = OrderedDict()
        self._summary_bytes = 0
        self._summary_lock = threading.Lock()
        context = multiprocessing.get_context("spawn")
        started = context.Event()
        self.pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=context,
                                        initializer=_initialize_worker, initargs=(started,))
        try:
            # Eagerly create all workers in the unrestricted startup thread.
            # The gate prevents an early idle worker from suppressing a spawn.
            ready = [self.pool.submit(_ready) for _ in range(self.workers)]
            started.set()
            for future in ready:
                future.result(timeout=60)
        except BaseException:
            started.set()
            self.close()
            raise

    def acquire(self, path):
        # Conservative estimate for parser dictionaries, cropped copies, NumPy
        # staging arrays and IPC/HTTP copies. This is not a strict RSS guarantee.
        cost = max(8 * MIB, Path(path).stat().st_size * 32)
        with self.lock:
            budget = min(2 * 1024 ** 3, int(available_memory() * .20))
            if self.active >= self.workers:
                return None
            if self.reserved + cost > budget:
                if self.active:
                    return None
                raise ValueError("SWC 读取内存估计超过当前预算，请释放系统内存后重试")
            self.active += 1
            self.reserved += cost
        return cost

    def release(self, cost):
        with self.lock:
            self.active -= 1
            self.reserved -= cost

    def read(self, path, crop):
        signature = _source_signature(path)
        if self._cached_summary(path, signature) is not None:
            return [self.pool.submit(_read_packet, path, crop).result()]
        packet, summary = self.pool.submit(_read_packet_with_summary, path, crop).result()
        if summary is not None:
            self._remember_summary(path, signature, summary)
        return [packet]

    def somas(self, path, rule):
        return self.pool.submit(soma_candidates, path, rule).result()

    def hemispheres(self, entries, plane, *, include_soma=True):
        return self._neuron_batch(entries, neuron_statistics, plane, include_soma=include_soma)

    def plane_summaries(self, entries):
        # Repeated angle searches reuse only bounded summaries, never full large
        # morphologies. Ordinary display reads prepare these same summaries.
        summaries, missing = [None] * len(entries), []
        for i, entry in enumerate(entries):
            path = entry["path"]
            signature = _source_signature(path)
            cached = self._cached_summary(path, signature)
            if cached is not None:
                summaries[i] = cached
            else:
                missing.append((i, signature))
        if missing:
            results = self._neuron_batch([entries[i] for i, _ in missing], neuron_plane_summary)
            for (i, signature), summary in zip(missing, results):
                summaries[i] = summary
                self._remember_summary(entries[i]["path"], signature, summary)
        return summaries

    def _cached_summary(self, path, signature):
        with self._summary_lock:
            cached = self._summary_cache.pop(path, None)
            if cached is not None:
                if cached[0] == signature:
                    self._summary_cache[path] = cached
                    return cached[1]
                self._summary_bytes -= cached[2]
        return None

    def _remember_summary(self, path, signature, summary):
        budget = min(64 * MIB, int(available_memory() * .02))
        cost = sum(getattr(value, "nbytes", 0) for value in summary.values()) + 1024
        if cost > budget:
            return
        with self._summary_lock:
            previous = self._summary_cache.pop(path, None)
            if previous is not None:
                self._summary_bytes -= previous[2]
            self._summary_cache[path] = (signature, summary, cost)
            self._summary_bytes += cost
            while self._summary_bytes > budget:
                _, (_, _, removed) = self._summary_cache.popitem(last=False)
                self._summary_bytes -= removed

    def plane_counts(self, entries, planes, summaries):
        parsed = [plane_parameters(value) for value in planes]
        prepared = parsed, np.asarray([p["normal_xyz"] for p in parsed]), np.asarray([p["offset"] for p in parsed])
        classes = [summary_plane_classes(summary, planes, prepared=prepared) for summary in summaries]
        uncertain = [(i, (values == 0).nonzero()[0]) for i, values in enumerate(classes) if (values == 0).any()]
        if uncertain:
            results = self._neuron_batch(
                [entries[i] for i, _ in uncertain], neuron_refined_classes,
                argument_sets=[([planes[j] for j in indices], min(1024, summaries[i].get("detail", 64)*4))
                               for i, indices in uncertain],
            )
            for (i, indices), (result, summary) in zip(uncertain, results):
                classes[i][indices] = result
                summaries[i] = summary
                self._remember_summary(entries[i]["path"], _source_signature(entries[i]["path"]), summary)
        return candidate_counts(classes)

    def _neuron_batch(self, entries, worker, *args, argument_sets=None, **kwargs):
        """Bounded batch using the same admission budget as ordinary reads."""
        results = [None] * len(entries)
        pending, cursor = {}, 0
        try:
            while cursor < len(entries) or pending:
                while cursor < len(entries):
                    entry = entries[cursor]
                    reservation = self.acquire(entry["path"])
                    if reservation is None:
                        break
                    try:
                        arguments = argument_sets[cursor] if argument_sets is not None else args
                        future = self.pool.submit(worker, entry["path"], *arguments, **kwargs)
                    except BaseException:
                        self.release(reservation)
                        raise
                    pending[future] = (cursor, reservation)
                    cursor += 1
                if not pending:
                    time.sleep(.05)
                    continue
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    index, reservation = pending.pop(future)
                    try:
                        results[index] = future.result()
                    finally:
                        self.release(reservation)
        finally:
            # A bad neuron aborts the batch; drain already-started reads.
            for future, (_, reservation) in pending.items():
                future.cancel()
                try:
                    future.result()
                except Exception:
                    pass
                finally:
                    self.release(reservation)
        return results

    def close(self):
        self._summary_cache.clear()
        self._summary_bytes = 0
        pool, self.pool = self.pool, None
        if pool is not None:
            # Drop executor/event references in the unrestricted shutdown
            # thread, not later during GC in a read-only HTTP request thread.
            pool.shutdown(wait=True)
