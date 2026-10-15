"""Read-only plane-search benchmark on user SWCs, including warm summary reuse."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from swclib.visualization.hemisphere import hemisphere_counts
from swclib.visualization.loading import SWCReaders
from swclib.visualization.plane_optimization import optimize_plane


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeat-entries', type=int, default=1)
    args = parser.parse_args()
    paths = sorted(Path(args.path).glob('*.swc'))
    assert paths, 'No SWC files'
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_mode) for p in paths}
    entries = [dict(path=str(p)) for p in paths] * args.repeat_entries
    readers = SWCReaders(workers=4)
    measurements = []
    original_batch = readers._neuron_batch
    def measured(entries, worker, *args, **kwargs):
        start = time.perf_counter()
        result = original_batch(entries, worker, *args, **kwargs)
        measurement = dict(worker=worker.__name__, files=len(entries), seconds=round(time.perf_counter()-start, 3))
        measurements.append(measurement)
        print(json.dumps(measurement), flush=True)
        return result
    readers._neuron_batch = measured
    try:
        runs = []
        for label in ['initial', 'cached']:
            start = time.perf_counter()
            summaries = readers.plane_summaries(entries)
            summary_seconds = time.perf_counter()-start
            boxes = np.asarray([s['source_bounds'] for s in summaries])
            center = (boxes[:, 0].min(axis=0)+boxes[:, 1].max(axis=0))/2
            result = optimize_plane(dict(azimuth_deg=0, elevation_deg=0, offset=float(center[0])), center, 10,
                                    lambda planes: readers.plane_counts(entries, planes, summaries))
            run = dict(label=label, seconds=round(time.perf_counter()-start, 3),
                       summary_seconds=round(summary_seconds, 3), result=result)
            runs.append(run)
            print(json.dumps(run, ensure_ascii=False), flush=True)
        # Independently recompute the selected plane with the existing export
        # lengths on every requested neuron, outside the timed search.
        plane = {key: result['plane'][key] for key in ('azimuth_deg', 'elevation_deg', 'offset')}
        assert hemisphere_counts(readers.hemispheres(entries, plane, include_soma=False)) == result['counts']
        assert all(snapshot == (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_mode) for p, snapshot in before.items())
        report = dict(source=str(Path(args.path)), unique_files=len(paths), entries=len(entries),
                      repeated_entries=args.repeat_entries, nodes=sum(s['node_count'] for s in summaries),
                      workers=readers.workers, summary_cache_mib=readers._summary_bytes/1048576,
                      measurement='Search includes source summaries and exact fallbacks; excludes worker startup, rendering and final independent verification. Cached run reuses summaries.',
                      runs=runs, batches=measurements, exact_export_counts_match=True, source_metadata_unchanged=True)
        report['cpu_model'] = next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                                    if line.startswith('model name')), 'unknown')
        report['cpu_affinity'] = len(os.sched_getaffinity(0))
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps({key: value for key, value in report.items() if key not in ('runs', 'batches')}, ensure_ascii=False), flush=True)
    finally:
        readers.close()


if __name__ == '__main__':
    main()
