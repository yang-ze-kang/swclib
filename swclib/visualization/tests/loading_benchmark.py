"""Compare warm HTTP SWC throughput with one vs multiple protected processes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import statistics
import tempfile
import threading
import time
import urllib.error
import urllib.request

from swclib.visualization.server import create_server


def benchmark(root, files=32, nodes=10000, repeats=3, parallel=4):
    sources=root/'swcs';sources.mkdir(exist_ok=True)
    for i in range(files):
        (sources/f'{i:04}.swc').write_text(''.join(
            f'{j+1} 3 {j%300+i*.1:.3f} {j%211:.3f} {j%197:.3f} 1 {j if j else -1}\n'
            for j in range(nodes)))
    servers=[];results={};expected=None
    try:
        for workers in [1,parallel]:
            start=time.perf_counter();server=create_server(port=0,swc_workers=workers)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            servers.append((workers,server,thread))
            entries=server.catalog.add(sources,'swc')
            results[workers]={'startup_seconds':round(time.perf_counter()-start,3),'seconds':[],'entries':entries}

        def load(server,entry):
            req=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/api/swc',
                data=json.dumps({'id':entry['id']}).encode(),
                headers={'Content-Type':'application/json','X-Viewer-Token':server.token})
            for _ in range(200):
                try:
                    with urllib.request.urlopen(req) as response:
                        return hashlib.sha256(response.read()).hexdigest()
                except urllib.error.HTTPError as error:
                    if error.code!=409:raise
                    time.sleep(.005)
            raise RuntimeError('Reader queue did not drain')

        # Warm filesystem/import caches once per configuration; alternate trials.
        for trial in range(repeats+1):
            for workers,server,_ in (servers if trial%2==0 else servers[::-1]):
                start=time.perf_counter()
                with ThreadPoolExecutor(workers) as clients:
                    hashes=list(clients.map(lambda e:load(server,e),results[workers]['entries']))
                elapsed=time.perf_counter()-start
                if expected is None:expected=hashes
                assert hashes==expected,'parallel read changed the binary morphology payload'
                if trial:results[workers]['seconds'].append(round(elapsed,4))
        for value in results.values():
            del value['entries']
            value['median_seconds']=round(statistics.median(value['seconds']),4)
        return {'files':files,'nodes_per_file':nodes,'total_nodes':files*nodes,
                'cpu_affinity':len(os.sched_getaffinity(0)), 'warm_cache':True,
                'measurement':'HTTP parsing, binary IPC, transfer and SHA256; excludes browser/GPU and startup',
                'runs':results,'speedup':round(results[1]['median_seconds']/results[parallel]['median_seconds'],2)}
    finally:
        for _,server,thread in servers:
            server.shutdown();server.server_close();thread.join(timeout=5)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output');parser.add_argument('--files',type=int,default=32)
    parser.add_argument('--nodes',type=int,default=10000);parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args();root=Path(args.output or tempfile.mkdtemp(prefix='swclib-parallel-'));root.mkdir(parents=True,exist_ok=True)
    result=benchmark(root,args.files,args.nodes,args.repeats)
    report=json.dumps(result,indent=2,ensure_ascii=False);(root/'results.json').write_text(report);print(report,flush=True)


if __name__=='__main__':main()
