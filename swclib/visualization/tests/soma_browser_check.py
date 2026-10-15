"""Real-browser acceptance of the soma plugin; fixtures stay under --output."""
import argparse
import hashlib
import json
from pathlib import Path
import threading

import numpy as np
import nrrd
import tifffile
from playwright.sync_api import sync_playwright


def fixtures(root):
    brain=root/'brain01';brain.mkdir(parents=True,exist_ok=True)
    z,y,x=np.indices((32,64,80));volume=(x+2*y+4*z).astype('uint16')
    for index,plane in enumerate(volume):tifffile.imwrite(brain/f'z_{index}.tif',plane)
    neurons=root/'neurons';neurons.mkdir(exist_ok=True)
    (neurons/'typed.swc').write_text('10 1 30 30 15 1 -1\n11 3 31 30 15 1 10\n12 1 55 40 20 1 -1\n13 3 56 40 20 1 12\n')
    (neurons/'roots.swc').write_text('1 3 1 2 1 1 -1\n2 3 2 2 1 1 1\n3 3 75 60 30 1 -1\n')
    (neurons/'outside.swc').write_text('1 1 -1 0 0 1 -1\n')
    (neurons/'empty.swc').write_text('1 3 2 3 4 1 99\n')
    (neurons/'broken.swc').write_text('1 1 invalid 2 3 1 -1\n')
    (neurons/'many.swc').write_text(''.join(f'{i+1} 1 {20+i%20} 20 15 1 -1\n' for i in range(61)))
    nested=neurons/'nested';nested.mkdir(exist_ok=True)
    (nested/'one.swc').write_text('1 1 30 30 15 1 -1\n')
    fnt=root/'fnt';fnt.mkdir(exist_ok=True)
    (fnt/'catalog').write_text('[CH00]\npattern=z<6Z>y<6Y>x<6X>.nrrd\nsize=80 64 32\ncubesize=80 64 32\n')
    nrrd.write(str(fnt/'z000000y000000x000000.nrrd'),volume,index_order='C')


def check(page,root):
    errors=[];volumes=[];active=set();maximum=[0]
    page.on('pageerror',lambda error:errors.append(str(error)))
    def request(req):
        if req.url.endswith('/api/volume'):volumes.append(req)
        if req.url.endswith('/api/somas'):active.add(req);maximum[0]=max(maximum[0],len(active))
    page.on('request',request)
    page.on('requestfinished',lambda req:active.discard(req));page.on('requestfailed',lambda req:active.discard(req))
    page.wait_for_function('!!window.swcStudio')
    tabs=page.locator('.tabs button');assert tabs.all_text_contents()==['神经元 SWC','图像 VOLUME','插件 PLUGINS']
    page.locator('#tab-plugins').click();assert page.locator('#panel-swc').is_hidden() and page.locator('#panel-image').is_hidden()
    page.locator('#plugin-soma summary').click();assert page.locator('#soma-form').is_hidden()
    page.locator('#plugin-soma summary').click()
    page.locator('#soma-brain-path').fill(str(root/'brain01'));page.locator('#soma-brain-mode').select_option('slice')
    page.locator('#soma-length').fill('12');page.locator('#soma-swc-path').fill(str(root/'neurons'))
    page.locator('#soma-scan').click()
    page.wait_for_function('document.getElementById("soma-progress").textContent.includes("已识别 6 / 6") && !document.getElementById("soma-scan").disabled')
    assert page.locator('#soma-count').text_content()=='66'
    assert page.locator('#soma-results .soma-candidate').count()==50
    assert not volumes,'scan decoded image pixels'
    assert page.evaluate('swcStudio.renderer.count')==0
    assert page.evaluate('swcStudio.entries.filter(e=>e.kind==="swc").every(e=>!e.visible && e.count===undefined)')
    assert maximum[0]==2,maximum
    page.locator('#soma-next').click();assert page.locator('#soma-results .soma-candidate').count()==18
    page.locator('#soma-search').fill('outside');assert page.locator('#soma-results button').is_disabled()
    for name,hint in [('empty','没有 type=1'),('broken','读取失败')]:
        page.locator('#soma-search').fill(name);assert hint in page.locator('#soma-results').text_content()

    def candidates(name):
        page.locator('#soma-search').fill(name);return page.locator('#soma-results button')
    def state():
        return page.evaluate('({origin:swcStudio.renderer.volume?.origin,center:swcStudio.renderer.center,crop:swcStudio.renderer.cropStart,count:swcStudio.renderer.count,revision:swcStudio.revision})')
    candidates('typed').nth(0).click()
    page.wait_for_function('swcStudio.renderer.volume?.origin[0]===24 && swcStudio.renderer.count===2')
    assert state()['center']==[6,6,6] and state()['crop']==[24,24,9]
    assert page.evaluate('swcStudio.renderer.volume.shape')==[12,12,12]
    assert page.evaluate('swcStudio.entries.filter(e=>e.kind==="swc"&&e.visible).map(e=>e.name)')==['typed.swc']
    candidates('roots').nth(0).click()
    page.wait_for_function('swcStudio.renderer.volume?.origin[0]===0 && swcStudio.renderer.count===2')
    assert state()['center']==[1,2,1]
    candidates('roots').nth(1).click()
    page.wait_for_function('swcStudio.renderer.volume?.origin[0]===68 && swcStudio.renderer.count===1')
    assert state()['crop']==[68,52,20] and state()['center']==[7,8,10]

    # Stage A behind an intercepted response, select B, then release A.
    held=[]
    def hold(route):
        if not held:held.append(route)
        else:route.continue_()
    page.route('**/api/volume',hold)
    previous=state();candidates('typed').nth(0).click()
    for _ in range(100):
        if held:break
        page.wait_for_timeout(20)
    assert len(held)==1 and state()==previous
    old=held[0].fetch();candidates('typed').nth(1).click();held[0].fulfill(response=old)
    page.wait_for_function('swcStudio.renderer.volume?.origin[0]===49 && swcStudio.renderer.count===2')
    page.wait_for_function('document.getElementById("soma-view-status").textContent.includes("已显示")')
    page.unroute('**/api/volume');assert state()['crop']==[49,34,14]
    assert page.locator('#soma-results [aria-pressed=true]').count()==1

    # A failed volume read preserves both halves of the previous scene.
    page.route('**/api/volume',lambda route:route.fulfill(status=400,content_type='application/json',body='{"error":"test image failure"}'))
    previous=state();candidates('typed').nth(0).click()
    page.wait_for_function('document.getElementById("soma-view-status").textContent.includes("test image failure")')
    assert state()==previous
    page.unroute('**/api/volume')

    # Manual image hiding supersedes a pending plugin selection.
    held=[];page.route('**/api/volume',hold);candidates('typed').nth(0).click()
    for _ in range(100):
        if held:break
        page.wait_for_timeout(20)
    assert held
    old=held[0].fetch();page.locator('#tab-image').click();page.locator('#hide-image').click();held[0].fulfill(response=old)
    page.locator('#tab-plugins').click()
    page.wait_for_function('document.getElementById("soma-view-status").textContent.includes("取消")')
    assert page.evaluate('swcStudio.renderer.volume===null');page.unroute('**/api/volume')
    candidates('typed').nth(1).click()
    page.wait_for_function('swcStudio.renderer.volume?.origin[0]===49')
    assert page.evaluate('swcStudio.renderer.gl.getError()')==0
    page.screenshot(path=str(root/'soma-plugin.png'))
    page.set_viewport_size({'width':390,'height':844})
    assert page.locator('#tab-plugins').is_visible()
    assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
    page.screenshot(path=str(root/'soma-mobile.png'))

    # Cancel an in-flight scan: delayed candidates cannot repopulate the list.
    page.set_viewport_size({'width':1280,'height':1000})
    held=[];page.route('**/api/somas',hold)
    page.locator('#soma-scan').click()
    for _ in range(100):
        if held:break
        page.wait_for_timeout(20)
    assert held
    old=held[0].fetch();page.locator('#soma-cancel').click();held[0].fulfill(response=old)
    page.wait_for_function('document.getElementById("soma-count").textContent==="0"')
    page.wait_for_timeout(150);assert page.locator('#soma-count').text_content()=='0'
    page.unroute('**/api/somas')
    page.locator('#soma-recursive').check();page.locator('#soma-scan').click()
    page.wait_for_function('document.getElementById("soma-progress").textContent.includes("已识别 7 / 7") && !document.getElementById("soma-scan").disabled')
    assert page.locator('#soma-count').text_content()=='67'
    # A single-file selection also works with the FNT backend.
    page.locator('#soma-brain-path').fill(str(root/'fnt'));page.locator('#soma-brain-mode').select_option('fnt')
    page.locator('#soma-swc-path').fill(str(root/'neurons/typed.swc'));page.locator('#soma-scan').click()
    page.wait_for_function('document.getElementById("soma-count").textContent==="2" && !document.getElementById("soma-scan").disabled')
    candidates('typed').nth(0).click()
    page.wait_for_function('swcStudio.renderer.volume?.origin[0]===24 && swcStudio.renderer.count===2')
    assert page.evaluate('swcStudio.renderer.volume.shape')==[12,12,12]
    assert page.evaluate('swcStudio.entries.some(e=>e.kind==="brain-region"&&e.name.startsWith("fnt_"))')
    assert not errors,errors
    return dict(candidate_count=66,maximum_parallel_scans=maximum[0],scan_volume_requests=0,
                checks='tabs, collapse, paging, filtering, type/root rules, invalid inputs, aligned cubes at both boundaries, latest selection, failed-read preservation, manual cancellation, scan cancellation, recursive directory, single SWC, TIFF and FNT, mobile layout')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--chromium',required=True);args=parser.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True);fixtures(root)
    snapshots={p:(hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_mtime_ns,p.stat().st_mode) for p in root.rglob('*') if p.suffix in ('.swc','.tif','.nrrd') or p.name=='catalog'}
    from swclib.visualization.server import create_server
    server=create_server(port=0,swc_workers=2);threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        with sync_playwright() as playwright:
            browser=playwright.chromium.launch(executable_path=args.chromium,args=['--no-sandbox','--enable-unsafe-swiftshader','--use-angle=swiftshader'])
            page=browser.new_page(viewport={'width':1280,'height':1000},bypass_csp=True)
            page.goto(f'http://127.0.0.1:{server.server_port}');result=check(page,root)
            assert all(snapshot==(hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_mtime_ns,p.stat().st_mode) for p,snapshot in snapshots.items())
            assert not list(root.rglob('*.aux.xml'))
            result['source_integrity']='all hashes, mtimes and modes unchanged'
            (root/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
            browser.close()
    finally:server.shutdown();server.server_close()


if __name__=='__main__':main()
