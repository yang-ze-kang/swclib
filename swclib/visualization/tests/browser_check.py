"""Optional real-browser acceptance/benchmark. Requires playwright + Chromium.

Run with --url against a running viewer, or omit it to start a temporary server.
Fixtures and screenshots are written only below --output (default: a temp dir).
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time

import numpy as np
import tifffile
from playwright.sync_api import sync_playwright


def fixtures(root):
    (root/'target.swc').write_text('1 1 10 10 5 1 -1\n2 3 90 10 5 1 1\n3 3 50 75 25 1 2\n')
    target=np.zeros((10,12,14),dtype='uint16');target[3,6,8]=1000;np.save(root/'target.npy',target)
    maximum=np.zeros((300,300,300),dtype='u1');maximum[150,150,150]=255
    np.save(root/'maximum300.npy',maximum);del maximum
    race=root/'race';race.mkdir(exist_ok=True)
    for i in range(16):
        (race/f'race_{i:02}.swc').write_text(''.join(f'{j+1} 3 {j} 0 0 1 {j if j else -1}\n' for j in range(64)))
    (race/'broken.swc').write_text('1 1 invalid 0 0 1 -1\n')
    for amount, label in [(10000, '10k'), (100000, '100k')]:
        target=root/label;target.mkdir(exist_ok=True)
        for neuron in range(10):
            n=amount//10
            with (target/f'neuron_{neuron:02}.swc').open('w') as out:
                for i in range(n):
                    t=i/n*35
                    x=60*np.sin(t)+neuron*22;y=60*np.cos(t)+neuron*8;z=i/n*180
                    out.write(f'{i+1} 3 {x:.5f} {y:.5f} {z:.5f} 1 {i if i else -1}\n')
    images=root/'images';images.mkdir(exist_ok=True)
    z,y,x=np.indices((32,64,80))
    volume=(np.exp(-((x-38)**2+(y-30)**2)/70)*3500+np.exp(-((x-20)**2+(y-45)**2+(z-15)**2)/50)*10000).astype('uint16')
    tifffile.imwrite(images/'signal.tif',volume,photometric='minisblack')
    tifffile.imwrite(images/'other.tif',volume//2,photometric='minisblack')
    (root/'overlay.swc').write_text('\n'.join(f'{i+1} 3 38 30 {i} 1 {i if i else -1}' for i in range(32)))
    brain=root/'slices';brain.mkdir(exist_ok=True)
    for i,plane in enumerate(volume):tifffile.imwrite(brain/f'slice_{i}.tif',plane)


def check(page, root):
    errors=[];requests=[];inflight=set();maximum=[0]
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.on('request',lambda request:requests.append(request.url))
    def begin(request):
        if request.url.endswith('/api/swc'):
            inflight.add(request);maximum[0]=max(maximum[0],len(inflight))
    page.on('request',begin)
    page.on('requestfinished',lambda request:inflight.discard(request))
    page.on('requestfailed',lambda request:inflight.discard(request))
    page.wait_for_function('!!window.swcStudio')
    workers=page.request.get(page.url.rstrip('/')+'/api/config').json()['swc_workers']
    results={'viewport':[1280,900], 'browser':page.context.browser.version}
    results['renderer']=page.evaluate('''() => {const gl=swcStudio.renderer.gl,e=gl.getExtension('WEBGL_debug_renderer_info');return e?gl.getParameter(e.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER);}''')
    for size in ['10k','100k']:
        page.locator('#hide-all').click()
        page.locator('#swc-path').fill(str(root/size))
        start=time.perf_counter();page.locator('#swc-form button').click()
        page.wait_for_function('(n)=>swcStudio.entries.filter(e=>e.kind==="swc"&&e.visible).reduce((n,e)=>n+(e.count||0),0)===n',arg=int(size[:-1])*1000)
        page.wait_for_function('swcStudio.renderer.count>=10000 && swcStudio.renderer.drawCalls===2')
        page.wait_for_timeout(150)
        results[size]={'load_seconds':round(time.perf_counter()-start,3)}
        results[size].update(page.evaluate('''async () => {
            const r=swcStudio.renderer,times=[],frames=[];let previous=performance.now();
            // This benchmark explicitly renders each frame; don't also queue
            // a second render through the interactive rotation invalidation.
            const invalidate=r.invalidate;r.invalidate=()=>{};
            for(let i=0;i<65;i++) {
                await new Promise(resolve=>requestAnimationFrame(resolve));
                const now=performance.now();if(i>4)frames.push(now-previous);previous=now;
                r.rotate(-.016/.006,0);const start=performance.now();r.render();r.gl.finish();if(i>4)times.push(performance.now()-start);
            }
            r.invalidate=invalidate;r.invalidate();
            times.sort((a,b)=>a-b);frames.sort((a,b)=>a-b);
            return {main_thread_submission_median_ms:times[30],main_thread_submission_p95_ms:times[57],interaction_fps:1000/(frames.reduce((a,b)=>a+b,0)/frames.length),draw_calls:r.drawCalls,nodes_in_gpu:r.count};
        }'''))
        print(size, json.dumps(results[size]), flush=True)
        page.locator('#color-mode').select_option('1')
        page.wait_for_function('swcStudio.renderer.settings.colorMode===1')
        assert page.locator('#legend').is_hidden()
        assert page.evaluate('new Set(swcStudio.entries.filter(e=>e.kind==="swc").map(e=>e.color.join(","))).size')==len(page.locator('#swc-list .visibility').all())
        page.screenshot(path=str(root/f'{size}.png'))
        page.locator('#color-mode').select_option('0')
    results['maximum_parallel_swc_requests']=maximum[0]
    assert min(2,workers)<=maximum[0]<=workers, 'SWC requests must overlap with a bounded queue'

    # Real trackball and roll drags change the basis without moving the center.
    original=page.evaluate('({rotation:[...swcStudio.renderer.rotation],center:[...swcStudio.renderer.center],span:swcStudio.renderer.span})')
    axes=page.locator('#orientation-axes');assert axes.is_visible()
    before=axes.evaluate('(c)=>c.toDataURL()')
    page.mouse.move(300,190);page.mouse.down();page.mouse.move(300,690,steps=12);page.mouse.up()
    page.wait_for_timeout(100)
    assert page.evaluate('swcStudio.renderer.rotation')!=original['rotation']
    assert axes.evaluate('(c)=>c.toDataURL()')!=before
    assert page.evaluate('swcStudio.renderer.center')==original['center']
    assert page.evaluate('swcStudio.renderer.span')==original['span']
    forward=page.evaluate('swcStudio.renderer.basis()[2]')
    right=page.evaluate('swcStudio.renderer.basis()[0]')
    page.keyboard.down('Alt');page.mouse.move(300,300);page.mouse.down();page.mouse.move(500,300,steps=12);page.mouse.up();page.keyboard.up('Alt')
    assert all(abs(a-b)<1e-12 for a,b in zip(page.evaluate('swcStudio.renderer.basis()[2]'),forward))
    assert page.evaluate('swcStudio.renderer.basis()[0]')!=right
    assert page.evaluate('''() => {
        const r=swcStudio.renderer,before=r.basis().flat();
        r.rotate(2*Math.PI/.006,0);r.rotate(0,2*Math.PI/.006);r.rotate(0,0,2*Math.PI/.006);
        return r.basis().flat().every((v,i)=>Math.abs(v-before[i])<1e-12);
    }''')
    assert page.evaluate('''() => {
        const r=swcStudio.renderer,previous=[...r.rotation];
        for(let k=0;k<1000;k++) {
            r.rotate(Math.sin(k)*8,Math.cos(k)*10,3);const b=r.basis();
            for(let i=0;i<3;i++)for(let j=0;j<3;j++)
                if(Math.abs(b[i].reduce((s,v,k)=>s+v*b[j][k],0)-(i===j?1:0))>1e-12)return false;
        }
        r.rotation=previous;return true;
    }''')
    page.screenshot(path=str(root/'360-axes.png'))
    page.evaluate('(v)=>{swcStudio.renderer.rotation=v.rotation;swcStudio.renderer.invalidate();}',original)
    page.locator('#hide-all').click()
    page.locator('#swc-path').fill(str(root/'overlay.swc'));page.locator('#swc-form button').click()
    page.wait_for_function('swcStudio.entries.some(e=>e.name==="overlay.swc"&&e.count===32)')
    page.locator('#tab-image').click();page.locator('#image-path').fill(str(root/'images'));page.locator('#image-form button').click()
    page.wait_for_function('swcStudio.entries.filter(e=>e.kind==="image").length===2')
    assert not any('/api/volume' in url for url in requests), 'directory registration decoded hidden images'
    page.locator('#image-list .file-row').filter(has_text='signal.tif').click()
    page.wait_for_function('!!swcStudio.renderer.volume')
    assert page.evaluate('swcStudio.renderer.volume.shape')==[80,64,32]
    for view in ['xy','xz','yz','volume','mip']:
        page.locator('#render-mode').select_option(view)
        page.wait_for_timeout(100)
        assert page.evaluate('swcStudio.renderer.gl.getError()')==0
    page.locator('#render-mode').select_option('xy')
    page.locator('#slice-index').fill('15');page.locator('#slice-index').dispatch_event('input')
    page.locator('#brightness').fill('0.2');page.locator('#brightness').dispatch_event('input')
    assert page.evaluate('swcStudio.renderer.settings.brightness')==0.2
    page.locator('#reset-window').click();page.screenshot(path=str(root/'overlay.png'))
    page.locator('#image-list .file-row').filter(has_text='other.tif').click()
    page.wait_for_function('swcStudio.renderer.volume?.intensity_range[1]<6000')
    assert page.locator('#image-list input:checked').count()==1
    page.locator('#hide-image').click();assert page.evaluate('swcStudio.renderer.volume===null')
    page.locator('#tab-swc').click()
    page.locator('#crop-start').fill('37, 29, 10');page.locator('#crop-end').fill('39, 31, 20');page.locator('#crop-form button[type=submit]').click()
    page.wait_for_function('swcStudio.renderer.count===10')
    assert page.evaluate('swcStudio.entries.find(e=>e.name==="overlay.swc").bounds')==[[1,1,0],[1,1,9]]
    page.locator('#crop-start').fill('500, 500, 500');page.locator('#crop-end').fill('510, 510, 510');page.locator('#crop-form button[type=submit]').click()
    page.wait_for_function('swcStudio.entries.find(e=>e.name==="overlay.swc").count===0')
    page.locator('#clear-crop').click();page.wait_for_function('swcStudio.renderer.count===32')
    page.locator('#tab-image').click();page.locator('#brain-details summary').click();page.locator('#brain-path').fill(str(root/'slices'))
    page.locator('#brain-mode').select_option('slice');page.locator('#brain-pattern').fill(r'slice_(\d+)\.tif');page.locator('#brain-form button').click()
    page.wait_for_function('swcStudio.entries.some(e=>e.kind==="brain")')
    page.wait_for_function('document.getElementById("brain-info").textContent.includes("80 × 64 × 32")')
    assert page.locator('#image-list .file-row').count()==2,'whole-brain source must not appear as an image'
    page.locator('#brain-start').fill('30, 20, 5');page.locator('#brain-end').fill('50, 40, 25');page.locator('#load-region').click()
    page.wait_for_function('!!swcStudio.renderer.volume')
    assert page.evaluate('swcStudio.renderer.volume.origin')==[30,20,5]
    assert page.evaluate('swcStudio.renderer.volume.shape')==[20,20,20]
    first='slices_[30,20,5]-[50,40,25]'
    assert page.locator('#image-list .file-row').filter(has_text=first).count()==1
    page.locator('#brain-start').fill('10, 10, 0');page.locator('#brain-end').fill('30, 30, 10');page.locator('#load-region').click()
    page.wait_for_function('swcStudio.renderer.volume?.origin[0]===10')
    second='slices_[10,10,0]-[30,30,10]'
    page.locator('#load-region').click()
    page.wait_for_function('!document.getElementById("load-region").disabled')
    assert page.locator('#image-list .file-row').filter(has_text=second).count()==1
    page.locator('#image-list .file-row').filter(has_text=first).locator('.file-display').click()
    page.wait_for_function('swcStudio.renderer.volume?.origin[0]===30')
    page.locator('#brain-end').fill('5000, 5000, 5000');page.locator('#load-region').click()
    page.wait_for_function('document.getElementById("toast").textContent.includes("超过当前显示上限")')
    assert page.evaluate('swcStudio.renderer.volume.origin')==[30,20,5]
    assert page.locator('#image-list .file-row').count()==4
    page.screenshot(path=str(root/'brain-regions.png'))
    page.locator('#remove-brain').click();page.wait_for_function('swcStudio.entries.every(e=>e.kind!=="brain")')
    page.locator('#image-list .file-row').filter(has_text=second).locator('.file-display').click()
    page.wait_for_function('swcStudio.renderer.volume?.origin[0]===10')
    # Explicit center preserves zoom and uses displayed XYZ coordinates.
    page.locator('.camera-controls summary').click()
    span=page.evaluate('swcStudio.renderer.span')
    page.locator('#view-center').fill('12.5, 23, 34');page.locator('#center-form button[type=submit]').click()
    assert page.evaluate('swcStudio.renderer.center')==[12.5,23,34]
    assert page.evaluate('swcStudio.renderer.span')==span
    page.locator('#read-center').click();assert page.locator('#view-center').input_value()=='12.5, 23, 34'
    page.locator('.camera-controls summary').click()

    # Removing a displayed SWC frees its worker/GPU data; surviving slots stay stable.
    page.locator('#tab-swc').click()
    page.locator('#swc-list .file-row').filter(has_text='overlay.swc').locator('.remove-entry').click()
    page.wait_for_function('!swcStudio.entries.some(e=>e.name==="overlay.swc") && swcStudio.renderer.count===0')
    page.locator('#select-all-swc').check();page.locator('#remove-selected-swc').click()
    page.wait_for_function('swcStudio.entries.every(e=>e.kind!=="swc")')
    page.locator('#swc-path').fill(str(root/'overlay.swc'));page.locator('#swc-form button').click()
    page.wait_for_function('swcStudio.renderer.count===32')
    assert page.evaluate('swcStudio.renderer.gl.getError()')==0

    # Remove an active image and a mixed batch of image/whole-brain registrations.
    page.locator('#tab-image').click()
    page.locator('#image-list .file-row').filter(has_text='signal.tif').locator('.file-display').click()
    page.wait_for_function('!!swcStudio.renderer.volume')
    page.locator('#image-list .file-row').filter(has_text='signal.tif').locator('.remove-entry').click()
    page.wait_for_function('!swcStudio.renderer.volume && !swcStudio.entries.some(e=>e.name==="signal.tif")')
    page.locator('#select-all-image').check();page.locator('#remove-selected-image').click()
    page.wait_for_function('swcStudio.entries.every(e=>e.kind==="swc")')

    # Hold successful old responses, then crop/remove while the queue is busy.
    # Late full-size responses must never resurrect a removed or uncropped SWC.
    page.locator('#tab-swc').click()
    held=[];held_limit=min(4,workers)
    def hold_old(route):
        if len(held)<held_limit:held.append(route)
        else:route.continue_()
    page.route('**/api/swc',hold_old)
    page.locator('#swc-path').fill(str(root/'race'));page.locator('#swc-form button').click()
    for _ in range(100):
        if len(held)==held_limit:break
        page.wait_for_timeout(20)
    assert len(held)==held_limit
    old=[(route,route.fetch()) for route in held]
    removed=next((route.request.post_data_json['id'] for route,response in old if response.status==200),None)
    if removed is None:removed=page.evaluate('swcStudio.entries.find(e=>e.name.startsWith("race_")).id')
    name=page.evaluate('(id)=>swcStudio.entries.find(e=>e.id===id).name',removed)
    page.locator('#swc-list .file-row').filter(has_text=name).locator('.remove-entry').click()
    page.wait_for_function('(id)=>!swcStudio.entries.some(e=>e.id===id)',arg=removed)
    page.locator('#crop-start').fill('10, -1, -1');page.locator('#crop-end').fill('20, 1, 1')
    page.locator('#crop-form button[type=submit]').click()
    for route,response in old:route.fulfill(response=response)
    # Keep the pass-through handler until the new requests finish. Removing it
    # while continue_() is active can double-handle a route in Playwright.
    try:
        page.wait_for_function('swcStudio.renderer.count===150 && swcStudio.entries.filter(e=>e.name.startsWith("race_")).every(e=>e.loadedRevision===swcStudio.revision)',timeout=30000)
    except Exception:
        print('Delayed-response state:',page.evaluate('''()=>({
            revision:swcStudio.revision,count:swcStudio.renderer.count,
            status:document.getElementById('swc-load-status').textContent,
            message:document.getElementById('status-text').textContent,
            entries:swcStudio.entries.map(({name,count,visible,loading,loadedRevision,loadingRevision})=>
                ({name,count,visible,loading,loadedRevision,loadingRevision}))
        })'''),flush=True)
        raise
    assert not page.evaluate('(id)=>swcStudio.entries.some(e=>e.id===id)',removed)
    assert page.evaluate('swcStudio.entries.find(e=>e.name==="broken.swc").visible') is False
    assert page.evaluate('swcStudio.entries.filter(e=>e.name.startsWith("race_")).every(e=>e.bounds[0][0]===0 && e.bounds[1][0]===9)')
    page.unroute('**/api/swc')
    page.locator('#select-all-swc').check();page.locator('#remove-selected-swc').click()
    page.wait_for_function('swcStudio.entries.length===0 && swcStudio.renderer.count===0')
    page.locator('#clear-crop').click()
    page.locator('#tab-image').click()

    # Full 300^3 must load, rather than still hitting the old 64 MiB browser cap.
    page.locator('#image-path').fill(str(root/'maximum300.npy'));page.locator('#image-form button').click()
    page.wait_for_function('swcStudio.entries.some(e=>e.name==="maximum300.npy")')
    page.locator('#render-mode').select_option('xy')
    page.locator('#image-list .file-row').filter(has_text='maximum300.npy').locator('.file-display').click()
    page.wait_for_function('swcStudio.renderer.volume?.shape.every(n=>n===300)',timeout=60000)
    error=page.evaluate('swcStudio.renderer.gl.getError()');assert error==0,error
    page.screenshot(path=str(root/'v040.png'))
    page.locator('#select-all-image').check();page.locator('#remove-selected-image').click()
    page.wait_for_function('swcStudio.renderer.volume===null')
    check_targets(page,root)
    assert not errors,errors
    results['checks']='independent named whole-brain ROIs, deduplication and lazy reload, SWC node/edge/3D/cropped targeting, slice/MIP/volume targeting, empty-space rejection, right-drag pan, overall center, free trackball and roll rotation, live XYZ axes, bounded concurrent SWC requests, concurrent crop/removal/error/late-response handling, full 300^3, single/batch source removal, read-only service'
    return results


def check_targets(page,root):
    def projection(point):
        return page.evaluate('''(p)=>{
            const r=swcStudio.renderer,rect=r.canvas.getBoundingClientRect(),[right,up]=r.basis(),q=p.map((v,i)=>v-r.center[i]);
            return [rect.left+rect.width*(.5+q.reduce((s,v,i)=>s+v*right[i],0)/(2*r.span*r.canvas.width/r.canvas.height)),
                    rect.top+rect.height*(.5-q.reduce((s,v,i)=>s+v*up[i],0)/(2*r.span))];
        }''',point)
    def target(point,tolerance=.25):
        before=page.evaluate('({span:swcStudio.renderer.span,rotation:swcStudio.renderer.rotation})')
        page.mouse.click(*projection(point),button='right')
        page.wait_for_function('!document.getElementById("target-menu").hidden && !document.getElementById("go-to-target").disabled')
        page.locator('#go-to-target').click()
        center=page.evaluate('swcStudio.renderer.center')
        assert all(abs(a-b)<=tolerance for a,b in zip(center,point)),(center,point)
        assert page.evaluate('({span:swcStudio.renderer.span,rotation:swcStudio.renderer.rotation})')==before
        assert page.locator('#target-menu').is_hidden()

    page.locator('#tab-swc').click();page.locator('#swc-path').fill(str(root/'target.swc'));page.locator('#swc-form button').click()
    page.wait_for_function('swcStudio.renderer.count===3')
    page.locator('[data-view="xy"]').click()
    target([90,10,5])
    span=page.evaluate('swcStudio.renderer.span')
    page.locator('#overall-center').click()
    assert page.evaluate('swcStudio.renderer.center')==[50,42.5,15]
    assert page.evaluate('swcStudio.renderer.span')==span
    page.locator('#show-points').uncheck();target([50,10,5]);page.locator('#show-points').check()
    page.locator('#overall-center').click()
    # Right dragging retains panning and must not open the targeting menu.
    page.mouse.move(300,300);page.mouse.down(button='right');page.mouse.move(350,320,steps=5);page.mouse.up(button='right')
    assert page.locator('#target-menu').is_hidden()
    assert page.evaluate('swcStudio.renderer.center')!=[50,42.5,15]
    page.locator('#overall-center').click()
    page.locator('#crop-start').fill('10, 0, 0');page.locator('#crop-end').fill('100, 100, 30');page.locator('#crop-form button[type=submit]').click()
    page.wait_for_function('swcStudio.entries.find(e=>e.name==="target.swc").bounds[0][0]===0 && swcStudio.renderer.count===3')
    target([80,10,5]);page.locator('#overall-center').click()
    assert page.evaluate('swcStudio.renderer.center')==[40,42.5,15]
    page.locator('[data-view="mip"]').click()
    page.evaluate('()=>{const r=swcStudio.renderer;r.rotation=[0,0,0,1];r.rotate(-.6/.006,0);r.rotate(0,2.1/.006);r.rotate(0,0,40);}')
    page.wait_for_timeout(50);target([40,75,25])
    page.locator('#hide-all').click()
    page.mouse.click(300,300,button='right')
    page.wait_for_function('document.getElementById("target-info").textContent.includes("未命中")')
    assert page.locator('#go-to-target').is_disabled()
    page.keyboard.press('Escape');assert page.locator('#target-menu').is_hidden()
    page.locator('#clear-crop').click()
    page.locator('#tab-image').click();page.locator('#image-path').fill(str(root/'target.npy'));page.locator('#image-form button').click()
    page.wait_for_function('swcStudio.entries.some(e=>e.name==="target.npy")')
    page.locator('#render-mode').select_option('xy')
    page.locator('#image-list .file-row').filter(has_text='target.npy').locator('.file-display').click()
    page.wait_for_function('swcStudio.renderer.volume?.shape[0]===14')
    page.locator('#slice-index').fill('3');page.locator('#slice-index').dispatch_event('input');target([8,6,3],.1)
    for mode in ['mip','volume']:
        page.locator('#render-mode').select_option(mode)
        page.evaluate('()=>{const r=swcStudio.renderer;r.rotation=[0,0,0,1];r.invalidate();}')
        page.locator('#overall-center').click();target([8,6,3],.8)
        assert page.evaluate('swcStudio.renderer.gl.getError()')==0
    page.locator('#render-mode').select_option('xy');page.locator('#slice-index').fill('3');page.locator('#slice-index').dispatch_event('input')
    page.locator('#overall-center').click()
    # Overall center also resets the orthogonal plane to the image center.
    assert page.evaluate('swcStudio.renderer.center')==[6.5,5.5,4.5]
    page.locator('#slice-index').fill('3');page.locator('#slice-index').dispatch_event('input')
    page.mouse.click(*projection([2,2,3]),button='right')
    page.wait_for_function('document.getElementById("target-info").textContent.includes("未命中")')
    assert page.locator('#go-to-target').is_disabled()
    page.keyboard.press('Escape');page.mouse.click(*projection([8,6,3]),button='right')
    page.wait_for_function('!document.getElementById("go-to-target").disabled')
    page.screenshot(path=str(root/'go-to-target.png'))
    # A subsequent view change invalidates a previously picked location.
    page.locator('[data-view="xz"]').click();assert page.locator('#target-menu').is_hidden()
    assert page.evaluate('swcStudio.renderer.gl.getError()')==0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--url');parser.add_argument('--chromium');parser.add_argument('--output')
    parser.add_argument('--hardware',action='store_true',help='Allow native GPU instead of forcing SwiftShader')
    args=parser.parse_args();root=Path(args.output or tempfile.mkdtemp(prefix='swclib-viewer-qa-'));root.mkdir(parents=True,exist_ok=True);fixtures(root)
    server=None
    if not args.url:
        from swclib.visualization.server import create_server
        server=create_server(port=0);threading.Thread(target=server.serve_forever,daemon=True).start();args.url=f'http://127.0.0.1:{server.server_port}'
    try:
        with sync_playwright() as p:
            flags=['--no-sandbox'] if args.hardware else ['--no-sandbox','--enable-unsafe-swiftshader','--use-angle=swiftshader']
            browser=p.chromium.launch(executable_path=args.chromium,args=flags)
            # Permit Playwright's injected polling helpers; production CSP stays strict.
            page=browser.new_page(viewport={'width':1280,'height':900},bypass_csp=True)
            sources={p:(hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_mtime_ns,p.stat().st_mode)
                     for p in root.rglob('*') if p.suffix in ('.swc','.tif','.npy')}
            page.goto(args.url);result=check(page,root)
            assert all(p.exists() and snapshot==(hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_mtime_ns,p.stat().st_mode)
                       for p,snapshot in sources.items()),'source data changed'
            assert not list(root.rglob('*.aux.xml')),'source sidecar was written'
            result['source_integrity']='all source hashes, mtimes and modes unchanged'
            (root/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False));print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)
            browser.close()
    finally:
        if server:server.shutdown();server.server_close()


if __name__=='__main__':main()
