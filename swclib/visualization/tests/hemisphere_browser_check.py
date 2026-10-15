"""Plane manipulation, crop invariance, export schema and source integrity."""
import argparse
import hashlib
import json
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright


def fixtures(root):
    neurons=root/'neurons';neurons.mkdir(parents=True,exist_ok=True)
    (neurons/'cross.swc').write_text('1 1 100 100 100 1 -1\n2 3 120 100 100 1 1\n3 3 130 100 100 1 2\n4 3 130 120 100 1 3\n')
    (neurons/'left.swc').write_text('1 1 101 120 110 1 -1\n2 3 105 120 110 1 1\n')
    (root/'ambiguous.swc').write_text('1 1 105 100 100 1 -1\n2 1 106 100 100 1 1\n')
    for name in ['output','ambiguous-output']:(root/name).mkdir(exist_ok=True)


def check(page,root):
    errors=[];reads=[]
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.on('request',lambda request:reads.append(request.url))
    page.wait_for_function('!!window.swcStudio')
    page.locator('#swc-path').fill(str(root/'neurons'));page.locator('#swc-form button').click()
    page.wait_for_function('swcStudio.renderer.count===6 && swcStudio.renderer.drawCalls===2')
    page.locator('#tab-plugins').click();page.locator('#plugin-soma summary').click()
    assert page.locator('#plugin-hemisphere summary').text_content()=='划分脑区'
    page.locator('#hemisphere-visible').check()
    page.wait_for_function('swcStudio.renderer.drawCalls===3')
    assert float(page.locator('#hemisphere-offset').input_value())==115
    assert page.locator('#hemisphere-left').is_visible() and page.locator('#hemisphere-right').is_visible()
    start=len(reads)
    for name,value,key in [('azimuth','124','azimuth_deg'),('elevation','6','elevation_deg'),('offset','4177.036228665835','offset')]:
        page.locator('#hemisphere-'+name).fill(value)
        assert abs(page.evaluate('(key)=>swcStudio.renderer.hemispherePlane[key]',key)-float(value))<1e-9
        assert abs(float(page.locator('#hemisphere-'+name+'-range').input_value())-float(value))<1e-9
    for name,value,key in [('azimuth','35','azimuth_deg'),('elevation','-20','elevation_deg'),('offset','115','offset')]:
        page.locator('#hemisphere-'+name+'-range').fill(value);page.locator('#hemisphere-'+name+'-range').dispatch_event('input')
        assert float(page.locator('#hemisphere-'+name).input_value())==float(value)
        assert page.evaluate('(key)=>swcStudio.renderer.hemispherePlane[key]',key)==float(value)
    page.locator('#hemisphere-elevation').fill('91')
    assert page.evaluate('swcStudio.renderer.hemispherePlane.elevation_deg')==-20
    page.locator('#hemisphere-elevation').fill('0');page.locator('#hemisphere-azimuth').fill('0');page.locator('#hemisphere-offset').fill('115')
    assert not any('/api/' in url for url in reads[start:]),'slider triggered a source read'
    page.locator('#hemisphere-count').click()
    page.wait_for_function('!document.getElementById("hemisphere-counts").hidden')
    assert [page.locator('#hemisphere-'+name).inner_text() for name in ['left-only','right-only','cross','total','left-total','right-total']]==['1','0','1','2','2','1']
    assert not list((root/'output').iterdir()),'counting exported a file'
    page.locator('#hemisphere-offset').fill('116')
    assert page.locator('#hemisphere-counts').is_hidden()
    page.locator('#hemisphere-offset').fill('115')
    # A response for an earlier plane must not replace the current result.
    def stale_count(route):
        response=route.fetch()
        page.locator('#hemisphere-offset').fill('116')
        route.fulfill(response=response)
    page.route('**/api/hemisphere/count',stale_count)
    page.locator('#hemisphere-count').click()
    page.wait_for_function('!document.getElementById("hemisphere-count").disabled')
    assert page.locator('#hemisphere-counts').is_hidden()
    assert '重新统计' in page.locator('#hemisphere-count-status').inner_text()
    page.unroute('**/api/hemisphere/count',stale_count)
    page.locator('#hemisphere-offset').fill('115')
    # An exactly edge-on plane stays visible as a line, using the same draw.
    page.locator('[data-view="xy"]').click()
    pixel=page.evaluate('''()=>{const r=swcStudio.renderer,gl=r.gl,p=new Uint8Array(4);r.render();gl.readPixels(Math.floor(r.canvas.width/2),Math.floor(r.canvas.height/2),1,1,gl.RGBA,gl.UNSIGNED_BYTE,p);return [...p];}''')
    assert pixel[0]>50 and page.evaluate('swcStudio.renderer.drawCalls')==3,pixel
    page.locator('[data-view="mip"]').click()
    # Labels follow 3D rotation, and the plane doesn't change the camera center.
    before=page.locator('#hemisphere-left').get_attribute('style')
    center=page.evaluate('swcStudio.renderer.center')
    page.mouse.move(300,300);page.mouse.down();page.mouse.move(440,400,steps=8);page.mouse.up()
    assert page.locator('#hemisphere-left').get_attribute('style')!=before
    assert page.evaluate('swcStudio.renderer.center')==center
    page.screenshot(path=str(root/'plane.png'))
    # Crop changes display coordinates only; global plane and full export stay fixed.
    page.locator('#tab-swc').click()
    page.locator('#swc-list .file-row').filter(has_text='left.swc').locator('.visibility').uncheck()
    page.locator('#crop-start').fill('119, 99, 99');page.locator('#crop-end').fill('131, 121, 111')
    page.locator('#crop-form button[type=submit]').click()
    page.wait_for_function('swcStudio.renderer.count===3')
    assert page.evaluate('swcStudio.renderer.hemispherePlane.offset')==115
    assert page.evaluate('swcStudio.renderer.cropStart')==[119,99,99]
    page.locator('#tab-plugins').click()
    page.locator('#hemisphere-count').click()
    page.wait_for_function('!document.getElementById("hemisphere-counts").hidden')
    assert page.locator('#hemisphere-cross').inner_text()=='1'
    assert page.locator('#hemisphere-total').inner_text()=='1'
    assert page.locator('#hemisphere-right-only').inner_text()=='0','counts used cropped nodes'
    page.screenshot(path=str(root/'counts.png'))
    page.locator('#hemisphere-directory').fill(str(root/'output'));page.locator('#hemisphere-export').click()
    page.wait_for_function('document.getElementById("hemisphere-status").textContent.startsWith("已导出")')
    report=json.loads((root/'output/brain_hemisphere.json').read_text())
    assert set(report)=={'description','plane','neurons'}
    assert report['plane']['normal_xyz']==[1,0,0] and report['plane']['point_xyz']==[115,0,0]
    assert len(report['neurons'])==1
    neuron=report['neurons'][0]
    assert neuron=={'swc_name':'cross.swc','soma':{'x':100,'y':100,'z':100,'side':'left','signed_distance':-15},'left_length':15,'right_length':35,'total_length':50,'hemisphere':'cross'}
    digest=hashlib.sha256((root/'output/brain_hemisphere.json').read_bytes()).hexdigest()
    dialogs=[]
    def dismiss(dialog):
        dialogs.append(dialog.message);dialog.dismiss()
    page.once('dialog',dismiss)
    page.locator('#hemisphere-export').click()
    page.wait_for_function('document.getElementById("hemisphere-status").textContent.includes("已取消导出")')
    assert len(dialogs)==1 and str(root/'output/brain_hemisphere.json') in dialogs[0]
    assert hashlib.sha256((root/'output/brain_hemisphere.json').read_bytes()).hexdigest()==digest
    # Accepting explicitly replaces the selected report with the current plane.
    page.locator('#hemisphere-offset').fill('116')
    def accept(dialog):
        dialogs.append(dialog.message);dialog.accept()
    page.once('dialog',accept)
    page.locator('#hemisphere-export').click()
    page.wait_for_function('document.getElementById("hemisphere-status").textContent.startsWith("已导出")')
    assert len(dialogs)==2
    assert json.loads((root/'output/brain_hemisphere.json').read_text())['plane']['offset']==116
    page.locator('#hemisphere-directory').fill(str(root/'output/custom.json'));page.locator('#hemisphere-export').click()
    page.wait_for_function('document.getElementById("hemisphere-status").textContent.includes("custom.json") && !document.getElementById("hemisphere-export").disabled')
    assert json.loads((root/'output/custom.json').read_text())['plane']['offset']==116
    page.locator('#hemisphere-visible').uncheck()
    page.wait_for_function('swcStudio.renderer.drawCalls===2')
    assert page.locator('#hemisphere-left').is_hidden() and page.locator('#hemisphere-right').is_hidden()
    page.locator('#tab-swc').click();page.locator('#clear-crop').click()
    page.locator('#swc-path').fill(str(root/'ambiguous.swc'));page.locator('#swc-form button').click()
    page.wait_for_function('swcStudio.entries.some(e=>e.name==="ambiguous.swc" && e.count===2)')
    page.locator('#tab-plugins').click();page.locator('#hemisphere-directory').fill(str(root/'ambiguous-output'));page.locator('#hemisphere-export').click()
    page.wait_for_function('document.getElementById("hemisphere-status").textContent.includes("2 个胞体候选")')
    assert not list((root/'ambiguous-output').iterdir())
    page.locator('#hemisphere-count').click()
    page.wait_for_function('!document.getElementById("hemisphere-counts").hidden')
    assert page.locator('#hemisphere-total').inner_text()=='2','counts unnecessarily required a unique soma'
    assert page.evaluate('swcStudio.renderer.gl.getError()')==0
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
    assert not errors,errors
    return dict(checks='plane visibility, moving L/R labels, controls, count classes and totals, stale count response rejection, visible full sources through crop, custom JSON path, overwrite cancel/confirm, export ambiguity abort, counts without unique soma, mobile layout')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--chromium',required=True);args=parser.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True);fixtures(root)
    before={p:(p.read_bytes(),p.stat().st_mtime_ns,p.stat().st_mode) for p in root.rglob('*.swc')}
    from swclib.visualization.server import create_server
    server=create_server(port=0,swc_workers=2);threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        with sync_playwright() as playwright:
            browser=playwright.chromium.launch(executable_path=args.chromium,args=['--no-sandbox','--enable-unsafe-swiftshader','--use-angle=swiftshader'])
            page=browser.new_page(viewport={'width':1280,'height':1000},bypass_csp=True)
            page.goto(f'http://127.0.0.1:{server.server_port}');result=check(page,root)
            assert all(snapshot==(p.read_bytes(),p.stat().st_mtime_ns,p.stat().st_mode) for p,snapshot in before.items())
            result['source_integrity']='all bytes, mtimes and modes unchanged'
            (root/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True);browser.close()
    finally:server.shutdown();server.server_close()


if __name__=='__main__':main()
