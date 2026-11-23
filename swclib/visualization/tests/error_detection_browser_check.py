"""Real WebGL/browser checks. Run with --output DIR --chromium EXECUTABLE."""
import argparse
import json
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright


def check(page, root):
    errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
    page.wait_for_function('!!window.swcStudio')
    page.locator('#tab-plugins').click()
    page.locator('#error-pred').fill(str(root/'pred.swc'))
    page.locator('#error-gt').fill(str(root/'gt.swc'))
    page.locator('#error-detect').click()
    page.wait_for_function('swcStudio.renderer.errors?.active && !document.querySelector("#error-detect").disabled')
    assert page.locator('#error-first').is_checked()
    page.locator('#error-category').select_option('confirmed')
    page.locator('#error-type').select_option('wrong_connection')
    assert page.locator('.error-event').count()==1
    first=page.evaluate('swcStudio.renderer.errors.event.id')
    layer=page.evaluate('({event:swcStudio.renderer.errors.event,center:swcStudio.renderer.center})')
    point=layer['event'].get('onset_estimate',{}).get('estimated_onset') or layer['event']['start']
    assert layer['center']==point['xyz']
    assert layer['event']['affected_subtree']['node_ids']==[3,4,5,6]
    assert page.evaluate("async()=>{const {affectedEdges}=await import('/error-renderer.js');const r=swcStudio.renderer.errors;return JSON.stringify(affectedEdges(r.event,r.pair.pred.nodes)[0][0])===JSON.stringify(r.event.start.xyz)}")
    page.locator('#error-first').uncheck()
    assert page.locator('.error-event').count()==2
    page.locator('#scene').click(position=dict(x=200,y=200));page.keyboard.press('ArrowRight')
    page.wait_for_function('swcStudio.renderer.errors.event.is_first_fj===false')
    second=page.evaluate('swcStudio.renderer.errors.event.id')
    page.keyboard.press('ArrowRight')
    assert page.evaluate('swcStudio.renderer.errors.event.id')==second
    page.locator('#error-pred').focus();page.keyboard.press('ArrowLeft')
    assert page.evaluate('swcStudio.renderer.errors.event.id')==second
    page.locator('#scene').click(position=dict(x=200,y=200));page.keyboard.press('ArrowLeft')
    page.wait_for_function(f'swcStudio.renderer.errors.event.id==="{first}"')
    page.locator('#error-first').check()
    assert page.locator('.error-event').count()==1
    assert 'Fiber TP' in page.locator('#error-status').inner_text()
    page.evaluate("swcStudio.renderer.setView('xy')")
    page.screenshot(path=str(root/'errors.png'))
    # The replay plugin owns arrows while active.
    page.locator('#trace-path').fill(str(root/'replay'))
    page.locator('#trace-load').click()
    page.wait_for_function('swcStudio.renderer.replay?.active && swcStudio.renderer.replay.data?.index===0')
    assert not page.evaluate('swcStudio.renderer.errors.active')
    page.locator('#scene').click(position=dict(x=200,y=200));page.keyboard.press('ArrowRight')
    page.wait_for_function('swcStudio.renderer.replay.data.index===1')
    page.keyboard.press('ArrowLeft');page.wait_for_function('swcStudio.renderer.replay.data.index===0')
    page.locator('#trace-path').focus();page.keyboard.press('ArrowRight')
    assert page.evaluate('swcStudio.renderer.replay.data.index')==0
    page.locator(f'[data-event-id="{first}"]').click()
    page.wait_for_function('swcStudio.renderer.errors.active && !swcStudio.renderer.replay.active')
    page.locator('#error-type').select_option('all')
    page.locator('#error-category').select_option('all')
    page.locator('#error-pred').fill(str(root/'gt.swc'))
    page.locator('#error-detect').click()
    page.wait_for_function('swcStudio.renderer.errors.active && !document.querySelector("#error-detect").disabled')
    assert page.locator('.error-event').count()==0
    assert page.evaluate('swcStudio.renderer.errors.layers[0].count')>0
    page.locator('#error-pred').fill(str(root/'empty.swc'))
    page.locator('#error-detect').click()
    page.wait_for_function('swcStudio.renderer.errors.event?.immediate_termination===true')
    assert page.locator('.error-event').count()==1
    assert page.evaluate('swcStudio.renderer.gl.getError()')==0
    page.locator('#error-close').click()
    assert not page.evaluate('swcStudio.renderer.errors.active')
    assert not errors,errors
    return dict(passed=True,errors=errors,checks=['original edge clipping','descendant IDs','first/all confirmed FJ',
        'status and type filters','onset focus','arrow boundaries','input focus',
        'exclusive plugin navigation','original Fiber scores','zero errors','empty prediction','WebGL'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--chromium',required=True)
    args=parser.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    (root/'pred.swc').write_text('1 1 0 0 0 1 -1\n2 3 35 0 0 1 1\n3 3 35 10 0 1 2\n4 3 45 10 0 1 3\n5 3 45 0 0 1 4\n6 3 60 0 0 1 5\n')
    (root/'gt.swc').write_text('1 1 0 0 0 1 -1\n2 3 20 0 0 1 1\n3 3 60 0 0 1 2\n4 3 20 10 0 1 2\n5 3 60 10 0 1 4\n6 3 20 -40 0 1 2\n')
    (root/'empty.swc').write_text('# empty prediction\n')
    from swclib.visualization.tests.test_trace_replay import fixture
    fixture(root/'replay')
    from swclib.visualization.server import create_server
    server=create_server(port=0,swc_workers=1)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path=args.chromium,args=['--no-sandbox','--enable-unsafe-swiftshader','--use-angle=swiftshader'])
            page=browser.new_page(viewport=dict(width=1450,height=1100),bypass_csp=True)
            page.goto(f'http://127.0.0.1:{server.server_port}')
            result=check(page,root)
            (root/'results.json').write_text(json.dumps(result,indent=2))
            print(json.dumps(result),flush=True);browser.close()
    finally:
        server.shutdown();server.server_close()


if __name__=='__main__':main()
