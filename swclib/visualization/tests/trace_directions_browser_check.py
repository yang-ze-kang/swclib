"""Real-browser direction and replay-toggle checks; writes only to --output."""
import argparse
import json
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright


def check(page, root):
    errors, requests = [], []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('request', lambda request: requests.append(request.url))
    page.wait_for_function('!!window.swcStudio')
    page.locator('#tab-plugins').click()
    page.locator('#trace-path').fill(str(root))
    page.locator('#trace-load').click()
    page.wait_for_function('swcStudio.renderer.replay?.data?.index===0')
    assert page.locator('#trace-directions').is_checked()
    assert page.evaluate('swcStudio.renderer.replay.showDirections')
    assert not any('/api/volume' in url for url in requests)
    page.evaluate("swcStudio.renderer.setView('xy')")
    page.wait_for_timeout(100)
    shown = page.locator('.trace-labels').evaluate('(canvas)=>canvas.toDataURL()')
    position = page.locator('#trace-position').input_value()
    before = len(requests)
    page.locator('#trace-directions').uncheck()
    page.wait_for_function('!swcStudio.renderer.replay.showDirections')
    page.wait_for_timeout(100)
    hidden = page.locator('.trace-labels').evaluate('(canvas)=>canvas.toDataURL()')
    assert shown != hidden
    assert len(requests) == before and page.locator('#trace-position').input_value() == position
    page.locator('#trace-next').click()
    page.wait_for_function('swcStudio.renderer.replay.data.index===1')
    assert not page.evaluate('swcStudio.renderer.replay.showDirections')
    page.locator('#trace-directions').check()
    page.locator('#trace-prev').click()
    page.wait_for_function('swcStudio.renderer.replay.data.index===0')
    assert page.evaluate('swcStudio.renderer.replay.showDirections')
    # Check the rendered arrow apex, independently of SWC node spacing, world
    # offset and camera rotation. No hidden candidate/GT topology is modified.
    result = page.evaluate('''async () => {
      const {drawDirectionArrows}=await import('/trace-renderer.js');
      const center=[20000,7000,6000],all=[];
      for(const reverse of [false,true])for(const rotated of [false,true]) {
        const paths=[];let path;
        const ctx={beginPath(){},stroke(){},moveTo(x,y){path=[[x,y]];paths.push(path);},lineTo(x,y){path.push([x,y]);}};
        let edges=[[[19900,7000,6000],[20100,7000,6000]]];
        if(reverse)edges=edges.map(([a,b])=>[b,a]);
        drawDirectionArrows(ctx,[{edges,color:'gray'}],{center,width:600,height:400,span:100,
          right:rotated?[0,1,0]:[1,0,0],up:rotated?[-1,0,0]:[0,1,0]});
        if(!paths.length)throw new Error('No direction arrows');
        const axis=rotated?1:0;
        for(const [a,tip,b] of paths)
          if((tip[axis]-a[axis])*(reverse?-1:1)<=0||(tip[axis]-b[axis])*(reverse?-1:1)<=0)
            throw new Error('Arrow points against parent-to-child direction');
        all.push(paths.length);
      }
      return {arrows_per_case:all};
    }''')
    page.screenshot(path=str(root/'arrows.png'))
    assert page.evaluate('swcStudio.renderer.gl.getError()') == 0
    page.locator('#trace-close').click()
    page.wait_for_function('!swcStudio.renderer.replay.active')
    assert not errors, errors
    return dict(result, errors=errors, checks='parent-to-child, rotation, large offset, default on, toggle without fetch, navigation, close')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--chromium', required=True)
    args = parser.parse_args()
    root = Path(args.output);root.mkdir(parents=True, exist_ok=True)
    from swclib.visualization.tests.test_trace_replay import fixture
    fixture(root)
    gt = root/'gt.swc'
    gt.write_text('1 1 -50 25 30 1 -1\n2 3 60 25 30 1 1\n3 3 60 -25 30 1 2\n')
    (root/'trace_replay.json').write_text(json.dumps({'ground_truth':str(gt)}))
    from swclib.visualization.server import create_server
    server = create_server(port=0, swc_workers=1)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=args.chromium,
                args=['--no-sandbox','--enable-unsafe-swiftshader','--use-angle=swiftshader'])
            page = browser.new_page(viewport=dict(width=1400,height=1000), bypass_csp=True)
            page.goto(f'http://127.0.0.1:{server.server_port}')
            result = check(page, root)
            (root/'results.json').write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
            browser.close()
    finally:
        server.shutdown();server.server_close()


if __name__ == '__main__':
    main()
