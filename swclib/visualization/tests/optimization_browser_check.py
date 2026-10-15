"""Real browser checks for bounded plane suggestions and stale-response safety."""
import argparse
import json
import math
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright


def check(page, root):
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.wait_for_function('!!window.swcStudio')
    page.locator('#swc-path').fill(str(root/'neurons'))
    page.locator('#swc-form button').click()
    page.wait_for_function('swcStudio.renderer.count===4')
    page.locator('#tab-plugins').click()
    page.locator('#plugin-soma summary').click()
    page.locator('#hemisphere-visible').check()
    assert page.locator('#hemisphere-max-angle').input_value() == '10'
    results = []
    page.on('response', lambda response: results.append(response.json()) if response.url.endswith('/api/hemisphere/optimize') else None)
    def optimize():
        page.locator('#hemisphere-optimize').click()
        page.wait_for_function('!document.getElementById("hemisphere-optimize").disabled')
    optimize()
    assert '微调完成' in page.locator('#hemisphere-optimize-status').inner_text()
    assert [page.locator('#hemisphere-'+key).inner_text() for key in ['left-only','right-only','cross']] == ['1','1','0']
    assert page.locator('#hemisphere-suggestions option').count() == 3
    for i in range(3):
        page.locator('#hemisphere-suggestions').select_option(str(i))
        state = page.evaluate('swcStudio.renderer.hemispherePlane')
        expected = results[0]['suggestions'][i]
        assert all(state[key] == expected['plane'][key] for key in ['azimuth_deg','elevation_deg','offset'])
        assert abs(sum(n*c for n,c in zip(expected['plane']['normal_xyz'], [100,200,300]))-state['offset']) < 1e-10
        assert expected['angle_change_deg'] <= 10
        for control,key in [('azimuth','azimuth_deg'),('elevation','elevation_deg'),('offset','offset')]:
            assert float(page.locator('#hemisphere-'+control).input_value()) == state[key]
            assert abs(float(page.locator('#hemisphere-'+control+'-range').input_value()) - state[key]) < 1e-9
            assert page.locator('#hemisphere-'+control).evaluate('(input)=>input.validity.valid')
    # Selected suggestion remains usable by both count and JSON export.
    page.locator('#hemisphere-count').click()
    page.wait_for_function('!document.getElementById("hemisphere-count").disabled')
    assert page.locator('#hemisphere-cross').inner_text() == str(expected['counts']['cross'])
    page.locator('#hemisphere-directory').fill(str(root/'selected.json'))
    page.locator('#hemisphere-export').click()
    page.wait_for_function('document.getElementById("hemisphere-status").textContent.startsWith("已导出")')
    report = json.loads((root/'selected.json').read_text())
    assert report['plane'] == expected['plane']
    page.screenshot(path=str(root/'suggestions.png'))
    # A manual parameter change invalidates suggestions; zero only recenters.
    page.locator('#hemisphere-max-angle').fill('0')
    assert page.locator('#hemisphere-suggestions').is_hidden()
    state = page.evaluate('swcStudio.renderer.hemispherePlane')
    page.locator('#hemisphere-offset').fill('-100')
    optimize()
    assert results[-1]['angle_change_deg'] == 0
    assert page.locator('#hemisphere-suggestions option').count() == 1
    actual = page.evaluate('swcStudio.renderer.hemispherePlane')
    assert actual['azimuth_deg'] == state['azimuth_deg'] and actual['elevation_deg'] == state['elevation_deg']
    page.locator('#hemisphere-max-angle').fill('3')
    optimize()
    assert results[-1]['max_angle_deg'] == 3
    assert all(candidate['angle_change_deg'] <= 3 for candidate in results[-1]['suggestions'])
    # Old replies must not overwrite later edits or a changed visible center.
    for change in ['plane', 'center', 'limit']:
        def stale(route):
            response = route.fetch()
            if change == 'plane':
                page.locator('#hemisphere-offset').fill('-333')
            elif change == 'limit':
                page.locator('#hemisphere-max-angle').fill('4')
            else:
                page.evaluate('swcStudio.renderer.cropStart=[1,2,3]')
            route.fulfill(response=response)
        page.route('**/api/hemisphere/optimize', stale)
        optimize()
        assert '未应用旧结果' in page.locator('#hemisphere-optimize-status').inner_text()
        assert page.locator('#hemisphere-suggestions').is_hidden()
        if change == 'plane':
            assert page.evaluate('swcStudio.renderer.hemispherePlane.offset') == -333
        page.unroute('**/api/hemisphere/optimize', stale)
    before = page.evaluate('swcStudio.renderer.hemispherePlane')
    page.route('**/api/hemisphere/optimize', lambda route: route.fulfill(status=404,content_type='application/json',body='{"error":"Not found"}'))
    optimize()
    assert '重启服务' in page.locator('#hemisphere-optimize-status').inner_text()
    assert page.evaluate('swcStudio.renderer.hemispherePlane') == before
    assert page.evaluate('swcStudio.renderer.gl.getError()') == 0
    page.set_viewport_size(dict(width=390,height=844))
    assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
    assert not errors, errors
    return dict(checks='angular cap and exact center, three selectable suggestions, full counts, slider precision, selected-plane export, zero-angle centering, configurable cap, stale plane/center/limit rejection, old-backend message, narrow viewport')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--chromium', required=True)
    args = parser.parse_args()
    root = Path(args.output);(root/'neurons').mkdir(parents=True,exist_ok=True)
    for side in (-1, 1):
        (root/'neurons'/f'{side}.swc').write_text(f'1 1 99.8 {200+side*10} 299.5 1 -1\n2 3 100.2 {200+side*10} 300.5 1 1\n')
    before = {p:(p.read_bytes(),p.stat().st_mtime_ns,p.stat().st_mode) for p in (root/'neurons').glob('*.swc')}
    from swclib.visualization.server import create_server
    server = create_server(port=0,swc_workers=2)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=args.chromium,args=['--no-sandbox','--enable-unsafe-swiftshader','--use-angle=swiftshader'])
            page = browser.new_page(viewport=dict(width=1280,height=1100),bypass_csp=True)
            page.goto(f'http://127.0.0.1:{server.server_port}')
            result = check(page,root)
            assert all(snapshot==(p.read_bytes(),p.stat().st_mtime_ns,p.stat().st_mode) for p,snapshot in before.items())
            result['source_integrity'] = 'bytes, mtimes and modes unchanged'
            (root/'results.json').write_text(json.dumps(result,indent=2))
            print(json.dumps(result,indent=2),flush=True)
            browser.close()
    finally:
        server.shutdown();server.server_close()


if __name__ == '__main__':
    main()
