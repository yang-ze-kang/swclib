"""Test probability inspection and jumps to exact recorded predictions in Chromium."""
import argparse
import json
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright


def check(page,root):
    errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
    page.wait_for_function('!!window.swcStudio')
    page.locator('#tab-plugins').click()
    page.locator('#error-mode').select_option('trace')
    assert not page.locator('#error-pred').is_visible()
    assert page.locator('#error-gt').get_attribute('required') is None
    page.locator('#error-trace-path').fill(str(root/'cross_step'/'trace_steps.jsonl'))
    page.locator('#error-detect').click()
    page.wait_for_function('swcStudio.renderer.errors?.event?.trace_origin?.start?.probability===0.82')
    page.locator('#error-type').select_option('unsupported_path_candidate')
    assert 'p=0.8200' in page.locator('.error-event.active').inner_text()
    assert 'p=0.820000' in page.locator('#error-predictions').inner_text()
    assert 'p=0.930000' in page.locator('#error-predictions').inner_text()
    assert page.locator('.error-replay-jump').count()==2
    assert page.evaluate('swcStudio.renderer.errors.markers.some(m=>m.label==="mismatch p=0.8200")')
    assert '不是确定的错误起点' in page.locator('#error-review').inner_text()
    assert page.evaluate('swcStudio.renderer.errors.reviewBounds!==null')
    page.locator('#error-review-focus').click()
    page.locator('#error-focus').click()
    page.locator('.error-replay-jump[data-prediction-index="0"]').click()
    page.wait_for_function('swcStudio.renderer.replay?.active&&swcStudio.renderer.replay.data?.prediction.step===1')
    assert page.locator('.trace-error-candidate').count()==1
    assert 'p=0.8200' in page.locator('.trace-error-candidate').inner_text()
    page.locator('#scene').click(position=dict(x=250,y=250));page.keyboard.press('ArrowRight')
    page.wait_for_function('swcStudio.renderer.replay.data.prediction.step===2')
    assert page.locator('.trace-error-candidate').count()==0
    page.keyboard.press('ArrowLeft');page.wait_for_function('swcStudio.renderer.replay.data.prediction.step===1')
    page.locator('.error-event').click()
    page.wait_for_function('swcStudio.renderer.errors.active')
    page.locator('.error-replay-jump[data-prediction-index="1"]').click()
    page.wait_for_function('swcStudio.renderer.replay.active&&swcStudio.renderer.replay.data.prediction.step===2')
    assert 'p=0.9300' in page.locator('.trace-error-candidate').inner_text()
    page.locator('.error-event').click()
    page.evaluate("swcStudio.renderer.setView('xy')")
    page.screenshot(path=str(root/'probabilities.png'))
    # Deviation starts in the previous prediction while still inside tolerance.
    page.locator('#error-trace-path').fill(str(root/'earlier_step'))
    page.locator('#error-detect').click()
    page.wait_for_function('swcStudio.renderer.errors?.event?.trace_origin?.start?.probability===0.73')
    assert '失配边预测 2' in page.locator('.error-event.active').inner_text()
    assert page.evaluate('JSON.stringify(swcStudio.renderer.errors.event.onset_review.prediction_indices)==="[0,1]"')
    previous=page.locator('.error-prediction-card[data-prediction-index="0"]')
    if not previous.get_attribute('open') is not None: previous.locator('summary').first.click()
    assert 'p=0.910000' in previous.inner_text()
    assert '不指定它是最早错误预测' in previous.inner_text()
    page.locator('#error-review>details>summary').click()
    page.locator('.error-review-edge').nth(1).click()
    assert page.evaluate('JSON.stringify(swcStudio.renderer.center)==="[20,1,0]"')
    previous.locator('.error-replay-jump').click()
    page.wait_for_function('swcStudio.renderer.replay?.active&&swcStudio.renderer.replay.data?.prediction.step===1')
    assert 'p=0.9100' in page.locator('.trace-error-candidate').inner_text()
    page.locator('.error-event').click()
    page.locator('#error-review-focus').click()
    page.screenshot(path=str(root/'onset_review.png'))
    # No selected candidate at a stop: scores remain inspectable but are not
    # displayed as a made-up break probability.
    page.locator('#error-type').select_option('all')
    page.locator('#error-trace-path').fill(str(root/'break'))
    page.locator('#error-detect').click()
    page.wait_for_function('swcStudio.renderer.errors?.event?.kind==="break_error"')
    assert '无法唯一归因' in page.locator('#error-predictions').inner_text()
    assert page.locator('#error-review').inner_text()==''
    assert page.locator('#error-review-focus').is_disabled()
    assert 'p=' not in page.locator('.error-event.active').inner_text()
    page.locator('.error-prediction-card>summary').click()
    page.locator('.error-prediction-card details>summary').click()
    assert '#1: 0.400000' in page.locator('.error-all-scores').inner_text()
    page.locator('.error-replay-jump').click()
    page.wait_for_function('swcStudio.renderer.replay.active&&swcStudio.renderer.replay.data.prediction.candidates.length===0')
    page.locator('#error-type').select_option('unsupported_path_candidate')
    page.locator('#error-trace-path').fill(str(root/'initial'))
    page.locator('#error-detect').click()
    page.wait_for_function('swcStudio.renderer.errors?.event?.trace_origin?.status==="initial_tree"')
    assert '初始树' in page.locator('#error-predictions').inner_text()
    assert page.locator('.error-replay-jump').count()==0
    assert page.evaluate('swcStudio.renderer.gl.getError()')==0
    assert not errors,errors
    return dict(passed=True,errors=errors,checks=['trace input mode','manifest GT','start/confirmation probabilities',
        'candidate highlight','exact replay index','arrows after jump','no forced break probability',
        'inspect rejected scores','initial-tree origin','uncertain upstream interval','earlier prediction probability',
        'click upstream edge','jump to preceding prediction','no forced causal onset','WebGL'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--chromium',required=True)
    args=parser.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    from swclib.error_detection.tests.test_trace_detection import fixture,prediction
    fixture(root/'cross_step')
    fixture(root/'earlier_step',[
        prediction(1,1,[20,0,0],[[2,1,20,2,0]],[[[20,0,0],[20,2,0]]],.91),
        prediction(2,2,[20,2,0],[[3,2,20,12,0]],[[[20,2,0],[20,12,0]]],.73)])
    fixture(root/'break',[prediction(1,1,[20,0,0],[],[],candidates=[])])
    fixture(root/'initial',[],([[0,0,0],[20,0,0],[20,12,0]],[-1,0,1]))
    from swclib.visualization.server import create_server
    server=create_server(port=0,swc_workers=1);threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path=args.chromium,args=['--no-sandbox','--enable-unsafe-swiftshader','--use-angle=swiftshader'])
            page=browser.new_page(viewport=dict(width=1450,height=1200),bypass_csp=True)
            page.goto(f'http://127.0.0.1:{server.server_port}')
            result=check(page,root);(root/'results.json').write_text(json.dumps(result,indent=2))
            print(json.dumps(result),flush=True);browser.close()
    finally:server.shutdown();server.server_close()


if __name__=='__main__':main()
