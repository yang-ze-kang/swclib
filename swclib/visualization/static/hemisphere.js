export function mountHemisphere(body,viewer) {
  body.innerHTML=`
    <p class="hint">调整全局坐标中的分隔平面，确定左右半脑。</p>
    <label class="check"><input id="hemisphere-visible" type="checkbox">显示划分平面</label>
    <div class="hemisphere-key"><span>左半脑 · −n</span><span>右半脑 · +n</span></div>
    <label for="hemisphere-azimuth">方位角（°）</label><div class="plane-control"><input id="hemisphere-azimuth" type="number" step="any" value="0"><input id="hemisphere-azimuth-range" aria-label="方位角滑块" type="range" min="0" max="360" step="any" value="0"></div>
    <label for="hemisphere-elevation">仰角（°）</label><div class="plane-control"><input id="hemisphere-elevation" type="number" min="-90" max="90" step="any" value="0"><input id="hemisphere-elevation-range" aria-label="仰角滑块" type="range" min="-90" max="90" step="any" value="0"></div>
    <label for="hemisphere-offset">法向偏移（全局体素坐标）</label><div class="plane-control"><input id="hemisphere-offset" type="number" step="any" value="0"><input id="hemisphere-offset-range" aria-label="法向偏移滑块" type="range" min="-100" max="100" step="any" value="0"></div>
    <button id="hemisphere-center" class="quiet full-width" type="button">平面经过可见数据中心</button>
    <div id="hemisphere-equation" class="small-status"></div>
    <p class="hint">方位角从 +X 朝 +Y 旋转，仰角朝 +Z。沿法向量 n 的正侧为右，负侧为左；平面上的点归右。Crop 不改变全局平面参数。</p>
    <button id="hemisphere-count" class="secondary full-width" type="button">统计左右脑神经元数量</button>
    <p class="hint">统计当前可见神经元的完整 SWC，沿用导出的线段长度分类；左右合计均包含跨侧神经元。</p>
    <dl id="hemisphere-counts" class="hemisphere-counts" hidden>
      <div><dt>完全左侧</dt><dd id="hemisphere-left-only">0</dd></div>
      <div><dt>完全右侧</dt><dd id="hemisphere-right-only">0</dd></div>
      <div><dt>跨两侧</dt><dd id="hemisphere-cross">0</dd></div>
      <div><dt>总数</dt><dd id="hemisphere-total">0</dd></div>
      <div><dt>左侧合计（含跨侧）</dt><dd id="hemisphere-left-total">0</dd></div>
      <div><dt>右侧合计（含跨侧）</dt><dd id="hemisphere-right-total">0</dd></div>
    </dl>
    <div id="hemisphere-count-status" class="small-status" role="status">点击按钮统计当前平面下的神经元数量。</div>
    <label for="hemisphere-max-angle">最大调整角度（°）</label><input id="hemisphere-max-angle" type="number" min="0" max="90" step="any" value="10" required>
    <button id="hemisphere-optimize" class="secondary full-width" type="button">微调平面并平衡两侧</button>
    <p class="hint">平面经过可见数据包围盒中心。在设定角度内搜索：先缩小完全左／右数量差；差小于 5 后优先减少跨侧数。分类使用可见神经元的完整 SWC。</p>
    <div id="hemisphere-optimize-status" class="small-status" role="status">默认最多转动 10°；设为 0° 时只居中。</div>
    <label for="hemisphere-suggestions" id="hemisphere-suggestions-label" hidden>建议平面（切换即预览）</label>
    <select id="hemisphere-suggestions" class="full-width" hidden></select>
    <label for="hemisphere-directory">导出目录或 JSON 文件路径</label><input id="hemisphere-directory" placeholder="/data/results 或 /data/results/custom.json" autocomplete="off">
    <p class="hint">导出当前显示 SWC 的完整原文件统计，不受 crop 范围影响。存在多个或没有胞体候选时停止导出。</p>
    <button id="hemisphere-export" class="primary full-width" type="button">导出 JSON</button>
    <div id="hemisphere-status" class="small-status" role="status">目录需已存在；填写目录时保存为 brain_hemisphere.json。已有文件会询问是否覆盖。</div>`;
  const $=id=>body.querySelector('#hemisphere-'+id);
  let initialized=false,exporting=false,counting=false,optimizing=false,countRevision=0,limitRevision=0,planeSignature='';
  let suggestions=[],suggestionContext=null,applyingSuggestion=false;
  const state={visible:false,azimuth_deg:0,elevation_deg:0,offset:0};
  const parameters=()=>({azimuth_deg:state.azimuth_deg,elevation_deg:state.elevation_deg,offset:state.offset});
  function invalidateCounts() {
    countRevision++;
    if(!$('counts').hidden||counting)$('count-status').textContent='平面或可见神经元已变化，请重新统计。';
    $('counts').hidden=true;
    if(!applyingSuggestion)clearSuggestions();
    if($('optimize-status').textContent.startsWith('微调完成'))$('optimize-status').textContent='平面或可见神经元已变化，可重新微调。';
  }
  viewer.onVisibleNeuronsChanged(invalidateCounts);
  function clearSuggestions() {suggestions=[];suggestionContext=null;$('suggestions').hidden=true;$('suggestions-label').hidden=true;}
  function applySuggestion(candidate) {
    applyingSuggestion=true;
    try {
      for(const [id,key] of [['azimuth','azimuth_deg'],['elevation','elevation_deg'],['offset','offset']]) {
        state[key]=candidate.plane[key];$(id).setCustomValidity('');$(id).value=state[key];$(id+'-range').value=state[key];
      }
      initialized=true;state.visible=true;$('visible').checked=true;update();showCounts(candidate.counts);
      suggestionContext.revision=countRevision;
      const difference=c=>Math.abs(c.left_only-c.right_only),before=suggestionContext.baseline,after=candidate.counts;
      $('optimize-status').textContent='微调完成：居中基准 → 当前建议，左右数量差 '+difference(before)+' → '+difference(after)+'，跨侧 '+before.cross+' → '+after.cross+'；转动 '+Number(candidate.angle_change_deg.toFixed(4))+'°，平面已过可见数据中心。';
    }finally{applyingSuggestion=false;}
  }
  $('suggestions').addEventListener('change',()=>{
    if(!suggestionContext)return;
    if(suggestionContext.revision!==countRevision||suggestionContext.limit!==limitRevision||suggestionContext.ids!==JSON.stringify(viewer.visibleNeurons())||suggestionContext.center!==JSON.stringify(visibleCenter())) {
      clearSuggestions();$('optimize-status').textContent='可见数据或参数已变化，请重新微调。';return;
    }
    applySuggestion(suggestions[Number($('suggestions').value)]);
  });
  function updateActions() {for(const id of ['count','optimize','export'])$(id).disabled=counting||optimizing||exporting;}
  function showCounts(counts) {
    for(const [id,key] of [['left-only','left_only'],['right-only','right_only'],['cross','cross'],['total','total'],['left-total','left'],['right-total','right']])
      $(id).textContent=counts[key].toLocaleString();
    $('counts').hidden=false;$('count-status').textContent='统计完成：'+counts.total+' 个神经元。';
  }
  function visibleCenter() {const box=viewer.planeBounds();return box?box[0].map((v,i)=>(v+box[1][i])/2):null;}
  function normal() {const a=state.azimuth_deg*Math.PI/180,e=state.elevation_deg*Math.PI/180;return [Math.cos(e)*Math.cos(a),Math.cos(e)*Math.sin(a),Math.sin(e)];}
  function offsetRange() {
    const box=viewer.planeBounds(),n=normal();
    let lo=-100,hi=100;
    if(box){lo=0;hi=0;for(let i=0;i<3;i++){lo+=n[i]*box[n[i]<0?1:0][i];hi+=n[i]*box[n[i]<0?0:1][i];}const margin=Math.max((hi-lo)*.15,1);lo-=margin;hi+=margin;}
    $('offset-range').min=Math.min(lo,state.offset);$('offset-range').max=Math.max(hi,state.offset);$('offset-range').value=state.offset;
  }
  function update() {
    const signature=JSON.stringify(parameters());
    if(signature!==planeSignature){planeSignature=signature;invalidateCounts();}
    offsetRange();const n=normal();
    $('equation').textContent='n = ('+n.map(v=>Number(v.toFixed(6))).join(', ')+') · n·XYZ = '+Number(state.offset.toFixed(6));
    viewer.setPlane({...state});
  }
  function centerPlane() {
    const center=visibleCenter();if(!center){$('status').textContent='请先添加并显示 SWC 或图像。';return false;}
    const n=normal();state.offset=center.reduce((sum,v,i)=>sum+n[i]*v,0);
    $('offset').setCustomValidity('');$('offset').value=state.offset;initialized=true;return true;
  }
  for(const [id,key] of [['azimuth','azimuth_deg'],['elevation','elevation_deg'],['offset','offset']]) {
    const number=$(id),range=$(id+'-range');
    number.addEventListener('input',()=>{
      const value=number.valueAsNumber;
      if(!Number.isFinite(value)||(id==='elevation'&&(value < -90||value > 90))){number.setCustomValidity('请输入有效数值，仰角范围为 -90° 到 90°');number.reportValidity();return;}
      number.setCustomValidity('');state[key]=id==='azimuth'?((value%360)+360)%360:value;
      range.value=state[key];initialized=true;update();
    });
    number.addEventListener('change',()=>{if(number.validity.valid)number.value=state[key];});
    range.addEventListener('input',()=>{number.setCustomValidity('');state[key]=Number(range.value);number.value=state[key];initialized=true;update();});
  }
  $('visible').addEventListener('change',()=>{state.visible=$('visible').checked;if(state.visible&&!initialized)centerPlane();update();});
  $('center').addEventListener('click',()=>{if(centerPlane()){state.visible=true;$('visible').checked=true;update();}});
  $('count').addEventListener('click',async()=>{
    if(counting||exporting||optimizing)return;
    if(!['azimuth','elevation','offset'].every(id=>$(id).reportValidity()))return;
    const ids=viewer.visibleNeurons();
    if(!ids.length){$('count-status').textContent='请先加载并显示至少一个 SWC。';return;}
    const current=countRevision,plane=parameters();
    counting=true;updateActions();$('counts').hidden=true;
    $('count-status').textContent='正在统计 '+ids.length+' 个完整 SWC…';
    try {
      const result=await viewer.countHemisphere({ids,plane});
      if(current!==countRevision||JSON.stringify(ids)!==JSON.stringify(viewer.visibleNeurons()))return;
      showCounts(result.counts);
    }catch(error){if(current===countRevision)$('count-status').textContent='统计失败：'+error.message;}
    finally{counting=false;updateActions();}
  });
  $('max-angle').addEventListener('input',()=>{limitRevision++;clearSuggestions();});
  $('optimize').addEventListener('click',async()=>{
    if(counting||exporting||optimizing)return;
    if(!['azimuth','elevation','offset','max-angle'].every(id=>$(id).reportValidity()))return;
    const ids=viewer.visibleNeurons(),center=visibleCenter(),limit=$('max-angle').valueAsNumber;
    if(!ids.length||!center){$('optimize-status').textContent='请先加载并显示至少一个 SWC。';return;}
    if(!Number.isFinite(limit)){$('optimize-status').textContent='请输入 0° 到 90° 的最大调整角度。';return;}
    const current=countRevision,currentLimit=limitRevision,plane=parameters();
    optimizing=true;updateActions();clearSuggestions();
    $('optimize-status').textContent='正在为 '+ids.length+' 个完整 SWC 搜索平面方向…';
    try {
      const result=await viewer.optimizeHemisphere({ids,plane,center,max_angle_deg:limit});
      if(current!==countRevision||currentLimit!==limitRevision||!['azimuth','elevation','offset'].every(id=>$(id).validity.valid)||JSON.stringify(ids)!==JSON.stringify(viewer.visibleNeurons())||JSON.stringify(center)!==JSON.stringify(visibleCenter())) {
        $('optimize-status').textContent='平面、可见数据或角度上限已变化，未应用旧结果，请重新微调。';return;
      }
      suggestions=result.suggestions;
      suggestionContext={revision:countRevision,limit:limitRevision,ids:JSON.stringify(ids),center:JSON.stringify(center),baseline:result.centered_counts};
      $('suggestions').replaceChildren(...suggestions.map((candidate,i)=>{
        const option=document.createElement('option'),c=candidate.counts;option.value=i;
        option.textContent=(i+1)+'. 左 '+c.left_only+' / 右 '+c.right_only+' / 跨 '+c.cross+' · '+Number(candidate.angle_change_deg.toFixed(2))+'°';return option;
      }));
      $('suggestions').hidden=false;$('suggestions-label').hidden=false;applySuggestion(suggestions[0]);
    }catch(error){$('optimize-status').textContent='微调失败：'+(error.status===404?'当前后端不支持此接口，请重启服务后刷新页面。':error.message);}
    finally{optimizing=false;updateActions();}
  });
  $('export').addEventListener('click',async()=>{
    if(exporting||counting||optimizing)return;
    if(!['azimuth','elevation','offset'].every(id=>$(id).reportValidity()))return;
    const path=$('directory').value.trim();if(!path){$('status').textContent='请输入导出目录或 JSON 文件路径。';return;}
    const ids=viewer.visibleNeurons();if(!ids.length){$('status').textContent='请先加载并显示至少一个 SWC。';return;}
    const plane=parameters();
    exporting=true;$('export').disabled=true;
    for(const control of body.querySelectorAll('input,button,select'))control.disabled=true;
    $('status').textContent='正在统计 '+ids.length+' 个完整 SWC 并导出…';
    try {
      let result;
      try {result=await viewer.exportHemisphere({ids,plane,path});}
      catch(error) {
        if(error.code!=='export_exists')throw error;
        if(!window.confirm('文件已存在：\n'+error.path+'\n\n是否覆盖？')) {
          $('status').textContent='已取消导出，已有文件保持不变。';return;
        }
        $('status').textContent='正在覆盖导出…';
        result=await viewer.exportHemisphere({ids,plane,path:error.path,overwrite:true});
      }
      $('status').textContent='已导出 '+result.count+' 个神经元：'+result.path;
    }catch(error){$('status').textContent='导出失败：'+error.message;}
    finally{exporting=false;for(const control of body.querySelectorAll('input,button,select'))control.disabled=false;}
  });
  update();
}
