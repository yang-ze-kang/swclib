// Built-in plugins receive a small viewer bridge; source data stays read-only.
import {mountHemisphere} from './hemisphere.js';
const plugins=[{id:'soma',name:'查看胞体',mount:mountSoma},{id:'hemisphere',name:'划分脑区',mount:mountHemisphere}];

export function mountPlugins(container,viewer) {
  for(const plugin of plugins) {
    const details=document.createElement('details');details.className='plugin-card';details.id='plugin-'+plugin.id;details.open=true;
    const summary=document.createElement('summary');summary.textContent=plugin.name;
    const body=document.createElement('div');body.className='plugin-body';details.append(summary,body);container.append(details);
    plugin.mount(body,viewer);
  }
}

function centeredCube(center,length,bounds) {
  if(!Number.isInteger(length)||length<1||length>300)throw new Error('crop 边长需要为 1–300 的整数体素');
  const [lo,hi]=bounds;
  if(center.some((v,i)=>!Number.isFinite(v)||v<lo[i]||v>=hi[i]))throw new Error('胞体候选超出全脑数据范围');
  if(lo.some((v,i)=>hi[i]-v<length))throw new Error('全脑数据某一轴小于 crop 边长，请缩小边长');
  // Preserve the requested cube size at boundaries by shifting it inward.
  const start=center.map((v,i)=>Math.max(lo[i],Math.min(hi[i]-length,Math.floor(v-length/2))));
  return {start,end:start.map(v=>v+length)};
}

function mountSoma(body,viewer) {
  body.innerHTML=`
    <p class="hint">从 SWC 自动列出胞体候选，点击后查看对应图像与标注。</p>
    <form id="soma-form">
      <label for="soma-brain-path">全脑数据目录</label><input id="soma-brain-path" placeholder="/data/brain" required autocomplete="off">
      <label for="soma-brain-mode">全脑数据模式</label><select id="soma-brain-mode"><option value="fnt">FNT · catalog + NRRD 分块</option><option value="slice">TIFF · 二维切片目录</option></select>
      <div id="soma-fnt-options"><label for="soma-channel">通道</label><input id="soma-channel" value="ch00"></div>
      <div id="soma-slice-options" hidden><label for="soma-pattern">文件名正则（Z 索引捕获组）</label><input id="soma-pattern"><label for="soma-extension">切片扩展名</label><input id="soma-extension" value=".tif"></div>
      <label for="soma-length">Crop 立方体边长（体素）</label><input id="soma-length" type="number" min="1" max="300" step="1" value="128" required>
      <label for="soma-swc-path">Neuron SWC 文件或目录</label><input id="soma-swc-path" placeholder="/data/neurons" required autocomplete="off">
      <label class="check"><input id="soma-recursive" type="checkbox">包含子文件夹</label>
      <p class="hint">优先 type=1 节点；没有时列出 parent=-1 根节点候选。SWC 须使用全脑全局体素 XYZ 坐标。靠近边界时向内平移 crop，保持边长。</p>
      <div class="button-row"><button id="soma-scan" class="primary" type="submit">识别胞体候选</button><button id="soma-cancel" class="quiet" type="button" disabled>取消</button></div>
    </form>
    <div id="soma-progress" class="small-status" role="status">输入路径后开始识别，扫描时不加载图像像素。</div>
    <div id="soma-view-status" class="small-status" role="status"></div>
    <label for="soma-search">胞体候选 <span id="soma-count">0</span></label><input id="soma-search" type="search" placeholder="按 neuron 名称或节点 ID 筛选">
    <div id="soma-results" class="soma-results"></div>
    <div class="soma-pages"><button id="soma-prev" class="quiet" type="button">上一页</button><span id="soma-page"></span><button id="soma-next" class="quiet" type="button">下一页</button></div>`;
  const $=id=>body.querySelector('#soma-'+id),pageSize=50;
  $('pattern').value=String.raw`.*?(\d+)\.tif`;
  let generation=0,selection=0,scanning=false,context=null,groups=[],page=0,active=null,pending=null,queued=false;
  function draw() {
    queued=false;const query=$('search').value.trim().toLowerCase();
    let candidates=0;
    const rows=[];
    for(const group of groups)if(group) {
      candidates+=group.candidates.length;
      for(const candidate of group.candidates)rows.push({group,candidate});
      if(group.error)rows.push({group,error:group.error});
    }
    $('count').textContent=String(candidates);
    const filtered=rows.filter(({group,candidate,error})=>(group.entry.name+' '+group.entry.path+' '+(candidate?.node_id??'')+' '+(error||'')).toLowerCase().includes(query));
    const pages=Math.max(1,Math.ceil(filtered.length/pageSize));page=Math.min(page,pages-1);
    $('prev').disabled=page===0;$('next').disabled=page>=pages-1;$('page').textContent=(page+1)+' / '+pages;
    const fragment=document.createDocumentFragment();
    for(const {group,candidate,error} of filtered.slice(page*pageSize,(page+1)*pageSize)) {
      const key=candidate?group.entry.id+':'+candidate.node_id:null;
      const row=document.createElement(candidate?'button':'div');row.className='soma-candidate';
      const name=document.createElement('div');name.className='file-name';name.textContent=group.entry.name;
      const detail=document.createElement('div');detail.className='hint';
      row.append(name,detail);row.title=group.entry.path;
      if(candidate) {
        row.type='button';row.dataset.key=key;row.setAttribute('aria-pressed',String(active===key));row.classList.toggle('active',active===key);
        let problem='';try{centeredCube(candidate.center,context.length,context.info.bounds);}catch(e){problem=e.message;}
        row.disabled=!!problem;
        detail.textContent='节点 '+candidate.node_id+' · '+(candidate.method==='type=1'?'type=1':'根节点候选')+' · XYZ '+candidate.center.map(v=>Number(v.toFixed(3))).join(', ')+(problem?' · '+problem:pending===key?' · 正在加载…':'');
        row.addEventListener('click',()=>open(group.entry,candidate,key));
      } else {detail.textContent=error;row.classList.add('unavailable');}
      fragment.append(row);
    }
    if(!filtered.length){const empty=document.createElement('div');empty.className='list-empty';empty.textContent=scanning?'正在识别…':rows.length?'没有匹配结果':'尚无胞体候选';fragment.append(empty);}
    $('results').replaceChildren(fragment);
  }
  function scheduleDraw() {if(!queued){queued=true;requestAnimationFrame(draw);}}
  function invalidate(text) {
    generation++;selection++;viewer.cancelView();scanning=false;pending=null;context=null;groups=[];active=null;page=0;
    $('scan').disabled=false;$('cancel').disabled=true;$('progress').textContent=text;$('view-status').textContent='';scheduleDraw();
  }
  function mode() {$('fnt-options').hidden=$('brain-mode').value!=='fnt';$('slice-options').hidden=$('brain-mode').value!=='slice';}
  $('form').addEventListener('input',()=>{mode();invalidate('参数已更新，请重新识别候选。');});
  $('cancel').addEventListener('click',()=>invalidate('已取消识别，当前显示保留。'));
  $('search').addEventListener('input',()=>{page=0;draw();});
  $('prev').addEventListener('click',()=>{page--;draw();});$('next').addEventListener('click',()=>{page++;draw();});
  async function open(entry,candidate,key) {
    const scan=generation,ticket=++selection,ctx=context;
    const valid=()=>generation===scan&&selection===ticket;
    pending=key;$('view-status').textContent='正在加载 '+entry.name+' · 节点 '+candidate.node_id;scheduleDraw();
    try {
      const region=centeredCube(candidate.center,ctx.length,ctx.info.bounds);
      const shown=await viewer.showSoma({swc:entry,brain:ctx.brain,region,center:candidate.center},valid);
      if(valid()) {
        if(shown)active=key;
        $('view-status').textContent=shown?'已显示 '+entry.name+' · 节点 '+candidate.node_id+' · '+ctx.length+'³':'显示请求已取消，请重新选择候选。';
      }
    }catch(error){if(valid())$('view-status').textContent='加载失败：'+error.message;}
    finally{if(valid()){pending=null;scheduleDraw();}}
  }
  $('form').addEventListener('submit',async event=>{
    event.preventDefault();invalidate('正在检查全脑数据…');
    const scan=generation,valid=()=>generation===scan;
    const length=Number($('length').value),brainPath=$('brain-path').value,swcPath=$('swc-path').value,recursive=$('recursive').checked;
    const options={mode:$('brain-mode').value,channel:$('channel').value,pattern:$('pattern').value,extension:$('extension').value};
    scanning=true;$('scan').disabled=true;$('cancel').disabled=false;
    try {
      if(!Number.isInteger(length)||length<1||length>300)throw new Error('crop 边长需要为 1–300 的整数体素');
      const [brain]=await viewer.register({kind:'brain',path:brainPath,options});if(!valid())return;
      const info=await viewer.describe(brain,valid);if(!info||!valid())return;
      if(length>info.limits.max_edge||length**3>info.limits.max_voxels)throw new Error('crop 大于当前显示容量，请缩小边长（建议 ≤ '+info.limits.suggested_cube+'）');
      if(info.bounds[0].some((v,i)=>info.bounds[1][i]-v<length))throw new Error('全脑数据某一轴小于 crop 边长，请缩小边长');
      const entries=await viewer.register({kind:'swc',path:swcPath,recursive});if(!valid())return;
      context={brain,info,length};groups=new Array(entries.length);
      let cursor=0,done=0,failed=0,empty=0;
      const progress=()=>{$('progress').textContent='已识别 '+done+' / '+entries.length+' 个 SWC'+(failed?' · 失败 '+failed:'')+(empty?' · 无候选 '+empty:'');};
      progress();
      async function consume() {
        while(valid()&&cursor<entries.length) {
          const index=cursor++,entry=entries[index];
          try {
            const result=await viewer.candidates(entry,'type-first',valid);if(!result||!valid())return;
            groups[index]={entry,candidates:result.candidates,error:result.candidates.length?'':'没有 type=1 或 parent=-1 节点'};
            if(!result.candidates.length)empty++;
          }catch(error){if(!valid())return;groups[index]={entry,candidates:[],error:'读取失败：'+error.message};failed++;}
          done++;progress();scheduleDraw();
        }
      }
      await Promise.all(Array.from({length:Math.min(viewer.concurrency,entries.length)},consume));
      if(valid())$('progress').textContent+=' · 点击候选加载图像与标注';
    }catch(error){if(valid())$('progress').textContent='识别失败：'+error.message;}
    finally{if(valid()){scanning=false;$('scan').disabled=false;$('cancel').disabled=true;scheduleDraw();}}
  });
  draw();
}
