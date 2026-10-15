import {Renderer} from './renderer.js';
import {mountPlugins} from './plugins.js';

const $ = id => document.getElementById(id);
const entries = [], swcs = [], images = [], brains = [], entryById = new Map();
let renderer, worker, token, crop=null, revision=0, activeImage=null, imageEpoch=0, io=Promise.resolve(), toastTimer;
let activeBrain=null, brainEpoch=0;
let pickSequence=0, pendingPick=null;
let geometrySequence=0, lastGeometry=0, fitAfterBuild=false, nextSlot=0;
let somaViewEpoch=0;
function cancelSomaView() {somaViewEpoch++;}
const swcQueue=[];
const visibleNeuronListeners=new Set();
let visibleNeuronSignature='';
let swcConcurrency=1, swcActive=0, swcDone=0, swcTotal=0;

function message(text, error=true) {
  clearTimeout(toastTimer);$('toast').textContent=text;$('toast').hidden=false;$('toast').classList.toggle('success',!error);
  $('status-text').textContent=text;
  toastTimer=setTimeout(()=>{$('toast').hidden=true;},error?16000:5000);
}
function coordinates(id, integer=false) {
  const values=$(id).value.trim().split(/[,，\s]+/).map(Number);
  if(values.length!==3||values.some(v=>!Number.isFinite(v))||(integer&&values.some(v=>!Number.isInteger(v))))throw new Error('请输入三个有效的 XYZ '+(integer?'整数':'数值')+'，用逗号分隔');
  return values;
}
function regionInputs(start,end,integer=false) {
  const a=coordinates(start,integer),b=coordinates(end,integer);
  if(a.some((v,i)=>v>=b[i]))throw new Error('每个轴均需要 start < end');
  return {start:a,end:b};
}
async function api(path, body, binary=false) {
  const response=await fetch('/api/'+path,{method:'POST',headers:{'Content-Type':'application/json','X-Viewer-Token':token},body:JSON.stringify(body)});
  if(!response.ok) {const data=await response.json();const error=new Error(data.error||'请求失败');error.status=response.status;error.code=data.code;error.path=data.path;throw error;}
  return binary?response.arrayBuffer():response.json();
}
async function retryRead(path,body,binary,valid) {
  // A busy tab/reader is backpressure, not a failed source. Skip stale jobs.
  while(valid()) {
    try{return await api(path,body,binary);}catch(error){if(error.status!==409)throw error;await new Promise(resolve=>setTimeout(resolve,250+Math.random()*100));}
  }
  return null;
}
function swcProgress() {
  $('swc-load-status').textContent='SWC 并行读取 · '+swcConcurrency+' 个进程'+
    (swcTotal?' · 已处理 '+swcDone+' / '+swcTotal:'')+(swcActive?' · 读取中 '+swcActive:'');
}
function pumpSWCs() {
  while(swcActive<swcConcurrency&&swcQueue.length) {
    const task=swcQueue.shift();swcActive++;
    retryRead(task.path,task.body,task.binary,task.valid).then(task.resolve,task.reject).finally(()=>{
      swcActive--;swcDone++;pumpSWCs();
    });
  }
  swcProgress();
}
function read(path,body,binary,valid=()=>true) {
  if(path==='swc'||path==='somas') {
    if(!swcActive&&!swcQueue.length)swcDone=swcTotal=0;
    swcTotal++;
    return new Promise((resolve,reject)=>{swcQueue.push({path,body,binary,valid,resolve,reject});pumpSWCs();});
  }
  const job=io.then(()=>retryRead(path,body,binary,valid));
  io=job.catch(()=>{});return job;
}
function unpack(buffer) {
  const length=new DataView(buffer).getUint32(0,true);
  return {meta:JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,4,length))),offset:length+4,buffer};
}
function neuronColor(index) {
  // Golden-angle hues remain stable across hide/show, crops, and mode switches.
  const h=(index*.61803398875+.43)%1,s=.62,l=.67;
  const channel=n=>{const k=(n+h*12)%12;return Math.round(255*(l-s*Math.min(l,1-l)*Math.max(-1,Math.min(k-3,9-k,1))));};
  return [channel(0),channel(8),channel(4)];
}
function palette() {renderer.setPalette(swcs);updateStats();}
function visibleNeurons() {return swcs.filter(e=>e.visible&&e.loadedRevision===revision&&!e.removed&&renderer.geometrySlots?.has(e.slot)).map(e=>e.id);}
function updateStats() {
  if(pendingPick&&pendingPick.serial!==renderer.stateSerial)closeTargetMenu();
  const count=swcs.reduce((n,e)=>n+(e.visible&&e.loadedRevision===revision?e.count||0:0),0);
  $('node-stats').textContent=count.toLocaleString()+' 个节点';
  $('draw-stats').textContent=renderer.drawCalls+' 次绘制';
  $('empty-state').hidden=count>0||!!renderer.volume;
  $('center-status').textContent='中心：'+renderer.center.map(v=>Number(v.toFixed(3))).join(', ');
  const signature=JSON.stringify(visibleNeurons());
  if(signature!==visibleNeuronSignature){visibleNeuronSignature=signature;for(const notify of visibleNeuronListeners)notify();}
}
function visibleBounds() {return swcs.filter(e=>e.visible&&e.loadedRevision===revision).map(e=>e.bounds);}
function fit() {renderer.fit(visibleBounds());}
function closeTargetMenu() {pickSequence++;pendingPick=null;$('target-menu').hidden=true;}
function positionTargetMenu(x,y) {
  const menu=$('target-menu'),rect=$('scene').getBoundingClientRect();
  menu.style.left=Math.max(6,Math.min(x-rect.left,rect.width-menu.offsetWidth-6))+'px';
  menu.style.top=Math.max(6,Math.min(y-rect.top,rect.height-menu.offsetHeight-6))+'px';
}
function finishPick(data) {
  const pick=pendingPick;
  if(!pick||data.requestId!==pick.id)return;
  if(data.revision!==revision||pick.serial!==renderer.stateSerial){closeTargetMenu();return;}
  try {
    pick.target=data.target||renderer.pickVolume(pick.ray);
    $('target-info').textContent=pick.target?(data.target?'SWC':'图像')+' · XYZ '+pick.target.map(v=>Number(v.toFixed(3))).join(', '):'未命中可见 SWC 或图像信号';
    $('go-to-target').disabled=!pick.target;positionTargetMenu(pick.x,pick.y);
  }catch(error){$('target-info').textContent=error.message;$('go-to-target').disabled=true;}
}
function showTargetMenu(x,y) {
  const id=++pickSequence,ray=renderer.screenRay(x,y),s=renderer.settings;
  pendingPick={id,serial:renderer.stateSerial,ray,x,y,target:null};
  $('target-menu').hidden=false;$('target-info').textContent='正在定位…';$('go-to-target').disabled=true;positionTargetMenu(x,y);
  const visible=swcs.filter(e=>e.visible&&e.loadedRevision===revision&&renderer.geometrySlots?.has(e.slot)).map(e=>e.slot);
  if(visible.length&&(s.points||s.lines)&&s.swcOpacity>0)
    worker.postMessage({type:'pick',requestId:id,revision,ray,visible,points:s.points,lines:s.lines,pointSize:s.pointSize});
  else finishPick({requestId:id,revision,target:null});
}
function updateRow(entry) {
  if(entry.removed||entry.kind==='brain')return;
  const selected=entry.kind==='swc'?entry.visible:entry.id===activeImage;
  entry.row.classList.toggle('visible',selected);entry.input.checked=selected;
  entry.detail.textContent=entry.loading?'正在读取…':entry.kind==='swc'?(entry.count===undefined?'未读取':entry.count.toLocaleString()+' nodes'):
    entry.kind==='brain-region'?'ROI · '+entry.options.region.start.map((v,i)=>entry.options.region.end[i]-v).join(' × '):(selected&&renderer.volume?'已显示':'按需加载');
}
function updateLists() {
  $('entry-count').textContent=entries.length+' 个数据源';$('swc-count').textContent=swcs.length;$('image-count').textContent=images.length;
  $('remove-brain').disabled=!activeBrain;
  for(const [kind,list] of [['swc',swcs],['image',images]]) {
    const selected=list.filter(e=>e.selected).length,all=$('select-all-'+kind),button=$('remove-selected-'+kind);
    all.checked=list.length>0&&selected===list.length;all.indeterminate=selected>0&&selected<list.length;
    button.textContent='移除所选 ('+selected+')';button.disabled=selected===0;
    if(!list.length&&!$(kind+'-list').querySelector('.list-empty')) {
      const empty=document.createElement('div');empty.className='list-empty';empty.textContent=kind==='swc'?'还没有添加 SWC':'还没有添加图像';$(kind+'-list').append(empty);
    }
  }
}
async function removeEntries(list) {
  cancelSomaView();
  const candidates=list.filter(e=>!e.removed&&!e.removing);
  if(!candidates.length)return;
  for(const e of candidates)e.removing=true;
  try {
    // Small requests also support catalogues exceeding the HTTP body-size cap.
    for(let offset=0;offset<candidates.length;offset+=500) {
      const batch=candidates.slice(offset,offset+500);
      await api('remove',{ids:batch.map(e=>e.id)});
      if(batch.some(e=>e.id===activeImage))hideImage();
      if(batch.some(e=>e.id===activeBrain)) {
        activeBrain=null;brainEpoch++;$('brain-source').value='';$('region-options').hidden=true;
        $('brain-info').textContent='选择全脑数据源后设置区域。';
      }
      const ids=new Set(batch.map(e=>e.id)),slots=[];
      for(const e of batch) {e.removed=true;e.visible=false;entryById.delete(e.id);e.row.remove();if(e.kind==='swc')slots.push(e.slot);}
      for(const collection of [entries,swcs,images,brains])for(let i=collection.length-1;i>=0;i--)if(ids.has(collection[i].id))collection.splice(i,1);
      if(slots.length){worker.postMessage({type:'remove',slots});palette();buildGeometry();}
      updateLists();updateStats();
    }
    message('已从列表移除 '+candidates.length+' 项，源文件未修改。',false);
  }catch(error){message(error.message);}finally{for(const e of candidates)e.removing=false;updateLists();}
}
function addEntries(items) {
  const fresh=[];
  for(const item of items) {
    if(entryById.has(item.id))continue;
    const e={...item,visible:false};entries.push(e);entryById.set(e.id,e);fresh.push(e);
    if(e.kind==='brain') {
      brains.push(e);e.row=document.createElement('option');e.row.value=e.id;e.row.textContent=e.name;e.row.title=e.path;
      $('brain-source').append(e.row);continue;
    }
    const list=e.kind==='swc'?swcs:images;
    e.slot=e.kind==='swc'?nextSlot++:0;e.color=neuronColor(e.slot);list.push(e);
    const row=document.createElement('div');row.className='file-row';row.title=e.name+'\n'+e.path;e.row=row;
    const selection=document.createElement('input');selection.type='checkbox';selection.className='source-selection';selection.setAttribute('aria-label','选择 '+e.name+' 以批量移除');e.selection=selection;
    selection.addEventListener('change',()=>{e.selected=selection.checked;updateLists();});row.append(selection);
    const display=document.createElement('label');display.className='file-display';
    const input=document.createElement('input');input.type=e.kind==='swc'?'checkbox':'radio';input.name=e.kind==='swc'?'swc-'+e.id:'active-image';input.className='visibility';input.setAttribute('aria-label','显示 '+e.name);e.input=input;
    const meta=document.createElement('div');meta.className='file-meta';
    const name=document.createElement('div');name.className='file-name';name.textContent=e.name;
    const detail=document.createElement('div');detail.className='file-detail';e.detail=detail;
    meta.append(name,detail);display.append(input,meta);row.append(display);
    if(e.kind==='swc') {const dot=document.createElement('span');dot.className='file-color';dot.style.setProperty('--neuron-color','rgb('+e.color.join(',')+')');row.append(dot);}
    const remove=document.createElement('button');remove.className='remove-entry';remove.textContent='×';remove.title='从列表移除（源文件不变）';remove.setAttribute('aria-label','移除 '+e.name);remove.addEventListener('click',()=>removeEntries([e]));row.append(remove);
    input.addEventListener('change',()=>{
      if(e.kind==='swc') {cancelSomaView();e.visible=input.checked;updateRow(e);palette();if(e.visible)loadSWCs([e]);}
      else selectImage(e).catch(error=>message(error.message));
    });
    const parent=$(e.kind==='swc'?'swc-list':'image-list');parent.querySelector('.list-empty')?.remove();parent.append(row);updateRow(e);
  }
  updateLists();
  return fresh;
}
function buildGeometry(shouldFit=false) {
  fitAfterBuild=fitAfterBuild||shouldFit;
  worker.postMessage({type:'build',revision,sequence:++geometrySequence});
}
async function loadSWCs(list,shouldFit=false) {
  const current=revision,requestedCrop=crop;
  let changed=false;
  await Promise.all([...list].map(async e=>{
    if(current!==revision||e.removed||!e.visible||e.loadedRevision===current||e.loadingRevision===current)return;
    e.loading=true;e.loadingRevision=current;updateRow(e);$('status-text').textContent='读取 SWC · '+e.name;
    try {
      const buffer=await read('swc',{id:e.id,crop:requestedCrop},true,()=>current===revision&&e.visible&&!e.removed);
      if(!buffer||current!==revision||e.removed)return;
      const {meta,offset}=unpack(buffer);
      Object.assign(e,{count:meta.count,bounds:meta.bounds,loadedRevision:current});
      worker.postMessage({type:'add',slot:e.slot,count:meta.count,edgeCount:meta.edge_count,offset,buffer},[buffer]);changed=true;
    }catch(error){if(current===revision&&!e.removed){e.visible=false;message(e.name+'：'+error.message);}}
    finally{if(e.loadingRevision===current){e.loading=false;e.loadingRevision=undefined;updateRow(e);}}
  }));
  if(current!==revision)return;
  palette();if(changed)buildGeometry(shouldFit);else if(shouldFit)fit();
  if(!$('toast').hidden)return;
  $('status-text').textContent='就绪';
}
function resetCrop(next) {
  crop=next;revision++;
  renderer.cropStart=crop?crop.start:[0,0,0];
  worker.postMessage({type:'clear'});renderer.setGeometry({vertices:new ArrayBuffer(0),edges:new ArrayBuffer(0),count:0,edgeCount:0});
  for(const e of swcs){e.loadedRevision=undefined;e.loadingRevision=undefined;e.loading=false;updateRow(e);}
  $('crop-status').textContent=crop?'已裁剪 · '+crop.start.join(', ')+' → '+crop.end.join(', '):'未启用裁剪 · 全局坐标';
  $('coordinate-status').textContent=crop?'XYZ · 局部坐标':'XYZ · 全局坐标';
}
function applyCrop(next) {
  cancelSomaView();resetCrop(next);palette();loadSWCs(swcs,true);syncSlice();
}
function limitsText(limits) {
  $('memory-limit').textContent='图像最大 300 × 300 × 300。当前可用 '+(limits.max_voxels/1e6).toFixed(1)+' M 体素 · 单轴 ≤ '+limits.max_edge+' · 建议立方体 ≤ '+limits.suggested_cube+'³。内存不足时进一步收紧。';
}
function hideImage() {
  cancelSomaView();
  imageEpoch++;activeImage=null;renderer.clearVolume();
  for(const e of images){e.loading=false;updateRow(e);}
  $('volume-info').textContent='选择图像以查看尺寸和显示上限。';syncSlice();updateStats();
  $('block-options').hidden=false;
}
async function selectBrain(e) {
  const epoch=++brainEpoch;activeBrain=e?.id||null;$('brain-source').value=activeBrain||'';
  $('region-options').hidden=!e;$('remove-brain').disabled=!e;
  if(!e){$('brain-info').textContent='选择全脑数据源后设置区域。';return;}
  $('brain-info').textContent='正在检查全脑数据范围…';$('load-region').disabled=true;
  try {
    const info=await read('info',{id:e.id,kind:'brain',client:renderer.capabilities()},false,()=>epoch===brainEpoch&&!e.removed);
    if(!info||epoch!==brainEpoch||e.removed)return;
    e.info=info;$('brain-info').textContent=e.name+' · '+info.shape.join(' × ')+' · XYZ';limitsText(info.limits);
    const start=crop?crop.start:info.bounds[0];
    const end=crop?crop.end:start.map((v,i)=>Math.min(info.bounds[1][i],v+Math.min(128,info.limits.suggested_cube)));
    $('brain-start').value=start.join(', ');$('brain-end').value=end.join(', ');
  }catch(error){if(epoch===brainEpoch){$('brain-info').textContent=error.message;message(error.message);}}
  finally{if(epoch===brainEpoch)$('load-region').disabled=false;}
}
async function selectImage(e) {
  cancelSomaView();
  const epoch=++imageEpoch;activeImage=e.id;renderer.clearVolume();
  for(const image of images){image.loading=false;updateRow(image);}
  e.loading=true;updateRow(e);updateStats();
  $('block-options').hidden=e.kind!=='image';
  $('volume-info').textContent='读取图像元数据…';$('status-text').textContent='检查图像 · '+e.name;
  try {
    const info=await read('info',{id:e.id,kind:e.kind,client:renderer.capabilities()},false,()=>epoch===imageEpoch);
    if(!info||epoch!==imageEpoch)return;
    e.info=info;$('volume-info').textContent=info.shape.join(' × ')+' · XYZ · '+info.dtype;limitsText(info.limits);
    if(e.kind==='image')$('image-origin').value=(e.origin||[0,0,0]).join(', ');
    await loadImage(e,epoch);
  } catch(error) {if(epoch===imageEpoch){hideImage();message(error.message);}}
  finally {if(epoch===imageEpoch){e.loading=false;updateRow(e);}}
}
async function loadImage(e,epoch) {
  const body={id:e.id,kind:e.kind,client:renderer.capabilities()};
  if(e.kind==='image'){body.origin=coordinates('image-origin');e.origin=body.origin;}
  renderer.clearVolume();e.loading=true;updateRow(e);$('status-text').textContent='读取图像 · '+e.name;
  try {
    const buffer=await read('volume',body,true,()=>epoch===imageEpoch);
    if(!buffer||epoch!==imageEpoch)return;
    const {meta,offset}=unpack(buffer);
    renderer.setVolume(meta,new Float32Array(buffer,offset));limitsText(meta.limits);
    $('volume-info').textContent=meta.shape.join(' × ')+' · XYZ · 原始灰度 '+meta.intensity_range.map(v=>Number(v.toPrecision(6))).join(' – ');
    setWindow(meta.window[0],meta.window[1]);syncSlice();fit();updateStats();
    $('status-text').textContent='已显示 '+e.name+' · '+(buffer.byteLength/1048576).toFixed(1)+' MiB';
  }finally {if(epoch===imageEpoch){e.loading=false;updateRow(e);}}
}
function syncSlice() {
  const axis={xy:2,xz:1,yz:0}[renderer.mode],volume=renderer.volume;
  $('slice-control').hidden=axis===undefined||!volume;
  if(axis!==undefined&&volume){$('slice-index').max=volume.shape[axis]-1;$('slice-index').value=renderer.slices[axis];$('slice-value').textContent=renderer.slices[axis]+' / '+(volume.shape[axis]-1);}
}
function setView(mode) {
  renderer.setView(mode);$('render-mode').value=mode;
  for(const button of $('view-buttons').children)button.classList.toggle('active',button.dataset.view===(mode==='volume'?'mip':mode));
  $('view-name').textContent={mip:'三维 · 最大强度投影',volume:'三维 · 体渲染',xy:'XY · 正交切片',xz:'XZ · 正交切片',yz:'YZ · 正交切片'}[mode];syncSlice();
}
function setWindow(black,white,reset=false) {
  renderer.settings.black=black;renderer.settings.white=white;$('black').value=black;$('white').value=white;
  if(reset)for(const [id,value] of [['brightness',0],['gamma',1]]){$(id).value=value;renderer.settings[id]=value;$(id+'-value').textContent=value.toFixed(2);}
  updateWindowLabels();renderer.invalidate();
}
function updateWindowLabels() {
  const [lo,hi]=renderer.volume?.intensity_range||[0,1];
  for(const id of ['black','white'])$(id+'-value').textContent=Number((lo+(hi-lo)*renderer.settings[id]).toPrecision(5));
}
function bindRange(id,key,format) {
  $(id).addEventListener('input',()=>{renderer.settings[key]=Number($(id).value);$(id+'-value').textContent=format(renderer.settings[key]);renderer.invalidate();});
}

async function showSoma({swc,brain,region,center},valid) {
  const epoch=++somaViewEpoch;
  const current=()=>epoch===somaViewEpoch&&valid()&&!swc.removed&&!brain.removed;
  // Stage both reads first. A bad image or a newer selection leaves the old
  // scene intact, and never pairs one neuron's SWC with another one's image.
  const result=await read('region',{id:brain.id,...region,client:renderer.capabilities()},false,current);
  if(!result||!current())return false;
  addEntries([result.entry]);const block=entryById.get(result.entry.id);
  const [swcBuffer,imageBuffer]=await Promise.all([
    read('swc',{id:swc.id,crop:region},true,current),
    read('volume',{id:block.id,kind:block.kind,client:renderer.capabilities()},true,current)
  ]);
  if(!swcBuffer||!imageBuffer||!current()||block.removed)return false;
  const morphology=unpack(swcBuffer),volume=unpack(imageBuffer);
  renderer.setVolume(volume.meta,new Float32Array(imageBuffer,volume.offset));
  imageEpoch++;activeImage=block.id;
  for(const e of images){e.loading=false;updateRow(e);}
  for(const e of swcs)e.visible=e===swc;
  resetCrop(region);
  $('crop-start').value=region.start.join(', ');$('crop-end').value=region.end.join(', ');
  Object.assign(swc,{count:morphology.meta.count,bounds:morphology.meta.bounds,loadedRevision:revision});updateRow(swc);
  worker.postMessage({type:'add',slot:swc.slot,count:swc.count,edgeCount:morphology.meta.edge_count,offset:morphology.offset,buffer:swcBuffer},[swcBuffer]);
  fitAfterBuild=false;palette();buildGeometry();
  $('block-options').hidden=true;
  $('volume-info').textContent=block.name+' · '+volume.meta.shape.join(' × ')+' · XYZ';limitsText(volume.meta.limits);
  setWindow(...volume.meta.window);fit();renderer.setCenter(center.map((v,i)=>v-region.start[i]));syncSlice();
  $('view-center').value=renderer.center.map(v=>Number(v.toFixed(3))).join(', ');
  $('status-text').textContent='查看胞体 · '+swc.name;
  return true;
}
function handlers() {
  renderer.onTargetMenu=showTargetMenu;renderer.onNavigate=closeTargetMenu;
  document.addEventListener('pointerdown',event=>{if(!$('target-menu').contains(event.target))closeTargetMenu();});
  document.addEventListener('keydown',event=>{if(event.key==='Escape')closeTargetMenu();});
  $('go-to-target').addEventListener('click',()=>{
    const pick=pendingPick;if(!pick?.target||pick.serial!==renderer.stateSerial){closeTargetMenu();return;}
    const target=pick.target;closeTargetMenu();renderer.setCenter(target);syncSlice();
    $('view-center').value=target.map(v=>Number(v.toFixed(3))).join(', ');
  });
  $('overall-center').addEventListener('click',()=>{closeTargetMenu();renderer.recenter(visibleBounds());syncSlice();});
  $('center-form').addEventListener('submit',event=>{event.preventDefault();try{renderer.setCenter(coordinates('view-center'));syncSlice();}catch(error){message(error.message);}});
  $('read-center').addEventListener('click',()=>{$('view-center').value=renderer.center.map(v=>Number(v.toFixed(3))).join(', ');});
  for(const kind of ['swc','image','plugins']) {
    $('tab-'+kind).addEventListener('click',()=>{for(const k of ['swc','image','plugins']){$('tab-'+k).classList.toggle('active',kind===k);$('tab-'+k).setAttribute('aria-selected',String(kind===k));$('panel-'+k).hidden=kind!==k;}});
  }
  for(const kind of ['swc','image']) {
    const list=kind==='swc'?swcs:images;
    $('select-all-'+kind).addEventListener('change',()=>{for(const e of list){e.selected=$('select-all-'+kind).checked;e.selection.checked=e.selected;}updateLists();});
    $('remove-selected-'+kind).addEventListener('click',()=>removeEntries(list.filter(e=>e.selected)));
    $(kind+'-form').addEventListener('submit',async event=>{
      event.preventDefault();cancelSomaView();const button=event.target.querySelector('button');button.disabled=true;
      try {
        const result=await api('add',{kind,path:$(kind+'-path').value,recursive:$(kind+'-recursive').checked});
        const added=addEntries(result.entries);
        if(kind==='swc'){for(const e of added){e.visible=true;updateRow(e);}palette();await loadSWCs(added,true);}
        else message('已登记 '+added.length+' 幅图像，选择后才加载像素。',false);
      }catch(error){message(error.message);}finally{button.disabled=false;}
    });
  }
  $('brain-mode').addEventListener('change',()=>{$('fnt-options').hidden=$('brain-mode').value!=='fnt';$('slice-options').hidden=$('brain-mode').value!=='slice';});
  $('brain-form').addEventListener('submit',async event=>{
    event.preventDefault();const button=event.target.querySelector('button');button.disabled=true;
    try {const result=await api('add',{kind:'brain',path:$('brain-path').value,options:{mode:$('brain-mode').value,channel:$('brain-channel').value,pattern:$('brain-pattern').value,extension:$('brain-extension').value}});addEntries(result.entries);await selectBrain(entryById.get(result.entries[0].id));}catch(error){message(error.message);}finally{button.disabled=false;}
  });
  $('brain-source').addEventListener('change',()=>selectBrain(entryById.get($('brain-source').value)));
  $('remove-brain').addEventListener('click',()=>{const source=entryById.get(activeBrain);if(source)removeEntries([source]);});
  $('show-all').addEventListener('click',()=>{cancelSomaView();for(const e of swcs){e.visible=true;updateRow(e);}palette();loadSWCs(swcs,true);});
  $('hide-all').addEventListener('click',()=>{cancelSomaView();for(const e of swcs){e.visible=false;updateRow(e);}palette();});
  $('fit').addEventListener('click',fit);
  document.addEventListener('keydown',e=>{if(e.key.toLowerCase()==='f'&&!['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))fit();});
  $('color-mode').addEventListener('change',()=>{renderer.settings.colorMode=Number($('color-mode').value);$('legend').hidden=renderer.settings.colorMode===1;renderer.invalidate();});
  bindRange('point-size','pointSize',v=>v+' px');bindRange('swc-opacity','swcOpacity',v=>Math.round(v*100)+'%');
  for(const name of ['points','lines'])$('show-'+name).addEventListener('change',()=>{renderer.settings[name]=$('show-'+name).checked;renderer.invalidate();});
  $('crop-form').addEventListener('submit',e=>{e.preventDefault();try{applyCrop(regionInputs('crop-start','crop-end'));}catch(error){message(error.message);}});
  $('clear-crop').addEventListener('click',()=>applyCrop(null));$('hide-image').addEventListener('click',hideImage);
  $('load-region').addEventListener('click',async()=>{
    const e=entryById.get(activeBrain),epoch=brainEpoch,selection=imageEpoch;
    if(!e){message('请先选择全脑数据源');return;}
    cancelSomaView();
    $('load-region').disabled=true;
    try {
      const result=await read('region',{id:e.id,...regionInputs('brain-start','brain-end',true),client:renderer.capabilities()},false,()=>!e.removed&&epoch===brainEpoch);
      if(!result)return;
      addEntries([result.entry]);
      if(epoch===brainEpoch&&selection===imageEpoch)await selectImage(entryById.get(result.entry.id));
    }catch(error){if(epoch===brainEpoch)message(error.message);}
    finally{if(epoch===brainEpoch)$('load-region').disabled=false;}
  });
  $('use-swc-crop').addEventListener('click',()=>{try{const value=crop||regionInputs('crop-start','crop-end',true);$('brain-start').value=value.start.join(', ');$('brain-end').value=value.end.join(', ');}catch(error){message(error.message);}});
  function origin(align=false) {try{cancelSomaView();const e=images.find(e=>e.id===activeImage);if(!e||e.kind!=='image')throw new Error('请先选择小块图像');if(align)$('image-origin').value=(crop?crop.start:[0,0,0]).join(', ');e.origin=coordinates('image-origin');if(renderer.volume){renderer.volume.origin=e.origin;fit();}}catch(error){message(error.message);}}
  $('apply-origin').addEventListener('click',()=>origin());$('align-origin').addEventListener('click',()=>origin(true));
  for(const button of $('view-buttons').children)button.addEventListener('click',()=>setView(button.dataset.view));
  $('render-mode').addEventListener('change',()=>setView($('render-mode').value));
  $('slice-index').addEventListener('input',()=>{const axis={xy:2,xz:1,yz:0}[renderer.mode];renderer.slices[axis]=Number($('slice-index').value);syncSlice();renderer.invalidate();});
  $('slab').addEventListener('change',()=>{const value=Number($('slab').value);if(Number.isFinite(value)&&value>0){renderer.settings.slab=value;renderer.invalidate();}});
  for(const id of ['black','white'])$(id).addEventListener('input',()=>{let b=Number($('black').value),w=Number($('white').value);if(w<=b){if(id==='black')b=Math.max(0,w-.001);else w=Math.min(1,b+.001);}setWindow(b,w);});
  bindRange('brightness','brightness',v=>v.toFixed(2));bindRange('gamma','gamma',v=>v.toFixed(2));bindRange('image-opacity','opacity',v=>Math.round(v*100)+'%');
  $('auto-window').addEventListener('click',()=>setWindow(...(renderer.volume?.window||[0,1]),true));$('reset-window').addEventListener('click',()=>setWindow(0,1,true));
}

async function start() {
  try {
    renderer=new Renderer($('scene'),updateStats,message);
    renderer.hemisphereBounds=()=>renderer.sceneBounds(visibleBounds());
    renderer.hemisphereLabels=[$('hemisphere-left'),$('hemisphere-right')];
    worker=new Worker('/geometry-worker.js');
    worker.onerror=event=>message('几何数据处理失败：'+event.message);
    worker.onmessage=({data})=>{if(data.type==='pick'){finishPick(data);return;}if(data.revision!==revision||data.sequence!==geometrySequence||data.sequence<lastGeometry)return;lastGeometry=data.sequence;renderer.setGeometry(data);palette();if(fitAfterBuild){fitAfterBuild=false;fit();}};
    const response=await fetch('/api/config');if(!response.ok)throw new Error('无法读取服务配置');const config=await response.json();token=config.token;
    swcConcurrency=Math.max(1,Math.min(8,config.swc_workers||1));swcProgress();
    $('version').textContent='v'+config.version;limitsText(config.limits);handlers();
    mountPlugins($('plugin-list'),{
      concurrency:swcConcurrency,cancelView:cancelSomaView,showSoma,
      setPlane:plane=>{renderer.hemispherePlane=plane;renderer.invalidate();},
      planeBounds:()=>renderer.sceneBounds(visibleBounds())?.map(point=>point.map((v,i)=>v+renderer.cropStart[i])),
      visibleNeurons,onVisibleNeuronsChanged:callback=>visibleNeuronListeners.add(callback),
      countHemisphere:body=>api('hemisphere/count',body),
      optimizeHemisphere:body=>api('hemisphere/optimize',body),
      exportHemisphere:body=>api('hemisphere/export',body),
      register:async body=>{const result=await api('add',body);addEntries(result.entries);return result.entries.map(e=>entryById.get(e.id));},
      describe:(entry,valid)=>read('info',{id:entry.id,kind:entry.kind,client:renderer.capabilities()},false,valid),
      candidates:(entry,rule,valid)=>read('somas',{id:entry.id,rule},false,valid)
    });
    $('readonly-status').title='服务端文件写入保护：'+config.filesystem_protection;
    addEntries(config.entries);for(const e of swcs){e.visible=true;updateRow(e);}palette();await loadSWCs(swcs,true);
    // Read-only diagnostics for reproducible performance tests, no render loop when idle.
    window.swcStudio={renderer,get entries(){return entries;},get revision(){return revision;},fit};
  }catch(error){message(error.message);$('status-text').textContent='初始化失败';}
}
start();
