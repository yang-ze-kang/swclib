import {traceColors} from './trace-renderer.js';

export function mountTraceReplay(body,viewer){
  body.innerHTML=`
    <p class="hint">每一步对应一次预测，并显示本次处理后的追踪树。GT 为灰色，已追踪树为青色；候选颜色与概率对应，实线为实际保留线段，长虚线为未推进尾部，淡点线为未采用的前缀。</p>
    <form id="trace-form">
      <label for="trace-path">神经元推理输出目录 / 旧版 block 目录</label><input id="trace-path" required placeholder="/data/run/neuron-id" autocomplete="off">
      <label for="trace-gt">GT SWC（留空使用记录中的路径）</label><input id="trace-gt" autocomplete="off">
      <div class="button-row"><button id="trace-load" type="submit" class="primary">加载追踪记录</button><button id="trace-close" type="button" class="quiet">退出回放</button></div>
    </form>
    <label class="check"><input id="trace-directions" type="checkbox" checked>显示背景方向箭头（父节点 → 子节点）</label>
    <label class="check"><input id="trace-image" type="checkbox">显示原始图像（默认关闭）</label>
    <div id="trace-image-options" hidden>
      <label for="trace-brain">全脑图像目录</label><input id="trace-brain" autocomplete="off">
      <label for="trace-mode">图像模式</label><select id="trace-mode"><option value="fnt">FNT</option><option value="slice">TIFF 切片</option></select>
      <label for="trace-channel">FNT 通道</label><input id="trace-channel" value="ch00">
      <label for="trace-length">图像视野边长（体素）</label><input id="trace-length" type="number" value="128" min="1" max="300">
    </div>
    <div id="trace-status" class="small-status" role="status">尚未加载记录</div>
    <div id="trace-controls" hidden>
      <div class="button-row"><button id="trace-prev" class="quiet">上一步</button><button id="trace-next" class="primary">下一步</button></div>
      <p class="hint">左右方向键：上一步 / 下一步。输入框内保留文字编辑与滑块操作。</p>
      <label for="trace-position">回放位置 <span id="trace-position-label"></span></label><input id="trace-position" type="range" min="0" value="0" step="1">
      <div class="button-row"><button id="trace-play" class="quiet">自动播放</button><select id="trace-speed" aria-label="播放间隔"><option value="1500">1.5 秒 / 步</option><option value="500">0.5 秒 / 步</option><option value="3000">3 秒 / 步</option></select></div>
      <div id="trace-detail" class="hint"></div><div id="trace-candidates"></div>
    </div>`;
  const $=id=>body.querySelector('#trace-'+id);
  const reasons={advanced:'继续推进',stationary:'位移过小',boundary:'到达边界',backward:'反向停止',overlap:'重合回退',node_budget:'节点上限',root_removed:'seed 已移除',overlap_halted:'重合停止'};
  const navigation=viewer.navigation;
  let info=null,index=0,epoch=0,playing=false,timer=null,busy=false,highlight=null;
  function pause(){playing=false;clearTimeout(timer);$('play').textContent='自动播放';}
  function controls(){ $('controls').hidden=!info;$('prev').disabled=busy||!info||index===0;$('next').disabled=busy||!info||index===info.frame_count-1;$('position').disabled=busy;$('play').disabled=busy&&!playing;$('load').disabled=busy; }
  function close(){epoch++;pause();info=null;busy=false;viewer.stopReplay();navigation.release();controls();$('status').textContent='已退出回放';}
  $('close').addEventListener('click',close);
  function imageOptions(){return {enabled:$('image').checked,path:$('brain').value,mode:$('mode').value,channel:$('channel').value,length:Number($('length').value)};}
  async function go(target,first=false){
    if(!info||target<0||target>=info.frame_count)return;navigation.activate();const ticket=++epoch,current=()=>ticket===epoch,session=info;busy=true;controls();$('status').textContent='正在读取追踪步骤…';
    try{
      const frame=await viewer.replayFrame({id:session.id,index:target});if(!current())return;
      if(first||!viewer.replayActive())viewer.beginReplay(session,frame);
      viewer.setReplayDirections($('directions').checked);
      await viewer.showReplayFrame(frame,imageOptions(),current);if(!current())return;
      index=target;$('position').value=index;$('position-label').textContent=(index+1)+' / '+session.frame_count;
      $('status').textContent=frame.mode==='direct'?`Iter ${frame.iteration} · 预测：查看候选与概率`:`Block ${frame.iteration??frame.block_index+1} · 预测 ${frame.step}`;
      const p=frame.prediction;
      $('detail').textContent='Seed XYZ：'+frame.seed_xyz.map(v=>Number(v.toFixed(3))).join(', ')+(p?` · NMS 后 ${p.candidates.length} 条 · 推进上限 ${p.forward_steps} 点`:'')+
        (p?.backtracks?.length?' · 所有出口重合，向上回退 '+p.backtracks.reduce((sum,b)=>sum+b.removed_nodes.length,0)+' 个节点 / '+p.backtracks.reduce((sum,b)=>sum+b.length,0).toFixed(2)+' voxel':'')+
        (frame.outcome?' · '+(frame.outcome.accepted?'拼接已接受':'拼接未接受 / 已回退'):'')+
        (session.gt_path?'':' · 尚未提供 GT SWC')+
        (session.warnings.length?' · '+session.warnings.slice(0,3).join('；'):'');
      $('candidates').replaceChildren();
      if(p&&frame.phase==='prediction')for(const [i,candidate] of p.candidates.entries()){
        const row=document.createElement('div');row.className='trace-candidate';
        row.classList.toggle('trace-error-candidate',highlight?.index===target&&highlight.ids.includes(candidate.index));
        const swatch=document.createElement('i');swatch.style.backgroundColor=traceColors[i%traceColors.length];
        const text=document.createElement('span');text.textContent=`#${candidate.index} · p=${candidate.probability.toFixed(4)} · ${candidate.actual_edges_xyz.length} 段 · ${reasons[candidate.stop_reason]||candidate.stop_reason}`;row.append(swatch,text);$('candidates').append(row);
      }
      if(playing){if(index+1<session.frame_count)timer=setTimeout(()=>go(index+1),Number($('speed').value));else pause();}
    }catch(error){if(current()){pause();$('status').textContent='回放失败：'+error.message;}}
    finally{if(current()){busy=false;controls();}}
  }
  async function openAt({path,gt_path,index:target=0,candidate_indices=[]}){
    navigation.activate();pause();const ticket=++epoch;busy=true;controls();$('status').textContent='正在索引追踪记录…';
    highlight={index:target,ids:candidate_indices};
    try{const result=await viewer.openReplay({path,gt_path:gt_path||null});if(ticket!==epoch)return;
      if(!Number.isInteger(target)||target<0||target>=result.frame_count)throw new Error('错误对应的预测位置已不在记录中，请重新检测');
      info=result;index=target;$('position').max=info.frame_count-1;
      if(info.metadata.image_source)$('brain').value=info.metadata.image_source;
      await go(target,true);
    }catch(error){if(ticket===epoch){$('status').textContent='加载失败：'+error.message;busy=false;controls();}}
  }
  $('form').addEventListener('submit',event=>{
    event.preventDefault();openAt({path:$('path').value,gt_path:$('gt').value});
  });
  $('prev').addEventListener('click',()=>{pause();go(index-1);});$('next').addEventListener('click',()=>{pause();go(index+1);});
  $('position').addEventListener('input',()=>{pause();go(Number($('position').value));});
  $('directions').addEventListener('change',()=>viewer.setReplayDirections($('directions').checked));
  $('play').addEventListener('click',()=>{if(playing){pause();return;}playing=true;$('play').textContent='暂停';go(index===info.frame_count-1?0:index+1);});
  for(const id of ['image','brain','mode','channel','length'])$(id).addEventListener('change',()=>{pause();$('image-options').hidden=!$('image').checked;if(info)go(index);});
  navigation.setStep(delta=>{if(!info||busy)return false;pause();go(index+delta);return true;});
  navigation.onDeactivate(()=>{epoch++;pause();busy=false;controls();});
  controls();
  return {openPrediction:async options=>{
    $('path').value=options.path;$('gt').value=options.gt_path||'';
    body.closest('details').open=true;
    await openAt(options);
    $('status').scrollIntoView({block:'nearest'});
  }};
}
