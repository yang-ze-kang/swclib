const names={wrong_connection:'接错路径',extra_branch:'多出分支',terminal_overextension:'末端过度延伸',premature_termination:'提前终止',missed_branch:'漏分支',cross_neuron_merge:'跨神经元错接',disconnection:'断裂',missing_subtree:'整支／子树缺失',false_branch_candidate:'假分支候选',false_fragment_candidate:'假片段候选',unsupported_path_candidate:'无 GT 支持路径候选',ambiguous_connection:'连接待定',ambiguous_missing_branch:'缺失分支待定',uncertain_missing_continuation:'缺失后续原因待定',geometric_deviation:'几何偏差',pure_reversal:'纯回溯',duplicate_tracing:'重复追踪',short_geometry:'短几何抖动'};
const xyz=point=>point.map(v=>Number(v.toFixed(3))).join(', ');
export function mountErrorDetection(body,viewer){
  body.innerHTML=`
    <p class="hint">比较同一坐标系的预测与 GT SWC，列出错误并定位。原始 Fiber TP/FP/FN 保持独立；已确认错误、候选及几何诊断分别列出。短距离拓扑错接不按长度过滤。</p>
    <form id="error-form">
      <label for="error-mode">输入方式</label><select id="error-mode"><option value="swc">Pred SWC + GT SWC</option><option value="trace">追踪过程记录 + GT</option></select>
      <div id="error-swc-input"><label for="error-pred">Pred SWC 文件</label><input id="error-pred" required autocomplete="off" placeholder="/data/pred.swc"></div>
      <div id="error-trace-input" hidden><label for="error-trace-path">trace_steps.jsonl 文件或目录</label><input id="error-trace-path" autocomplete="off" placeholder="/data/run/trace_steps.jsonl"><p class="hint">需要同目录的 trace_initial.npz。检测最后保留的树，追溯生成错误边的预测；回退删除的临时错误不计入。</p></div>
      <label id="error-gt-label" for="error-gt">GT SWC 文件</label><input id="error-gt" required autocomplete="off" placeholder="/data/gt.swc">
      <details><summary>检测参数</summary>
        <label for="error-distance">空间匹配容差</label><input id="error-distance" type="number" min="0.001" step="any" value="3" required>
        <label for="error-coverage">近似覆盖阈值 τ</label><input id="error-coverage" type="number" min="0.001" max="1" step="any" value="0.9" required>
        <label for="error-short">短几何段参考长度（留空为 2r）</label><input id="error-short" type="number" min="0.001" step="any" placeholder="自动：2 × 容差">
        <label for="error-min-length">持续失配 / 缺失确认弧长</label><input id="error-min-length" type="number" min="0.001" step="any" value="3" required>
        <label for="error-step">重采样步长</label><input id="error-step" type="number" min="0.001" step="any" value="0.5" required>
      </details>
      <div class="button-row"><button id="error-detect" class="primary" type="submit">检测错误</button><button id="error-close" class="quiet" type="button">退出查看</button></div>
    </form>
    <div id="error-status" class="small-status" role="status">尚未检测</div>
    <div id="error-controls" hidden>
      <label class="check"><input id="error-first" type="checkbox" checked>只显示首次 FJ（保留全部独立 break）</label>
      <div id="error-counts" class="error-counts"></div>
      <label for="error-category">判定状态</label><select id="error-category"><option value="all">全部（含候选、诊断）</option><option value="confirmed">已确认错误</option><option value="candidate">候选 / 待定</option><option value="diagnostic">几何偏差 / 其他诊断</option></select>
      <label for="error-type">错误类型筛选</label><select id="error-type"><option value="all">全部错误类型</option>${Object.entries(names).map(([key,name])=>`<option value="${key}">${name}</option>`).join('')}</select>
      <div class="button-row"><button id="error-prev" type="button" class="quiet">上一处 ←</button><button id="error-next" type="button" class="primary">下一处 →</button></div>
      <div id="error-position" class="hint" aria-live="polite"></div>
      <div id="error-list" class="error-list"></div>
      <div id="error-location" class="error-location"></div>
      <div id="error-review" class="error-review"></div>
      <div id="error-predictions" class="error-predictions"></div>
      <label for="error-view-length">局部视野边长（voxel）</label><input id="error-view-length" type="number" min="1" step="any" value="64">
      <div class="button-row"><button id="error-focus" type="button" class="quiet">定位事件</button><button id="error-review-focus" type="button" class="quiet">查看回溯区间</button></div>
      <div class="button-row"><button id="error-subtree" type="button" class="quiet">查看后续区域</button><button id="error-whole" type="button" class="quiet">显示全局</button></div>
      <label class="check"><input id="error-directions" type="checkbox" checked>显示父子方向箭头</label>
      <p class="error-legend">灰色：GT；青色：预测；红色：FJ 后代子树；橙色：缺失 GT。<br>琥珀色虚线：FJ 前段待核对区间。黄色 mismatch：匹配失效点；白色 anchor：前一匹配采样点；粉色 confirmed：持续失配确认点及路径。</p>
      <p class="hint">左右键切换当前筛选列表。输入框内保留编辑操作。estimated onset 是回溯估计，不是确定真值；mismatch 是阈值失配点。候选不计入已确认 FJ。红色后代包括重新接回 GT 的部分。</p>
    </div>`;
  const $=id=>body.querySelector('#error-'+id),navigation=viewer.navigation;
  let pair=null,rows=[],index=-1,epoch=0,busy=false;
  function buttons(){
    $('detect').disabled=busy;$('controls').hidden=!pair;
    $('prev').disabled=busy||index<=0;$('next').disabled=busy||index<0||index>=rows.length-1;
    $('focus').disabled=$('subtree').disabled=index<0;
    $('review-focus').disabled=index<0||!rows[index]?.onset_review?.segments?.length;
    $('position').textContent=rows.length?(index+1)+' / '+rows.length+' 处':'当前筛选无错误';
  }
  function ensureScene(){
    navigation.activate();
    if(!viewer.errorsActive())viewer.beginErrors(pair);
    viewer.setErrorDirections($('directions').checked);
  }
  function focus(){
    if(index<0)return;
    const length=Number($('view-length').value);
    if(!Number.isFinite(length)||length<=0){$('status').textContent='视野边长须为正数';return;}
    ensureScene();viewer.selectError(rows[index]);viewer.focusError({...rows[index],start:rows[index].onset_estimate?.estimated_onset||rows[index].start},length);
  }
  function select(target,zoom=true){
    if(!pair||target<0||target>=rows.length)return;
    index=target;ensureScene();const event=rows[index];viewer.selectError(event);
    for(const [i,row] of [...$('list').children].entries()){
      row.classList.toggle('active',i===index);row.setAttribute('aria-pressed',String(i===index));
      if(i===index)row.scrollIntoView({block:'nearest'});
    }
    const p=event.start,edge=p.edge_parent_id===null?'根节点 '+p.node_id:`原始边 ${p.edge_parent_id} → ${p.edge_child_id} · t=${p.edge_fraction.toFixed(4)}`;
    $('location').textContent=`${event.id} · ${names[event.error_type]} · ${event.source.toUpperCase()}\n${event.start_role==='matching_failure'?'匹配失效点':'起点'} XYZ：${xyz(p.xyz)}\n${edge}\n前一匹配采样点：${xyz(event.anchor.xyz)}`+
      (event.confirmation?`\n确认点：${xyz(event.confirmation.xyz)}\n后代节点：${event.affected_subtree.node_ids.length} · ${event.kind==='false_junction'?(event.is_first_fj?'首次 FJ':'后续 FJ'):(event.status||'诊断')}`:'')+
      (event.immediate_termination?'\n空预测 / 无有效追踪长度：从 GT seed 立即终止':'');
    if(event.onset_estimate?.estimated_onset)$('location').textContent+='\n估计起偏点（可能仍在容差内）：'+xyz(event.onset_estimate.estimated_onset.xyz)+'\n此点距参照 GT：'+event.onset_estimate.distance_to_reference.toFixed(3);
    if(event.evidence)$('location').textContent+='\n依据：'+JSON.stringify(event.evidence);
    drawReview(event);drawPredictions(event);buttons();if(zoom)focus();
  }
  function drawReview(event){
    const container=$('review');container.replaceChildren();
    const review=event.onset_review;if(!review)return;
    const note=document.createElement('p');note.className='hint';
    if(!review.segments.length){note.textContent='当前事件没有可回溯的上游路径，无法由此确定更早的错误起点。';container.append(note);return;}
    const boundary={prediction_root:'预测根',prediction_fork:'最近的预测分叉',gt_reentry:'前次重新接回 GT 的位置'}[review.boundary_kind]||review.boundary_kind;
    note.textContent=`起始段待核对：从${boundary}回溯边界 ${xyz(review.start.xyz)} 到失配点，共 ${review.length_voxel.toFixed(2)} voxel。琥珀色虚线中可能包含正确部分；边界不是确定的错误起点，更早的偏离也未被排除。`;
    container.append(note);
    const details=document.createElement('details'),summary=document.createElement('summary');
    summary.textContent='区间内父子边（'+review.segments.length+'）';details.append(summary);
    for(const segment of review.segments){
      const row=document.createElement('button');row.type='button';row.className='quiet error-review-edge';
      const source=segment.trace_origin;
      row.textContent=segment.edge_parent_id+' → '+segment.edge_child_id+
        (source?.prediction_step!=null?' · 预测 '+source.prediction_step:source?.status==='initial_tree'?' · 初始树':'')+
        ' · 定位此段';
      row.addEventListener('click',()=>{ensureScene();viewer.selectError(event);
        const point=segment.start_xyz.map((v,i)=>(v+segment.end_xyz[i])/2),length=Number($('view-length').value);
        viewer.focusError({start:{xyz:point}},Number.isFinite(length)&&length>0?length:64);
      });details.append(row);
    }
    container.append(details);
  }
  function drawPredictions(event){
    const container=$('predictions');container.replaceChildren();
    const origin=event.trace_origin;if(!origin)return;
    const note=document.createElement('p');note.className='hint';
    const references=new Map();
    const add=(index,label,ids=[],uncertain=false)=>{
      if(index==null)return;
      const previous=references.get(index)||{labels:[],ids:[],uncertainOnly:true};
      previous.labels=[...new Set([...previous.labels,label])];previous.ids=[...new Set([...previous.ids,...ids])];previous.uncertainOnly=previous.uncertainOnly&&uncertain;references.set(index,previous);
    };
    if(origin.start){
      const start=origin.start;
      note.textContent=start.status==='initial_tree'?'失配点位于初始树，没有本次模型预测概率。':
        start.status==='multiple_candidates'?'失配点所在边由同次预测的多个候选共享，列出全部匹配概率，不指定唯一候选。':
        start.status==='matched'?'失配点所在边已关联到生成它的预测与候选；不代表这就是最早追偏的预测。':'可定位生成步骤，但记录未提供唯一候选来源。';
      for(const [key,label] of [['estimated_onset','估计起偏边'],['start','失配点边'],['confirmation','确认点边']]){
        const ref=origin[key];add(ref?.prediction_index,label,(ref?.candidates||[]).map(c=>c.candidate_index));
      }
      for(const segment of event.onset_review?.segments||[]){
        const source=segment.trace_origin;add(source?.prediction_index,'回溯区间 · 待核对',(source?.candidates||[]).map(c=>c.candidate_index),true);
      }
    }else{
      note.textContent='break 无法唯一归因：以下仅为 GT 匹配锚点/分叉附近的相关预测，候选概率不作为漏分支或提前终止的概率。';
      for(const index of origin.related_prediction_indices||[])add(index,'相关预测');
      if(!references.size)note.textContent+=' 没有找到相关记录。';
    }
    container.append(note);
    for(const [index,ref] of references){
      const record=pair.result.trace_predictions[String(index)];if(!record)continue;
      const card=document.createElement('details');card.className='error-prediction-card';card.open=!!origin.start&&!ref.uncertainOnly;
      card.dataset.predictionIndex=index;card.classList.toggle('error-review-prediction',ref.uncertainOnly);
      const summary=document.createElement('summary');summary.textContent=ref.labels.join(' / ')+' · 预测 '+record.prediction_step+' · 日志第 '+record.log_line+' 行';card.append(summary);
      if(ref.labels.includes('回溯区间 · 待核对')){const hint=document.createElement('p');hint.className='hint';hint.textContent='此预测生成了回溯区间内的边，是否已追偏仍需核对；不指定它是最早错误预测。';card.append(hint);}
      if(!record.candidates.length){const empty=document.createElement('p');empty.className='hint';empty.textContent='本次没有被选择的候选；可展开全部候选分数检查。';card.append(empty);}
      for(const c of record.candidates){
        const row=document.createElement('div');row.className='error-probability';row.classList.toggle('matched',ref.ids.includes(c.candidate_index));
        row.textContent=(ref.ids.includes(c.candidate_index)?'对应边 · ':'')+'候选 #'+c.candidate_index+' · p='+(c.probability==null?'未记录':c.probability.toFixed(6))+' · '+(c.stop_reason||'未记录停止原因');card.append(row);
      }
      if(record.candidate_scores.length){
        const scores=document.createElement('details'),title=document.createElement('summary'),text=document.createElement('p');
        title.textContent='全部候选分数（'+record.candidate_scores.length+'）';text.className='error-all-scores';text.textContent=record.candidate_scores.map((p,i)=>'#'+i+': '+(p==null?'未记录':p.toFixed(6))).join(' · ');scores.append(title,text);card.append(scores);
      }
      const jump=document.createElement('button');jump.type='button';jump.className='quiet error-replay-jump';jump.textContent='回放此预测';jump.dataset.predictionIndex=index;
      jump.addEventListener('click',()=>viewer.openPrediction({path:pair.result.trace.path,gt_path:pair.result.trace.gt_path,index,candidate_indices:ref.ids}));card.append(jump);container.append(card);
    }
    const legend=document.createElement('p');legend.className='hint';legend.textContent='p 是记录中的模型候选概率，不是该候选出错的概率。';container.append(legend);
  }
  function rebuild(){
    if(!pair)return;const previous=rows[index]?.id,first=$('first').checked,type=$('type').value;
    const eligible=pair.result.events.filter(e=>!first||e.is_first_fj!==false);
    const category=$('category').value;
    rows=eligible.filter(e=>(type==='all'||e.error_type===type)&&(category==='all'||e.status===category||(category==='candidate'&&e.status==='undetermined')));
    const counts=first?pair.result.first_fj_counts:pair.result.counts;
    $('counts').replaceChildren();
    for(const [key,name] of Object.entries(names)){
      const item=document.createElement('div'),value=document.createElement('strong');item.textContent=name;value.textContent=(counts[key]||0)+(pair.result.candidate_counts?.[key]||0)+(pair.result.diagnostic_counts?.[key]||0);item.append(value);$('counts').append(item);
    }
    $('status').textContent=`${first?'首次':'全部'} FJ ${counts.false_junction} · break ${counts.break_error} · 合计 ${counts.total}`+
      ` · 候选/待定 ${pair.result.candidate_counts?.total||0} · 诊断 ${pair.result.diagnostic_counts?.total||0}`+
      (pair.result.fiber_metric?` · Fiber TP ${pair.result.fiber_metric.TP} / FP ${pair.result.fiber_metric.FP} / FN ${pair.result.fiber_metric.FN}`:'')+
      ` · 纯回溯诊断 ${pair.result.reversal_count} · FJ 归因去重 ${pair.result.suppressed_break_events.length}`+
      (pair.result.trace?` · ${pair.result.trace.prediction_count} 次预测 · 记录复原树`+(pair.result.trace.complete?'':' · 记录未完成，break 为当前状态'):'');
    const fragment=document.createDocumentFragment();
    rows.forEach((event,i)=>{
      const button=document.createElement('button'),detail=document.createElement('small');button.type='button';button.className='error-event';button.dataset.eventId=event.id;
      button.textContent=`${i+1}. ${names[event.error_type]||event.error_type} · ${{confirmed:'已确认',candidate:'候选',undetermined:'待定',diagnostic:'诊断'}[event.status]||''}`;
      detail.textContent=`${event.source.toUpperCase()} · ${xyz(event.start.xyz)}`+(event.kind==='false_junction'?(event.is_first_fj?' · 首次':' · 后续'):'');
      const source=event.trace_origin?.start;
      if(source?.prediction_step!=null)detail.textContent+=' · 失配边预测 '+source.prediction_step+(source.probability!=null?' · p='+source.probability.toFixed(4):' · 候选不唯一/未记录');
      else if(source?.status==='initial_tree')detail.textContent+=' · 初始树';
      button.append(detail);button.addEventListener('click',()=>select(i));fragment.append(button);
    });
    $('list').replaceChildren(fragment);index=-1;$('location').textContent='';$('review').replaceChildren();$('predictions').replaceChildren();buttons();
    ensureScene();
    if(rows.length)select(Math.max(0,rows.findIndex(e=>e.id===previous)));
    else{viewer.selectError(null);viewer.fitErrors(true);}
  }
  function close(){epoch++;busy=false;pair=null;rows=[];index=-1;viewer.stopErrors();navigation.release();buttons();$('status').textContent='已退出错误查看';}
  $('close').addEventListener('click',close);
  function inputMode(){
    const trace=$('mode').value==='trace';$('swc-input').hidden=trace;$('trace-input').hidden=!trace;
    $('pred').required=!trace;$('trace-path').required=trace;$('gt').required=!trace;
    $('gt-label').textContent=trace?'GT SWC（留空使用记录中的路径）':'GT SWC 文件';
  }
  $('form').addEventListener('input',()=>{inputMode();close();$('status').textContent='输入已更新，请重新检测。';});
  $('form').addEventListener('submit',async event=>{
    event.preventDefault();navigation.activate();const ticket=++epoch,valid=()=>ticket===epoch;busy=true;buttons();$('status').textContent='正在检测父子路径…';
    try{
      const source=$('mode').value==='trace'?{trace_path:$('trace-path').value}:{pred_path:$('pred').value};
      const result=await viewer.detectErrors({...source,gt_path:$('gt').value||null,
        distance_tolerance:Number($('distance').value),min_error_length:Number($('min-length').value),sample_step:Number($('step').value),coverage_threshold:Number($('coverage').value),short_length:$('short').value?Number($('short').value):null},valid);
      if(!valid()||!result)return;
      pair=result;viewer.beginErrors(pair);rebuild();
    }catch(error){if(valid())$('status').textContent='检测失败：'+error.message;}
    finally{if(valid()){busy=false;buttons();}}
  });
  for(const id of ['first','type','category'])$(id).addEventListener('change',rebuild);
  $('prev').addEventListener('click',()=>select(index-1));$('next').addEventListener('click',()=>select(index+1));
  $('focus').addEventListener('click',focus);$('view-length').addEventListener('change',focus);
  $('review-focus').addEventListener('click',()=>{if(index>=0){ensureScene();viewer.selectError(rows[index]);viewer.fitErrorReview();}});
  $('subtree').addEventListener('click',()=>{if(index>=0){ensureScene();viewer.selectError(rows[index]);viewer.fitErrors(false);}});
  $('whole').addEventListener('click',()=>{if(pair){ensureScene();viewer.fitErrors(true);}});
  $('directions').addEventListener('change',()=>viewer.setErrorDirections($('directions').checked));
  navigation.setStep(delta=>{if(!pair||busy||!rows.length)return false;select(index+delta);return true;});
  navigation.onDeactivate(()=>{epoch++;busy=false;buttons();});
  inputMode();buttons();
}
