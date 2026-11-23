"""Fiber-referenced detection, separate from the unchanged scoring metric."""
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree

from ._matching import (DenseTree, Parameters, _prediction_events, _missing_components)
from ._fast_subtypes import classify_junctions
from ._fiber_reference import FiberReference
from ._onset import estimate_onset

FJ_TYPES=('wrong_connection','cross_neuron_merge','extra_branch','terminal_overextension')
BREAK_TYPES=('premature_termination','disconnection','missed_branch','missing_subtree')
CANDIDATE_TYPES=('false_branch_candidate','false_fragment_candidate','unsupported_path_candidate',
                 'ambiguous_connection','ambiguous_missing_branch','uncertain_missing_continuation')
DIAGNOSTIC_TYPES=('geometric_deviation','pure_reversal','duplicate_tracing','short_geometry')


def counts(events):
    c=Counter(e['error_type'] for e in events)
    result={key:c[key] for key in FJ_TYPES+BREAK_TYPES+CANDIDATE_TYPES+DIAGNOSTIC_TYPES}
    result.update(false_junction=sum(c[k] for k in FJ_TYPES),
                  break_error=sum(c[k] for k in BREAK_TYPES),total=len(events))
    return result


def _path(tree, first, last):
    nodes=[]
    while last>=0 and tree.ancestor(first,last):
        nodes.append(last)
        if last==first:
            return nodes[::-1]
        last=int(tree.parent[last])
    return []


def _compressed_segments(tree, members):
    segments=[];previous=-2;edge=-2
    for i in sorted(members):
        parent=int(tree.parent[i])
        if parent<0 or tree.edge_length[i]<=1e-12:
            continue
        current=int(tree.original_id[i])
        if segments and current==edge and parent==previous:
            segments[-1][1]=tree.xyz[i].tolist()
        else:
            segments.append([tree.xyz[parent].tolist(),tree.xyz[i].tolist()])
        edge=current;previous=i
    return segments


def _reentries(event, gold, pred, states):
    entries=event.get('pred_reentry_indices',[])
    if not entries and states[event['pred_confirmation_index']]:
        entries=[event['pred_confirmation_index']]
    result=[]
    for p in entries:
        choices=states[p]
        if choices:
            distances=np.linalg.norm(gold.xyz[list(choices)]-pred.xyz[p],axis=1)
            nearest=[int(g) for g,d in zip(choices,distances) if d<=distances.min()+1e-8]
            result.append((p,nearest))
    return result


def _same_path_geometry(event, gold, pred, states):
    """Return exact skipped GT intervals only when the lineage is unambiguous."""
    frontier=event.get('gt_attribution_frontier_indices',[])
    entries=_reentries(event,gold,pred,states)
    if not frontier or not entries:
        return None
    routes=[]
    for p,targets in entries:
        routes_here=[]
        for b in targets:
            compatible=[a for a in frontier if gold.ancestor(a,b)]
            if not compatible:
                return None
            a=max(compatible,key=lambda g:gold.depth[g])
            route=_path(gold,a,b)
            # A shortcut over a fork needs explicit branch reasoning, not a
            # blanket ancestor/descendant exemption.
            if any(len(gold.children[g])>1 for g in route[1:-1]):
                return None
            routes_here.append(route)
        route_p=_path(pred,event['pred_start_index'],p)
        if any(len(pred.children[n])>1 for n in route_p[:-1]):
            return None
        routes.extend(routes_here)
    return sorted(set(n for route in routes for n in route))


def _external_match(event, pred, external, primary_tree, radius, short_length):
    if not external:
        return None
    start=event['pred_start_index']
    # Only the error excursion, not an unrelated distant descendant branch.
    end=pred.subtree_end[start]
    ids=np.arange(start,end)
    ids=ids[(pred.depth[ids]-pred.depth[start])<=max(4*short_length,20*radius)]
    if not len(ids):
        return None
    eligible=[]
    for label,tree in external.items():
        distance,_=tree.query(pred.xyz[ids],distance_upper_bound=radius+1e-9)
        primary_distance,_=primary_tree.query(pred.xyz[ids],distance_upper_bound=radius+1e-9)
        hit=(distance<=radius)&(primary_distance>radius)
        support=float(pred.edge_length[ids[hit]].sum())
        if support>=short_length:
            eligible.append(dict(neuron_id=label,supported_length=support))
    return eligible or None


def _associate_breaks(junctions, missing, gold, coverage, radius, *, confirmed=True):
    """Sparse structural adjacency; do not force an ambiguous attribution."""
    by_parent={}
    for event in missing:
        by_parent.setdefault(event['gt_parent_index'],[]).append(event)
    claims={}
    for event in junctions:
        queue=[(g,0.) for g in event.get('gt_attribution_frontier_indices',[])]
        seen=set();eligible={}
        while queue:
            g,d=queue.pop()
            if g in seen:
                continue
            seen.add(g)
            for item in by_parent.get(g,[]):
                eligible[item['event_id']]=item
            for child in gold.children[g]:
                nd=d+gold.edge_length[child]
                if coverage[child] and nd<=radius+1e-9:
                    queue.append((child,nd))
        event['eligible_break_ids']=sorted(eligible)
        event['ambiguous_break_attribution']=len(eligible)>1
        if len(eligible)==1:
            key=next(iter(eligible));claims.setdefault(key,[]).append(event)
    for item in missing:
        owners=claims.get(item['event_id'],[])
        if len(owners)==1:
            event=owners[0]
            if confirmed:
                item['suppressed_by_false_junction']=event['event_id']
                item['associated_fj_id']=event['id']
                event['suppressed_break_id']=item['event_id']
            else:
                item['associated_candidate_ids']=[event['id']]
                event.setdefault('associated_missing_ids',[]).append(item['event_id'])


def detect(pred_input,gt_input,*,distance_tolerance,min_error_length,sample_step,
           coverage_threshold,short_length,other_gts,fiber_metric_options,max_fiber_pairs):
    from .detection import _checked_swc, _position, _junction_geometry, _onset_review
    params=Parameters(float(distance_tolerance),float(min_error_length),float(sample_step))
    if not np.isfinite(coverage_threshold) or not 0<coverage_threshold<=1:
        raise ValueError('coverage_threshold must be in (0,1]')
    short=2*params.distance if short_length is None else float(short_length)
    if not np.isfinite(short) or short<=0:
        raise ValueError('short_length must be finite and positive')
    pred_swc,gt_swc=_checked_swc(pred_input,'pred'),_checked_swc(gt_input,'gt')
    gold,pred=DenseTree(gt_swc,params.sample_step),DenseTree(pred_swc,params.sample_step)
    if not len(gold):
        raise ValueError('GT must contain at least one node')
    fibers=FiberReference(gt_swc,pred_swc,gold,pred,params.distance,coverage_threshold,
                          fiber_metric_options,max_fiber_pairs)
    external={}
    if other_gts is not None:
        if not isinstance(other_gts,dict):
            raise ValueError('other_gts must map explicit neuron identities to SWCs/paths')
        for label,value in other_gts.items():
            other=DenseTree(_checked_swc(value,'other_gt'),params.sample_step)
            if len(other):
                external[str(label)]=cKDTree(other.xyz)
    raw,coverage,offsets,terminals,states,resets=_prediction_events(
        gold,pred,params,return_states=True,confirm_topology=True)
    geometric_coverage=coverage.copy()
    junctions=[];candidates=[];diagnostics=[];geometric_resets=set()
    retained=[]
    primary_tree=cKDTree(gold.xyz) if external else None
    for event in raw:
        event['evidence']=dict(raw_matching_subtype=event['subtype'])
        external_matches=_external_match(event,pred,external,primary_tree,params.distance,short)
        if external_matches:
            event['external_neuron_candidates']=external_matches
        if event.get('direction_reversal') and not event['outside_gt_seen']:
            event.update(error_type='pure_reversal',status='diagnostic',kind='diagnostic')
            diagnostics.append(event);continue
        route=_same_path_geometry(event,gold,pred,states) if not external_matches else None
        if route is not None:
            event.update(error_type='geometric_deviation',status='diagnostic',kind='diagnostic')
            event['evidence'].update(reason='same_directed_unbranched_gt_path_before_and_after',
                                     gt_interval_node_ids=list(dict.fromkeys(map(int,gold.original_id[route]))),
                                     annotation_error_proven=False)
            coverage[route]=True  # Topological continuity, NOT correct-length credit.
            geometric_resets.update(p for p,_ in _reentries(event,gold,pred,states))
            diagnostics.append(event);continue
        retained.append(event)
    classify_junctions(gold,pred,retained,states,[n for n in resets if n not in geometric_resets],params)
    for event in retained:
        entries=_reentries(event,gold,pred,states)
        frontier=event.get('gt_attribution_frontier_indices',[])
        changed=bool(entries and frontier and all(
            not any(gold.ancestor(a,b) or gold.ancestor(b,a) for a in frontier for b in targets)
            for _,targets in entries))
        event['evidence'].update(reason=event.get('classification_reason'),
                                 reentry_pred_indices=[p for p,_ in entries],
                                 incompatible_gt_lineage=changed)
        external_matches=event.get('external_neuron_candidates',[])
        if len(external_matches)==1 and frontier:
            event.update(error_type='cross_neuron_merge',status='confirmed',kind='false_junction')
            junctions.append(event)
        elif len(external_matches)>1:
            event.update(error_type='ambiguous_connection',status='undetermined',kind='candidate')
            candidates.append(event)
        elif changed:
            event.update(status='confirmed',kind='false_junction')
            junctions.append(event)
        elif event['error_type']=='terminal_overextension' and event['max_evidence_length_voxel']>=short:
            event.update(status='confirmed',kind='false_junction')
            junctions.append(event)
        elif not frontier:
            event.update(error_type='false_fragment_candidate',status='candidate',kind='candidate')
            candidates.append(event)
        elif entries:
            event.update(error_type='ambiguous_connection',status='undetermined',kind='candidate')
            candidates.append(event)
        elif event['max_evidence_length_voxel']<short:
            event.update(error_type='short_geometry',status='diagnostic',kind='diagnostic')
            diagnostics.append(event)
        else:
            event['candidate_fj_type']=event['error_type']
            event.update(error_type='false_branch_candidate' if event['error_type']=='extra_branch' else 'unsupported_path_candidate',
                         status='candidate',kind='candidate')
            candidates.append(event)
    # Canonical event origins already merge shared trunks, irrespective of how
    # many FiberMetric root-to-terminal paths include them.
    first_end=-1;first_anomaly_end=-1
    junction_identities={id(e) for e in junctions}
    for event in sorted(junctions+candidates+diagnostics,key=lambda e:e['pred_start_index']):
        origin=event['pred_start_index']
        if id(event) in junction_identities:
            event['is_first_fj']=origin>=first_end
            if event['is_first_fj']:
                first_end=int(pred.subtree_end[origin])
        if event['status']!='diagnostic':
            event['is_first_anomaly']=origin>=first_anomaly_end
            if event['is_first_anomaly']:
                first_anomaly_end=int(pred.subtree_end[origin])
    for prefix,group in [('fj',junctions),('candidate',candidates),('diagnostic',diagnostics)]:
        for i,event in enumerate(group):
            event.update(event_id=i,id=f'{prefix}-{i}')
            _junction_geometry(event,pred,pred_swc)
            event['start_role']='matching_failure'
            event['location_interval']=dict(start=event['anchor'],end=event['start'])
            event['pred_fiber_ids']=fibers.ids('pred',pred,event['pred_start_index'])
            event['gt_fiber_ids']=sorted(set(fid for g in event.get('gt_attribution_frontier_indices',[])
                                            for fid in fibers.ids('gt',gold,g)))
            event['onset_review']=_onset_review(event,pred,pred_swc)
            if event['status']!='diagnostic':
                event['onset_estimate']=estimate_onset(event,gold,pred,pred_swc,params,short)
    missing=_missing_components(gold,coverage,params,terminals)
    immediate=bool(not len(pred) or (pred.edge_length.sum()<=1e-12 and gold.edge_length.sum()>1e-12))
    if immediate:
        missing=[dict(event_id=0,kind='break_error',subtype='premature_termination',
                      gt_start_index=gold.roots[0],gt_parent_index=-1,
                      max_missing_path_length_voxel=float(gold.depth.max()),
                      _missing_dense_indices=list(range(len(gold))),
                      reconnected_downstream=False,suppressed_by_false_junction=None,
                      immediate_termination=True,empty_prediction=not len(pred),zero_length_prediction=bool(len(pred)))]
    _associate_breaks(junctions,missing,gold,coverage,params.distance)
    _associate_breaks(candidates,missing,gold,coverage,params.distance,confirmed=False)
    terminal_indices=[i for i in range(len(pred)) if not pred.children[i] and states[i]]
    terminal_tree=cKDTree(pred.xyz[terminal_indices]) if terminal_indices else None
    for i,event in enumerate(missing):
        start=int(event['gt_start_index']);parent=int(event['gt_parent_index'])
        members=event.pop('_missing_dense_indices',[])
        # A point-only prediction covers a tolerance ball around the seed,
        # but has no distal traced component. The immediate-termination
        # override includes all GT nodes, including that ball; do not turn
        # its covered nodes into false reconnected endpoints.
        ends=[] if event.get('immediate_termination') else sorted(set(
            c for g in members for c in gold.children[g] if coverage[c]))
        error='disconnection' if ends else event['subtype']
        if parent<0 and not immediate:
            error='missing_subtree' if not ends else 'disconnection'
        p=_position(gold,gt_swc,start)
        anchor=_position(gold,gt_swc,parent if parent>=0 else start)
        event.update(id=f'break-{i}',event_id=i,error_type=error,subtype=error,
                     status='confirmed',source='gt',start=p,anchor=anchor,
                     gt_location=p,gt_interval=dict(start=anchor,ends=[_position(gold,gt_swc,n) for n in ends]),
                     missing_segments_xyz=_compressed_segments(gold,members),
                     gt_fiber_ids=fibers.ids('gt',gold,start),pred_fiber_ids=[],
                     related_pred_positions=[],evidence=dict(reason='uncovered_gt_connected_component',
                     distal_support_does_not_prove_connectivity=bool(ends)))
        if error=='missed_branch':
            fork=int(gold.segment_anchor[start])
            if fork>=0:
                event['start']=_position(gold,gt_swc,fork)
                event['start_role']='missing_branch_fork'
        if terminal_tree is not None:
            nearby=terminal_tree.query_ball_point(anchor['xyz'],params.distance+params.sample_step+1e-9)
            valid=[]
            for k in nearby:
                n=terminal_indices[k]
                if any(gold.ancestor(g,start) for g in states[n]):
                    valid.append(n)
            if valid:
                n=min(valid,key=lambda n:np.linalg.norm(pred.xyz[n]-anchor['xyz']))
                event['related_pred_positions']=[_position(pred,pred_swc,v) for v in valid]
                event['pred_fiber_ids']=sorted(set(fid for v in valid for fid in fibers.ids('pred',pred,v)))
                if error=='premature_termination' and not immediate:
                    event.update(start=_position(pred,pred_swc,n),source='pred',start_role='predicted_terminal')
        event['location_xyz']=event['start']['xyz'];event['anchor_xyz']=event['anchor']['xyz']
    # Even sub-tolerance gaps can be real disconnections. Inspect separately
    # rooted predicted fragments instead of treating their close endpoints as edges.
    _component_connections(gold,pred,gt_swc,pred_swc,states,coverage,missing,diagnostics,fibers,params)
    for event in missing:
        if (event['error_type']=='premature_termination' and not event.get('immediate_termination')
                and not event.get('related_pred_positions') and event.get('suppressed_by_false_junction') is None):
            event.update(error_type='uncertain_missing_continuation',subtype='uncertain_missing_continuation',
                         status='undetermined',kind='candidate')
            event['evidence']['reason']='missing_GT_without_a_supported_predicted_terminal_or_unique_confirmed_FJ'
            candidates.append(event)
    breaks=[e for e in missing if e.get('suppressed_by_false_junction') is None and e['status']=='confirmed']
    suppressed=[e for e in missing if e.get('suppressed_by_false_junction') is not None]
    _duplicate_fp_fibers(fibers,junctions+breaks+candidates+diagnostics,diagnostics,pred,pred_swc,coverage_threshold)
    _ambiguous_fn_fibers(fibers,junctions+breaks+candidates+diagnostics,candidates,gold,gt_swc,coverage_threshold)
    events=junctions+breaks+candidates+diagnostics
    _fiber_explanations(fibers,events,pred,gold)
    confirmed=junctions+breaks
    first=[e for e in confirmed if e.get('is_first_fj',True)]
    combined=counts(confirmed+candidates)
    combined['false_junction_candidates']=sum(e['error_type'] in ('false_branch_candidate','unsupported_path_candidate','ambiguous_connection') for e in candidates)
    combined['break_error_candidates']=sum(e['error_type'] in ('ambiguous_missing_branch','uncertain_missing_continuation') for e in candidates)
    combined['false_junction_including_candidates']=len(junctions)+combined['false_junction_candidates']
    result=dict(schema_version=2,strategy='fiber',events=events,false_junction_events=junctions,
                break_events=breaks,candidate_events=candidates,diagnostic_events=diagnostics,
                suppressed_break_events=suppressed,counts=counts(confirmed),first_fj_counts=counts(first),
                candidate_counts=counts(candidates),diagnostic_counts=counts(diagnostics),
                counts_with_candidates=combined,fiber_metric=fibers.raw,
                fiber_metric_parameters=fibers.parameters,fiber_catalog=fibers.catalog,
                fiber_references=fibers.references,fp_fiber_ids=fibers.fp,fn_fiber_ids=fibers.fn,
                fiber_explanations=fibers.explanations,
                parameters=dict(distance_tolerance=params.distance,min_error_length=params.min_error_length,
                    sample_step=params.sample_step,coverage_threshold=float(coverage_threshold),short_length=short,
                    max_fiber_pairs=max_fiber_pairs,coordinate_units='input_xyz_units'),
                gt_nodes=len(gt_swc.nodes),pred_nodes=len(pred_swc.nodes),gt_dense_nodes=len(gold),pred_dense_nodes=len(pred),
                root_offsets_voxel=offsets,empty_prediction=not len(pred),
                zero_length_prediction=bool(len(pred) and pred.edge_length.sum()<=1e-12),
                immediate_termination=immediate,
                covered_gt_dense_nodes=int(geometric_coverage.sum()),
                reversal_diagnostics=[e for e in diagnostics if e['error_type']=='pure_reversal'])
    result['reversal_count']=len(result['reversal_diagnostics'])
    for key,value in result['counts'].items():
        result[key+'_count']=value
    result['has_false_junction']=bool(junctions);result['has_break_error']=bool(breaks)
    return result


def _component_connections(gold,pred,gt_swc,pred_swc,states,coverage,missing,diagnostics,fibers,params):
    from .detection import _position
    if len(pred.roots)<=1:
        return
    # Nearest directionally supported GT position for each actual component.
    component=np.empty(len(pred),dtype=int)
    projected=np.full(len(pred),-1,dtype=int)
    for i,p in enumerate(pred.parent):
        component[i]=i if p<0 else component[p]
        if states[i]:
            projected[i]=min(states[i],key=lambda g:np.linalg.norm(gold.xyz[g]-pred.xyz[i]))
    rooted={r for r in pred.roots if states[r] and any(gold.depth[g]<=params.distance for g in states[r])}
    rooted_points=np.flatnonzero(np.isin(component,list(rooted)) & (projected>=0))
    rooted_tree=cKDTree(pred.xyz[rooted_points]) if len(rooted_points) else None
    tips=np.asarray([i for i in range(len(pred)) if not pred.children[i] and projected[i]>=0],dtype=int)
    tip_tree=cKDTree(pred.xyz[tips]) if len(tips) else None
    for root in pred.roots:
        g=int(projected[root])
        if root in rooted or g<0 or not pred.children[root]:
            continue
        # An independent rooted path that already passes the attachment site
        # makes a duplicate fragment, not another break at the same site.
        duplicate=False
        if rooted_tree is not None:
            near=rooted_tree.query_ball_point(pred.xyz[root],params.distance)
            duplicate=any(gold.ancestor(g,int(projected[rooted_points[k]])) and
                          gold.depth[projected[rooted_points[k]]]>gold.depth[g]+params.sample_step
                          for k in near)
        if duplicate:
            fragment_tips=[i for i in range(root,pred.subtree_end[root]) if not pred.children[i]]
            for tip in fragment_tips:
                near_tip=rooted_tree.query_ball_point(pred.xyz[tip],params.distance)
                target=int(projected[tip])
                if target<0 or not any(gold.ancestor(int(projected[rooted_points[k]]),target)
                                      or gold.ancestor(target,int(projected[rooted_points[k]])) for k in near_tip):
                    duplicate=False;break
        if duplicate:
            diagnostics.append(dict(id=f'diagnostic-{len(diagnostics)}',event_id=len(diagnostics),
                kind='diagnostic',status='diagnostic',error_type='duplicate_tracing',source='pred',
                start=_position(pred,pred_swc,root),anchor=_position(pred,pred_swc,root),
                affected_subtree=dict(node_ids=list(dict.fromkeys(map(int,pred.original_id[root:pred.subtree_end[root]])))),
                pred_fiber_ids=fibers.ids('pred',pred,root),gt_fiber_ids=fibers.ids('gt',gold,g),
                evidence=dict(reason='independent_rooted_prediction_already_continues_here')))
            continue
        same=[e for e in missing if any(gold.ancestor(e['gt_start_index'],g) and
                   gold.ancestor(g,n['dense_index']) for n in e.get('gt_interval',{}).get('ends',[]))]
        if not same:
            same=[e for e in missing if e['error_type']=='disconnection' and any(
                  abs(gold.depth[n['dense_index']]-gold.depth[g])<=params.distance+params.sample_step
                  and (gold.ancestor(n['dense_index'],g) or gold.ancestor(g,n['dense_index']))
                  for n in e.get('gt_interval',{}).get('ends',[]))]
        if same:
            same[0].setdefault('related_pred_positions',[]).append(_position(pred,pred_swc,root))
            same[0]['connection_endpoints']=list(same[0]['related_pred_positions'])
            same[0]['pred_fiber_ids']=sorted(set(same[0]['pred_fiber_ids']+fibers.ids('pred',pred,root)))
            continue
        pos=_position(gold,gt_swc,g)
        endpoints=[_position(pred,pred_swc,root)]
        nearby_tips=tip_tree.query_ball_point(pred.xyz[root],2*params.distance+params.sample_step) if tip_tree else []
        for index in nearby_tips:
            tip=int(tips[index]);target=int(projected[tip])
            if (not pred.children[tip] and component[tip]!=root and target>=0
                    and gold.ancestor(target,g)
                    and np.linalg.norm(pred.xyz[tip]-pred.xyz[root])<=2*params.distance+params.sample_step):
                endpoints.insert(0,_position(pred,pred_swc,tip))
        missing.append(dict(id=f'break-{len(missing)}',event_id=len(missing),kind='break_error',
            status='confirmed',error_type='disconnection',subtype='disconnection',source='gt',
            start=pos,anchor=pos,gt_start_index=g,gt_parent_index=int(gold.parent[g]),gt_location=pos,
            gt_interval=dict(start=pos,ends=[pos]),missing_segments_xyz=[],
            related_pred_positions=endpoints,connection_endpoints=endpoints,
            pred_fiber_ids=fibers.ids('pred',pred,root),gt_fiber_ids=fibers.ids('gt',gold,g),
            evidence=dict(reason='supported_distal_component_has_no_parent_connection',
                          sub_tolerance_gap=True),suppressed_by_false_junction=None,
            location_xyz=pos['xyz'],anchor_xyz=pos['xyz']))


def _fiber_explanations(fibers,events,pred,gold):
    mapping={('pred',i):[] for i in fibers.fp}
    mapping.update({('gt',i):[] for i in fibers.fn})
    for event in events:
        for source in ('pred','gt'):
            for fid in event.get(source+'_fiber_ids',[]):
                if (source,fid) in mapping:
                    mapping[source,fid].append(event)
    explanations=[]
    for (source,fid),items in mapping.items():
        def order(event):
            if source=='pred':
                indices=([event['pred_start_index']] if 'pred_start_index' in event else
                         [p['dense_index'] for p in event.get('related_pred_positions',[])])
                distance=min((pred.depth[n] for n in indices),default=float('inf'))
            else:
                indices=([event['gt_start_index']] if 'gt_start_index' in event else
                         event.get('gt_attribution_frontier_indices',[]))
                distance=min((gold.depth[n] for n in indices),default=float('inf'))
            return (event['status']=='diagnostic',distance,event['id'])
        items.sort(key=order)
        explanations.append(dict(source=source,fiber_id=fid,metric_label='FP' if source=='pred' else 'FN',
                                 primary_event_id=items[0]['id'] if items else None,
                                 event_ids=[e['id'] for e in items],
                                 status='explained' if items else 'undetermined'))
    fibers.explanations=explanations


def _duplicate_fp_fibers(fibers,events,diagnostics,pred,pred_swc,threshold):
    from .detection import _position
    explained={fid for e in events for fid in e.get('pred_fiber_ids',[])}
    gt_matched={int(m[0]):int(m[1]) for m in (fibers.raw['matches'] or [])}
    nodes,ids=fibers.leaves['pred'];lookup=dict(zip(map(int,ids),map(int,nodes)))
    for reference in fibers.references:
        fid=reference['pred_fiber_id']
        if fid not in fibers.fp or fid in explained or reference['best_coverage_fraction']<threshold:
            continue
        choices=reference['candidates']
        if len(choices)!=1 or choices[0]['gt_fiber_id'] not in gt_matched:
            continue
        target=fibers.pred_fibers[gt_matched[choices[0]['gt_fiber_id']]]
        coordinates,_=fibers.pred_fibers[fid].cache_resample_by_distance(.5)
        _,tree=target.cache_resample_by_distance(.5)
        distances,indices=tree.query((coordinates[:-1]+coordinates[1:])*.5)
        if np.any(distances>fibers.parameters['dist_threshold']) or np.any(np.diff(indices)<0):
            continue
        tip=lookup[fid];anchor=int(pred.segment_anchor[tip])
        start=anchor if anchor>=0 else tip
        diagnostics.append(dict(id=f'diagnostic-{len(diagnostics)}',event_id=len(diagnostics),
            kind='diagnostic',status='diagnostic',error_type='duplicate_tracing',source='pred',
            start=_position(pred,pred_swc,start),anchor=_position(pred,pred_swc,start),
            affected_subtree=dict(node_ids=list(dict.fromkeys(map(int,pred.original_id[start:pred.subtree_end[start]])))),
            pred_fiber_ids=[fid],gt_fiber_ids=[choices[0]['gt_fiber_id']],
            evidence=dict(reason='supported_fp_path_already_explained_by_matched_gt_fiber')))


def _ambiguous_fn_fibers(fibers,events,candidates,gold,gt_swc,threshold):
    from .detection import _position
    explained={fid for e in events for fid in e.get('gt_fiber_ids',[])}
    references={}
    for ref in fibers.references:
        for choice in ref['candidates']:
            if choice['coverage_fraction']>=threshold:
                references.setdefault(choice['gt_fiber_id'],[]).append(ref['pred_fiber_id'])
    nodes,ids=fibers.leaves['gt'];lookup=dict(zip(map(int,ids),map(int,nodes)))
    for fid in fibers.fn:
        if fid in explained or fid not in references:
            continue
        leaf=lookup[fid];fork=int(gold.segment_anchor[leaf]);start=fork if fork>=0 else leaf
        pos=_position(gold,gt_swc,start)
        candidates.append(dict(id=f'candidate-{len(candidates)}',event_id=len(candidates),
            kind='candidate',status='undetermined',error_type='ambiguous_missing_branch',
            source='gt',start=pos,anchor=pos,gt_start_index=start,
            pred_fiber_ids=references[fid],gt_fiber_ids=[fid],
            missing_segments_xyz=_compressed_segments(gold,_path(gold,start,leaf)[1:]),
            evidence=dict(reason='fiber_FN_has_spatial_support_but_branch_identity_is_not_unique',
                          fiber_metric_assignment_is_not_causal_evidence=True)))
