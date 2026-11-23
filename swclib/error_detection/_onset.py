"""Conservative pre-threshold onset estimates, never exact biological causes."""
import numpy as np
from scipy.spatial import cKDTree


def estimate_onset(event, gold, pred, pred_swc, params, short_length):
    from .detection import _position
    start = int(event['pred_start_index'])
    end = int(event['pred_confirmation_index'])
    frontier = event.get('gt_attribution_frontier_indices', [])
    result = dict(status='insufficient_evidence', estimated_onset=None, exact_onset=None,
                  method='stable_offset_then_positive_drift', calibrated=False,
                  interval_start=event['anchor'], interval_end=event['start'],
                  earlier_onset_not_excluded=True, probability=None)
    if not frontier:
        return result
    # A forest's different candidate lineages cannot provide a unique baseline.
    if any(not (gold.ancestor(a,b) or gold.ancestor(b,a))
           for a,b in zip(sorted(frontier), sorted(frontier)[1:])):
        result['status']='ambiguous_reference'
        return result
    g = min(frontier, key=lambda n:np.linalg.norm(gold.xyz[n]-pred.xyz[event['pred_anchor_index']]))
    window = max(20*params.distance, 10*short_length)
    nodes, cursor = [], end
    reset = int(event.get('classification_last_reset_index', -1))
    while cursor >= 0:
        nodes.append(cursor)
        if cursor == reset or pred.depth[end]-pred.depth[cursor] >= window:
            break
        cursor = int(pred.parent[cursor])
    nodes.reverse()
    result['window_length'] = float(pred.depth[end]-pred.depth[nodes[0]])
    result['window_truncated'] = bool(pred.parent[nodes[0]] >= 0)
    if len(nodes) < 5:
        return result
    gt_nodes, cursor = [], g
    while cursor >= 0:
        gt_nodes.append(cursor)
        if gold.depth[g]-gold.depth[cursor] >= window+params.distance:
            break
        cursor=int(gold.parent[cursor])
    gt_nodes.reverse()
    cursor=g
    while len(gold.children[cursor]) == 1 and gold.depth[cursor]-gold.depth[g] < window:
        cursor=gold.children[cursor][0];gt_nodes.append(cursor)
    if len(gt_nodes)<2:
        return result
    xyz=pred.xyz[nodes]
    # Project onto short exact GT segments next to the nearest sampled vertex.
    path=gold.xyz[gt_nodes]
    _, nearest=cKDTree(path).query(xyz)
    distances=np.full(len(xyz), np.inf)
    for offset in (-1,0):
        a=np.clip(nearest+offset,0,len(path)-2)
        delta=path[a+1]-path[a]
        square=np.einsum('ij,ij->i',delta,delta)
        t=np.divide(np.einsum('ij,ij->i',xyz-path[a],delta),square,
                    out=np.zeros(len(a)),where=square>0)
        q=path[a]+np.clip(t,0,1)[:,None]*delta
        distances=np.minimum(distances,np.linalg.norm(xyz-q,axis=1))
    x=pred.depth[nodes]-pred.depth[nodes[0]];y=distances
    # Duplicate zero-length vertices cannot increase the evidence weight.
    keep=np.r_[True,np.diff(x)>1e-12]
    x,y,indices=x[keep],y[keep],np.asarray(nodes)[keep]
    if len(x)<5:
        return result
    n=len(x);sy=y.sum();syy=y@y
    sx=np.cumsum(x[::-1])[::-1];sxx=np.cumsum((x*x)[::-1])[::-1]
    sy_tail=np.cumsum(y[::-1])[::-1];sxy=np.cumsum((x*y)[::-1])[::-1]
    k=np.arange(n);m=n-k
    sz=sx-m*x;szz=sxx-2*x*sx+m*x*x;szy=sxy-x*sy_tail
    denominator=szz-sz*sz/n
    slope=np.divide(szy-sz*sy/n,denominator,out=np.zeros(n),where=denominator>1e-12)
    intercept=(sy-slope*sz)/n
    sse=np.maximum(0,syy-intercept*sy-slope*szy)
    eligible=(x>=short_length)&(x[-1]-x>=params.min_error_length)&(indices<=start)&(slope>0)
    if not eligible.any():
        return result
    best=int(np.argmin(np.where(eligible,sse,np.inf)))
    baseline=y[:best+1]
    median=float(np.median(baseline))
    noise=float(1.4826*np.median(np.abs(baseline-median)))
    # A numerical/sampling floor prevents an exactly straight baseline from
    # making floating-point noise look like a scientifically certain onset.
    noise_floor=max(noise,params.sample_step*.1,1e-8)
    rise=float(slope[best]*(x[-1]-x[best]))
    constant_sse=float(syy-sy*sy/n)
    if rise<3*noise_floor or constant_sse-sse[best]<n*noise_floor**2:
        return result
    plausible=np.flatnonzero(eligible & (sse<=sse[best]+n*noise_floor**2))
    lower,upper=int(indices[plausible[0]]),int(indices[plausible[-1]])
    model_slope=float(slope[best])
    # Direction may change before distance has risen appreciably. Backtrack a
    # continuous direction-change run that reaches the distance-change region.
    points=pred.xyz[indices]
    motion=np.vstack((np.diff(points,axis=0),points[-1]-points[-2]))
    _,near=cKDTree(path).query(points)
    segments=np.clip(near,0,len(path)-2)
    tangent=path[segments+1]-path[segments]
    denominator_angle=np.linalg.norm(motion,axis=1)*np.linalg.norm(tangent,axis=1)
    cosine=np.divide(np.einsum('ij,ij->i',motion,tangent),denominator_angle,
                     out=np.ones(len(points)),where=denominator_angle>1e-12)
    angles=np.degrees(np.arccos(np.clip(cosine,-1,1)))
    baseline_angles=angles[:max(1,best)]
    baseline_angle=float(np.median(baseline_angles))
    angular_noise=float(1.4826*np.median(np.abs(baseline_angles-baseline_angle)))
    angle_limit=baseline_angle+max(3*angular_noise,1e-5)
    angular_start=best
    while angular_start>0 and angles[angular_start-1]>angle_limit:
        angular_start-=1
    if (angular_start<best and angles[best]>angle_limit
            and x[-1]-x[angular_start]>=params.min_error_length):
        best=angular_start;lower=min(lower,int(indices[best]))
        result['direction_backtracked']=True
    result['direction_deviation_degrees']=float(angles[best])
    result['baseline_direction_noise_degrees']=angular_noise
    result['method']='distance_change_point_with_directional_backtracking'
    onset=_position(pred,pred_swc,int(indices[best]))
    # At an exact turn vertex the outgoing edge owns the first deviation.
    # Referencing the incoming edge would blame the prediction that reached
    # the still-correct turn point instead of the prediction that left it.
    if best+1<len(indices):
        next_position=_position(pred,pred_swc,int(indices[best+1]))
        if next_position['edge_child_id'] != onset['edge_child_id']:
            onset.update(edge_parent_id=next_position['edge_parent_id'],
                         edge_child_id=next_position['edge_child_id'],edge_fraction=0.)
    result.update(status='estimated', estimated_onset=onset,
                  interval_start=_position(pred,pred_swc,lower),
                  interval_end=_position(pred,pred_swc,upper),
                  distance_to_reference=float(y[best]),baseline_distance=median,
                  baseline_noise=noise,drift_slope=model_slope,
                  root_to_estimated_onset_length=float(pred.depth[indices[best]]),
                  root_length_interval=[float(pred.depth[lower]),float(pred.depth[upper])],
                  interval_is_statistical_confidence=False)
    return result
