"""Preserved-continuation queries without copying leaf sets along long trunks."""
import numpy as np
from scipy.spatial import cKDTree
from ._subtypes import _required_continuations


class _WitnessIndex:
    def __init__(self, gold, pred, states, last_reset):
        rows=[]
        for node,targets in enumerate(states):
            if not targets:
                continue
            distances=np.linalg.norm(gold.xyz[list(targets)]-pred.xyz[node],axis=1)
            best=float(distances.min())
            for g,distance in zip(targets,distances):
                if distance<=best+1e-9 and (gold.children[g] or not pred.children[node]):
                    rows.append((node,g,last_reset[node]))
        self.points=np.asarray(rows,dtype=np.int64).reshape(-1,3)
        self.tree=cKDTree(self.points,leafsize=16) if rows else None
        self.pred=pred

    def find(self, child, end, g, gend, fork):
        if self.tree is None:
            return None
        low=(child,g,-np.inf);high=(end-1,gend-1,fork)
        stack=[self.tree.tree]
        while stack:
            node=stack.pop()
            if node.split_dim<0:
                ids=node.indices;points=self.points[ids]
                hit=np.all((points>=low)&(points<=high),axis=1)
                hit &= self.pred.depth[points[:,0]]>self.pred.depth[fork]+1e-12
                choices=np.flatnonzero(hit)
                if len(choices):
                    p,gt,_=points[choices[0]]
                    return int(gt),int(p)
            else:
                dim=node.split_dim
                if low[dim]<=node.split:
                    stack.append(node.lesser)
                if high[dim]>=node.split:
                    stack.append(node.greater)
        return None


def classify_junctions(gold,pred,junctions,states,reset_nodes,params):
    if not junctions:
        return
    blocked=set(reset_nodes)
    last=np.full(len(pred),-1,dtype=int)
    for node,parent in enumerate(pred.parent):
        last[node]=node if node in blocked else (last[parent] if parent>=0 else -1)
    index=_WitnessIndex(gold,pred,states,last)
    for event in junctions:
        frontier=tuple(event['gt_attribution_frontier_indices'])
        required=_required_continuations(gold,frontier,params.distance)
        probe=event['pred_start_index'];boundary=int(last[probe]);candidates=[];extra=False
        while pred.segment_anchor[probe]>=0:
            fork=int(pred.segment_anchor[probe])
            if boundary>=0 and not pred.ancestor(boundary,fork):
                break
            event_child=int(pred.segment_start[probe]);represented={}
            for g in required:
                for child in pred.children[fork]:
                    if child==event_child or child in blocked:
                        continue
                    witness=index.find(child,pred.subtree_end[child],g,gold.subtree_end[g],fork)
                    if witness is not None:
                        represented[int(g)]=witness;break
            if len(pred.children[fork])>1:
                candidates.append(dict(pred_fork_index=fork,pred_fork_node_id=int(pred.original_id[fork]),
                    pred_event_child_index=event_child,preserved_gt_required_indices=sorted(represented),
                    witness_gt_indices=sorted({w[0] for w in represented.values()}),
                    witness_pred_indices=sorted({w[1] for w in represented.values()})))
            if required and len(represented)==len(required):
                event.update(error_type='extra_branch',classification_reason='same_gt_continuation_preserved_in_sibling',
                             extra_branch_evidence=candidates[-1])
                extra=True;break
            probe=fork
        if not extra:
            terminal=bool(frontier) and all(not gold.children[g] for g in frontier)
            event.update(error_type='terminal_overextension' if terminal else 'wrong_connection',
                         classification_reason='reached_terminal_gt_frontier' if terminal else 'correct_continuation_not_preserved')
        event.update(classification_frontier_indices=list(frontier),classification_last_reset_index=boundary,
                     classification_required_gt_indices=list(required),classification_fork_candidates=candidates,
                     classification_partial_sibling_support=any(0<len(c['preserved_gt_required_indices'])<len(required) for c in candidates))
