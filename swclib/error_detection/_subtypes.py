"""Split existing FJ events by preserved GT continuation, without changing counts.

Uses the original directed matching states; a sibling that becomes correct only
after its own FJ/reset cannot serve as evidence of a preserved correct path.
"""
import numpy as np

from ._matching import _maximal_states


FJ_TYPES = ('wrong_connection', 'extra_branch', 'terminal_overextension')


def _required_continuations(gt, frontier, distance):
    """Resolve a GT fork inside the existing local attribution tolerance.

    A common trunk at/near a GT fork must not let one preserved child stand in
    for the missing other child. Away from a fork the reached frontier suffices.
    """
    def expand(start, remaining):
        if remaining < -1e-9:
            return [start]
        node = start
        while len(gt.children[node]) == 1:
            child = gt.children[node][0]
            if gt.edge_length[child] > remaining + 1e-9:
                return [start]
            remaining -= gt.edge_length[child]
            node = child
        if len(gt.children[node]) > 1:
            return [target for child in gt.children[node]
                    for target in expand(child, remaining - gt.edge_length[child])]
        return [start]
    return tuple(sorted(set(target for node in frontier for target in expand(node, distance))))


def classify_junctions(gt, pred, junctions, states, reset_nodes, params):
    if not junctions:
        return
    # Stop at every actual correspondence reset. Event origins alone are not
    # sufficient: a queried fork can lie downstream of a shared error origin.
    # A short unconfirmed excursion that recovers the original directed match
    # remains eligible, consistently with the original persistence threshold.
    blocked = set(reset_nodes)
    last_reset = np.full(len(pred), -1, dtype=int)
    for node, parent in enumerate(pred.parent):
        last_reset[node] = node if node in blocked else (last_reset[parent] if parent >= 0 else -1)
    support = [{} for _ in range(len(pred))]
    for node in range(len(pred) - 1, -1, -1):
        targets = states[node]
        closest = ()
        if targets:
            distances = np.linalg.norm(gt.xyz[list(targets)] - pred.xyz[node], axis=1)
            best = float(distances.min())
            # Nearest among directionally reachable states, keeping geometric ties.
            # Using every point in the matching ball would turn a zero-length
            # sibling into apparent forward tracing by one distance tolerance.
            closest = tuple(g for g, distance in zip(targets, distances)
                            if distance <= best + 1e-9)
        # A GT leaf is preserved by a correctly ending prediction, not by a
        # second path that passes its matching ball and also overextends.
        witnesses = {g: node for g in closest if gt.children[g] or not pred.children[node]}
        for child in pred.children[node]:
            if child in blocked:
                continue
            for g, witness in support[child].items():
                if g not in witnesses or pred.depth[witness] > pred.depth[witnesses[g]]:
                    witnesses[g] = witness
        support[node] = {g: witnesses[g] for g in _maximal_states(gt, witnesses)}

    for event in junctions:
        frontier = tuple(event['gt_attribution_frontier_indices'])
        required = _required_continuations(gt, frontier, params.distance)
        candidates = []
        probe = event['pred_start_index']
        boundary = int(last_reset[probe])
        extra = False
        # Search successive actual prediction forks on the event's ancestry.
        while pred.segment_anchor[probe] >= 0:
            fork = int(pred.segment_anchor[probe])
            # A later error after reconnection is a new path transfer. An old
            # fork before that reconnection cannot explain it as a new branch.
            if boundary >= 0 and not pred.ancestor(boundary, fork):
                break
            event_child = int(pred.segment_start[probe])
            witnesses = [(g, witness) for child in pred.children[fork]
                         if child != event_child and child not in blocked
                         for g, witness in support[child].items()
                         if pred.depth[witness] > pred.depth[fork] + 1e-12]
            represented = {int(g): next((int(w), int(p)) for w, p in witnesses if gt.ancestor(g, w))
                           for g in required if any(gt.ancestor(g, w) for w, p in witnesses)}
            if len(pred.children[fork]) > 1:
                candidates.append({'pred_fork_index': fork,
                                   'pred_fork_node_id': int(pred.original_id[fork]),
                                   'pred_event_child_index': event_child,
                                   'preserved_gt_required_indices': sorted(represented),
                                   'witness_gt_indices': sorted({w[0] for w in represented.values()}),
                                   'witness_pred_indices': sorted({w[1] for w in represented.values()})})
            if required and len(represented) == len(required):
                event['error_type'] = 'extra_branch'
                event['classification_reason'] = 'same_gt_continuation_preserved_in_sibling'
                event['extra_branch_evidence'] = candidates[-1]
                extra = True
                break
            probe = fork
        if not extra:
            # No forward GT continuation remains once every reached frontier is
            # terminal. A sibling preserving it would already have won above.
            terminal = bool(frontier) and all(not gt.children[g] for g in frontier)
            event['error_type'] = 'terminal_overextension' if terminal else 'wrong_connection'
            event['classification_reason'] = ('reached_terminal_gt_frontier' if terminal else
                                               'correct_continuation_not_preserved')
        event['classification_frontier_indices'] = list(frontier)
        event['classification_last_reset_index'] = boundary
        event['classification_required_gt_indices'] = list(required)
        event['classification_fork_candidates'] = candidates
        event['classification_partial_sibling_support'] = any(
            0 < len(c['preserved_gt_required_indices']) < len(required) for c in candidates)
