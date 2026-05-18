import networkx as nx
from pathlib import Path

def read_swc(file_name):
    nodes = {}
    edges = []
    with open(file_name) as f:
        for line in f.readlines():
            if line.startswith("#"):
                continue
            if len(line.split()) != 7:
                continue
            id, ntype, x, y, z, r, pid = map(float, line.split())
            id = int(id)
            ntype = int(ntype)
            pid = int(pid)
            nodes[id] = {
                "id": id,
                "type": ntype,
                "x": x,
                "y": y,
                "z": z,
                "radius": r,
                "parent": pid,
            }
            edges.append((id, pid))
    return nodes, edges

def nx_clear_invalid_edges(G: nx.Graph):
    valid_nodes = set(G.nodes())
    invalid_edges = [
        (u, v) for u, v in G.edges() if u not in valid_nodes or v not in valid_nodes
    ]
    G.remove_edges_from(invalid_edges)
    return G

def nx_swc_to_grpah(swc_path, scale=(1.0, 1.0, 1.0)):
    if isinstance(swc_path, str) or isinstance(swc_path, Path):
        nodes, edges = read_swc(swc_path)
    G = nx.Graph()

    for nid in nodes:
        node = nodes[nid]
        swc_node = tuple([node["x"]*scale[0], node["y"]*scale[1], node["z"]*scale[2]])
        G.add_node(nid, coord=swc_node, ntype=0)
    for edge in edges:
        if edge[1]!=-1:
            G.add_edge(edge[0], edge[1])
    return G


def nx_graph_to_swc(G: nx.Graph, scale=(1, 1, 1), swc_path=None, radius=0.1):
    components = list(nx.connected_components(G))
    swc_lines = []
    node_id_map = {}
    swc_counter = 1
    for comp_nodes in components:
        tree = G.subgraph(comp_nodes)
        root_node = None
        for node in tree.nodes:
            if tree.nodes[node]["ntype"] == 1:
                root_node = node
                break
        if root_node is None:
            for node, degree in tree.degree:
                if degree==1:
                    root_node = node
                    break
        if root_node is None: # circle
            continue
        # degrees = tree.degree()
        # root_node = max(degrees, key=lambda x: x[1])[0]
        # if G.degree(root_node)==2:
        #     for node, degree in degrees:
        #         if degree==1:
        #             root_node = node
        #             break

        parent = {root_node: -1}
        visited = set([root_node])
        queue = [root_node]

        while queue:
            current = queue.pop(0)
            for neighbor in tree.neighbors(current):
                if neighbor not in visited:
                    parent[neighbor] = current
                    visited.add(neighbor)
                    queue.append(neighbor)
        for node in parent:
            coord = tree.nodes[node]["coord"]
            ntype = tree.nodes[node]["ntype"]
            pid = parent[node]
            pid_swc = node_id_map.get(pid, -1)

            swc_lines.append(
                f"{swc_counter} {ntype} {coord[0]*scale[0]:.13e} {coord[1]*scale[1]:.13e} {coord[2]*scale[2]:.13e} {radius:.13e} {pid_swc}\n"
            )
            node_id_map[node] = swc_counter
            swc_counter += 1
    with open(swc_path, "w") as f:
        f.writelines(swc_lines)