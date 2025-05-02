import networkx as nx
import matplotlib.pyplot as plt
from simulators import ContractionGraph
from typing import Tuple


def visualize_contraction_graph(cg: ContractionGraph,
                                figsize: Tuple[int, int] = (16, 9),
                                dpi: int = 300) -> None:
    G = nx.MultiGraph()
    for node_id in cg.nodes:
        G.add_node(node_id)

    leg_labels = {}
    for edge_id, edge in cg.edges.items():
        G.add_edge(edge.u, edge.v, key=edge_id)
        leg_labels[edge_id] = f"{edge.leg_u}↔{edge.leg_v}"

    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax)

    for u, v, key in G.edges(keys=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        offset = (hash(key) % 10 - 5) * 0.005

        ax.text(xm + offset, ym + 0.03 + offset,
                str(key),
                fontsize=10, color='gray',
                ha='center', va='center')
        
        ax.text(xm + offset, ym - 0.03 + offset,
                leg_labels[key],
                fontsize=10, color='black',
                ha='center', va='center')

    ax.axis('off')
    plt.title("Tensor Network Graph for QAOA")
    plt.show()