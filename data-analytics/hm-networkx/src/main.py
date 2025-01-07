import logging

import matplotlib.pyplot as plt
import networkx as nx

logger = logging.getLogger(__name__)


def create_network() -> nx.Graph:
    # Initialize a new undirected graph
    G = nx.Graph()

    # Add nodes
    nodes = ["A", "B", "C", "D", "E"]
    G.add_nodes_from(nodes)

    # Add edges with weights
    edges = [
        ("A", "B", 2),
        ("B", "C", 1),
        ("C", "D", 3),
        ("D", "E", 2),
        ("A", "E", 6),
        ("B", "E", 3),
    ]
    G.add_weighted_edges_from(edges)
    return G


def analyze_network(G: nx.Graph) -> None:
    # Calculate and print basic network metrics
    logger.info("Network Analysis:")
    logger.info(f"Number of nodes: {G.number_of_nodes()}")
    logger.info(f"Number of edges: {G.number_of_edges()}")

    # Calculate degree for each node
    logger.info("Node degrees:")
    for node in G.nodes():
        logger.info(f"Node {node}: {G.degree(node)}")

    # Calculate betweenness centrality
    betweenness = nx.betweenness_centrality(G)
    logger.info("Betweenness centrality:")
    for node, bc in betweenness.items():
        logger.info(f"Node {node}: {bc:.3f}")

    # Calculate shortest paths
    logger.info("Shortest paths from node A:")
    for target in G.nodes():
        if target != "A":
            path = nx.shortest_path(G, "A", target, weight="weight")
            distance = nx.shortest_path_length(G, "A", target, weight="weight")
            logger.info(f"A to {target}: {path} (distance: {distance})")


def visualize_network(G: nx.Graph) -> None:
    """
    Create and display a visualization of the network.
    """
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500)

    # Draw edges
    nx.draw_networkx_edges(G, pos)

    # Draw labels
    nx.draw_networkx_labels(G, pos)

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Network Graph")
    plt.axis("off")
    plt.show()


def main() -> None:
    """
    Main function to execute the network analysis.
    """
    graph = create_network()
    analyze_network(graph)
    visualize_network(graph)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
