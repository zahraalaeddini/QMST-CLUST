import time
import numpy as np
import igraph
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

def q_learning(graph, num_episodes, learning_rate, discount_factor):
    Q = np.ones((len(graph), len(graph)))
    for episode in range(num_episodes):
        # Reset the graph for each episode
        current_graph = graph.copy()
        # Choose a random starting node
        current_node = np.random.randint(len(graph))
        while True:
            # Select an action (next node) based on exploration or exploitation strategy
            action = epsilon_greedy(Q[current_node], episode)
            # Update the Q-table with Q-learning equation
            reward = -current_graph[current_node, action]  # Negative reward for selecting an edge
            Q[current_node, action] = (1 - learning_rate) * Q[current_node, action] + learning_rate * (
                        reward + discount_factor * np.min(Q[action]))
            # Remove the edge between the current node and the selected action
            current_graph[current_node, action] = 0
            current_graph[action, current_node] = 0
            # Move to the next node
            current_node = action
            # If all edges are removed, MST is found for this episode
            if np.sum(current_graph) == 0:
                break
    return Q

def epsilon_greedy(Q_values, epsilon=0.1):
    # Exploration or exploitation strategy
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.randint(len(Q_values))
    else:
        action = np.argmin(Q_values)  # Select the action with the minimum Q-value
    return action

def get_minimum_spanning_tree(Q_table):
    num_nodes = len(Q_table)
    MST = np.zeros_like(Q_table)

    # Create a list of all edges in the graph with their respective weights
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            edges.append((i, j, Q_table[i][j]))

    # Sort the edges in increasing order of weight
    edges.sort(key=lambda x: x[2])

    # Perform Kruskal's algorithm to construct a MST
    parent = [i for i in range(num_nodes)]  # Stores the parent of each node
    rank = [0] * num_nodes  # Stores the rank of each node

    # Find the parent of a node using path compression
    def find(parent, node):
        if parent[node] != node:
            parent[node] = find(parent, parent[node])
        return parent[node]

    # Union two sets using rank
    def union(parent, rank, x, y):
        x_root = find(parent, x)
        y_root = find(parent, y)
        if rank[x_root] < rank[y_root]:
            parent[x_root] = y_root
        elif rank[x_root] > rank[y_root]:
            parent[y_root] = x_root
        else:
            parent[y_root] = x_root
            rank[x_root] += 1

    # Iterate over all edges in increasing order of weight
    for edge in edges:
        x, y, weight = edge
        x_root = find(parent, x)
        y_root = find(parent, y)
        if x_root != y_root:
            MST[x][y] = 1
            union(parent, rank, x_root, y_root)

    return MST

def plot_graph(graph, MST, column_clusters, cluster_colors):
    # Create an igraph graph from the adjacency matrix
    g = igraph.Graph.Adjacency(graph.tolist())
    # Define layout for plotting
    layout = g.layout_reingold_tilford(mode="in", root=[0])
    # Plot the original graph
    fig, ax = plt.subplots()
    g.vs["color"] = [cluster_colors[i] for i in column_clusters]  # Set node color based on cluster
    g.es["color"] = "lightgray"  # Set edge color
    g.vs["label"] = range(len(graph))  # Set node labels
    g.es["width"] = 0.1  # Set edge width
    igraph.plot(g, target=ax, layout=layout, bbox=(300, 300))

    # Highlight edges from the minimum spanning tree
    for i in range(len(graph)):
        for j in range(len(graph)):
            if MST[i, j] == 1:
                g.add_edge(i, j)
    g.es.select(color="lightgray").delete()  # Delete non-tree edges
    g.es["color"] = "blue"  # Set edge color for tree edges
    g.vs['width'] = 0.5
    g.es["width"] = 1.0  # Set edge width for tree edges
    igraph.plot(g, target=ax, layout=layout, bbox=(300, 300))

def main():
    np.random.seed(0)
    start_time = time.time()
    dataset_path = input("Enter the dataset path: ")
    graph_data = pd.read_csv(dataset_path, index_col=False)
    counts = graph_data.values
    # Prompt the user for the number of clusters
    num_clusters = int(input("Enter the number of clusters: "))
    # Cluster columns using K-means
    kmeans = KMeans(n_clusters=num_clusters)
    column_clusters = kmeans.fit_predict(counts.T)
    # Create a new graph with cluster-wise counts
    cluster_counts = np.zeros((num_clusters, num_clusters))
    for i in range(counts.shape[1]):
        for j in range(i + 1, counts.shape[1]):
            cluster_counts[column_clusters[i], column_clusters[j]] += counts[i, j]
            cluster_counts[column_clusters[j], column_clusters[i]] += counts[j, i]

    learning_rate = 0.1
    discount_factor = 0.9
    num_episodes = 1000
    Q_table = q_learning(cluster_counts, num_episodes, learning_rate, discount_factor)
    MST = get_minimum_spanning_tree(Q_table)

    # Define cluster colors for plotting
    cluster_colors = ["green", "blue", "red", 'pink', 'violet', 'gray', 'yellow', 'brown',
                      'magenta', 'orange']
    plot_graph(cluster_counts, MST, column_clusters, cluster_colors)
    plt.show()

    # Print cells in each cluster
    cluster_cells = {}
    for i, cluster_idx in enumerate(column_clusters):
        if cluster_idx not in cluster_cells:
            cluster_cells[cluster_idx] = []
        cluster_cells[cluster_idx].append(i)

    for cluster_idx, cells in cluster_cells.items():
        print(f"Cluster {cluster_idx}: Cells {cells}")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")


if __name__ == '__main__':
    main()

