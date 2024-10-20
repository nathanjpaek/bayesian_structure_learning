import networkx as nx
import matplotlib.pyplot as plt

def read_gph(filename):
    G = nx.DiGraph()
    with open(filename, 'r') as f:
        for line in f:
            parent, child = line.strip().split(', ')
            G.add_edge(parent, child)
    return G

def plot_dag(G):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=3000, font_size=10, font_weight='bold', 
            arrows=True, edge_color='gray')
    
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("bayes net", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

G = read_gph('outputs/large.gph')

plot_dag(G)