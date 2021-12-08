import os
import json
import networkx as nx
from matplotlib import pyplot as plt
import pickle
import random

def show_graph(G, label_offset=0.02, plot_margin=0.1, show_edge_attribute=True):
    # Your code here
    fig, ax = plt.subplots(figsize=(20, 20))
    pos = nx.spring_layout(G) # obtain edges pos(ition)
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='red')
    scaling_factor = random.randint(1,10)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='grey', connectionstyle="arc3,rad=-0.1")
    nx.draw_networkx_labels(G, {k: (u, v + label_offset) for k, (u, v) in pos.items()}) # dictionary comprehension expression
    if show_edge_attribute:
        edge_labels = {(u, v): d['relation'] for (u, v, d) in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels)
    
    xs = [p[0] for p in pos.values()] # extract all x...
    ys = [p[1] for p in pos.values()] # ...and y values from edges positions
    
    ax.set_xlim((min(xs) - plot_margin, max(xs) + plot_margin))
    ax.set_ylim((min(ys) - plot_margin, max(ys) + plot_margin))
    
    plt.savefig("Graph.png", format="PNG")
    pass

relations = json.load(open('./checkpoints/result.json'))
entity_info = json.load(open('./data/DocRED/test.json'))
relation_info = json.load(open('./data/DocRED/rel_info.json'))
entity = entity_info[0]['vertexSet']
edges = []
for relation in relations:
    attribute = {}
    source = entity[relation['h_idx']][0]['name']
    attribute['relation'] = relation_info[relation['r']]
    predict = entity[relation['t_idx']][0]['name']
    edges.append((source, predict, attribute))
G = nx.MultiDiGraph()
G.add_edges_from(edges)
show_graph(G)
