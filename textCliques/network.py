# Copyright(C) Shruti Phadke
import networkx as nx
import numpy as np
from collections import defaultdict, Counter


def create_edges(tot_nonzero,nonzero_entries):
    edges = []
    for it in range(tot_nonzero):
        tup = sorted([nonzero_entries[0][it], nonzero_entries[1][it]])
        edges.append((tup[0], tup[1]))
        
    edges = list(set(edges))
    return edges


def create_network(edges, min_degree =2, max_degree=500):
    G = nx.Graph()
    for e in edges:
        G.add_edge(e[0], e[1])
        
    ndegree = nx.degree(G)
    valid_nodes = []
    for k in dict(ndegree).keys():
        #if ndegree[k] > 14 and ndegree[k] < 500:
        if ndegree[k] < 500:
            valid_nodes.append(k)
            
    invalid_nodes = list(set(list(G.nodes)).difference(set(valid_nodes)))
    for i in invalid_nodes:
        G.remove_node(i)
        
    return G



def calculate_cliques(G):
    sub_graphs = nx.connected_components(G)
    clique2nodes = defaultdict(list)
    clique_number = 0
    
    for sg in sub_graphs:
        clique2nodes[clique_number] = list(sg)
        clique_number = clique_number + 1
        
    node2cliques = defaultdict()
    for k in clique2nodes.keys():
        for n in clique2nodes[k]:
            node2cliques[n] = k
            
    return clique2nodes, node2cliques

