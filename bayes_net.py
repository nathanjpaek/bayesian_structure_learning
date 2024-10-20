# G - graph structure 
# r_i - number of instantiations of x_i
# q_i - number of parental instantiations of x_i
# m_ijk - number of times x_i takes on kth instantiation and parents take on jth instantiation
# P(G|D) proportional P(G)P(D|G)

import sys
import networkx as nx
import numpy as np
import pandas as pd
import csv
import math
from itertools import product


def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


# m_ijk - number of times x_i takes on kth instantiation and parents take on jth instantiation
# compute m_ijk for a given x_i in the network
def calculate_m_ijk(df, dag, x_i, rv_dict):
    
    r_i, values_xi = calculate_r_i(x_i, rv_dict)
    value_k_to_k_index = {value_k: k for k, value_k in enumerate(values_xi)}
    parents_of_xi = list(dag.predecessors(x_i))
    q_i, possible_parent_instantiations = calculate_q_i(parents_of_xi, rv_dict)

    if q_i == 1: # x_i has no parents
        m_ijk = np.zeros(r_i, dtype=int)
        counts = df[x_i].value_counts()
        for value_k, count in counts.items():
            k_index = value_k_to_k_index[value_k]
            m_ijk[k_index] = count
    else:
        parent_instantiation_to_j = {parent_values_j: j for j, parent_values_j in enumerate(possible_parent_instantiations)}
        m_ijk = np.zeros((q_i, r_i), dtype=int)

        # Group data by parents and x_i
        grouped = df.groupby(parents_of_xi + [x_i]).size()
        for index_tuple, count in grouped.items():
            parent_values = tuple(index_tuple[:-1])  # Parent values as a tuple
            x_i_value = index_tuple[-1]             # Value of x_i
            j_index = parent_instantiation_to_j[parent_values]
            k_index = value_k_to_k_index[x_i_value]
            m_ijk[j_index, k_index] = count

    return m_ijk, q_i, r_i


# q_i - number of parental instantiations of x_i
def calculate_q_i(parents_of_xi, rv_dict):
    if not parents_of_xi:
        q_i = 1
        possible_parent_instantiations = []
    else:
        values_parents = [rv_dict[parent] for parent in parents_of_xi]
        possible_parent_instantiations = list(product(*values_parents))
        q_i = len(possible_parent_instantiations)
    return q_i, possible_parent_instantiations


# r_i - number of instantiations of x_i
def calculate_r_i(x_i, rv_dict):
    values_xi = rv_dict[x_i]
    r_i = len(values_xi)
    return r_i, values_xi


def calculate_theta_ijk():
    # CODE HERE
    return 


def map_rvs_to_possible_values(infile):
    df = pd.read_csv(infile)
    rv_names = list(df)
    rv_dict = {}
    for name in rv_names:
        rv_dict[name] = df[name].unique()
    return rv_dict


def get_score(dag, df, rv_dict):
    score = 0
    for x_i in dag.nodes():
        parents_of_xi = list(dag.predecessors(x_i))
        m_ijk, q_i, r_i = calculate_m_ijk(df, x_i, parents_of_xi, rv_dict)
        
        # For variables with no parents
        if q_i == 1:
            m_ij = np.sum(m_ijk)
            score += math.lgamma(r_i) - math.lgamma(r_i + m_ij)
            for k in range(r_i):
                score += math.lgamma(1 + m_ijk[k]) - math.lgamma(1)
        else:
            for j in range(q_i):
                m_ij = np.sum(m_ijk[j, :])
                score += math.lgamma(r_i) - math.lgamma(r_i + m_ij)
                for k in range(r_i):
                    score += math.lgamma(1 + m_ijk[j, k]) - math.lgamma(1)
    return score



def compute(infile, outfile):
    # rv_dict = map_rvs_to_possible_values(infile)
    df = pd.read_csv(infile)
    rv_list = list(df.columns)
    dag = nx.DiGraph(name=infile)

    for rv in rv_list: # create a base graph from the rvs read from file
        dag.add_node(rv)

    # optimize the graph
    init_score = get_score(dag, df)
    best_dag = dag.copy()
    max_iterations = 30
    i = 0
    for node in best_dag.nodes():
        # greedy iteration over the graph looking for a better shape than current

    pass


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()