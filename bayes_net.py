import sys
import networkx as nx
import numpy as np
import pandas as pd
import csv

def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))

# m_ijk - number of times x_i takes on kth instantiation and parents take on jth instantiation
def calculate_m_ijk():

    return 

def calculate_theta_ijk():
    
    return 

# q_i - number of parental instantiations of x_i
def calculate_q_i():
    
    return 

# r_i - number of instantiations of x_i
def calculate_r_i():
    
    return 


def map_rvs_to_possible_values(infile):
    df = pd.read_csv(infile)
    rv_names = list(df)
    rv_dict = {}
    for name in rv_names:
        rv_dict[name] = df[name].unique()
    return rv_dict


def get_parents(rv, dag):
    return dag.predecessors(rv)

def get_score():
    return 


def compute(infile, outfile):
    # rv_dict = map_rvs_to_possible_values(infile)
    df = pd.read_csv(infile)
    rv_list = list(df.columns)
    init_dag = nx.DiGraph(name=infile)

    for rv in rv_list: # create a base graph from the rvs read from file
        init_dag.add_node(rv)

    # optimize the graph

    # G - graph structure 
    # r_i - number of instantiations of x_i
    # q_i - number of parental instantiations of x_i
    # m_ijk - number of times x_i takes on kth instantiation and parents take on jth instantiation
    # P(G|D) proportional P(G)P(D|G)
    

    pass


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()