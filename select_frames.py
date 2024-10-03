import argparse
import numpy as np
from ase.io import read, write
def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--fwd_trj",nargs="?",help="Path to the forward trajectory",default="fwd.xyz")
    parser.add_argument("--bwd_trj",nargs="?",help="Path to the backward trajectory",default="bwd.xyz")
    parser.add_argument("--atom_pairs",nargs="+",help="Atom pairs to consider for the selection of frames (example: --atom_pairs \"0 1\" \"2 3\")",default=[[0,1],[2,3]])
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (show warnings)")
    args=parser.parse_args()
    return args

def get_atom_pairs(args):
    atom_pairs=[]
    for pair in args.atom_pairs:
        if isinstance(pair,str):
            pair=pair.split()
            pair=[int(i) for i in pair]
        atom_pairs.append(pair)
    return atom_pairs



def make_histogram(distance_matrix,spacing,extend):
    '''
    returns a histogram object of the distance matrix, with the number of bins defined by the spacing and the minimum
    and maximum values in each dimension, which are extended by the extend parameter.
    '''
    nbins=int((max(distance_matrix)+extend)-(min(distance_matrix)-extend)/spacing)
    hist=np.histogramdd(distance_matrix,bins=spacing,range=((min(distance_matrix)-extend,max(distance_matrix)+extend),)*len(distance_matrix))
    return hist

def get_distance_matrix(atoms,atom_pairs):
    distance_matrix=[]
    for pair in atom_pairs:
        distance_matrix.append(atoms.get_distance(pair[0],pair[1]))
    return distance_matrix

if __name__=="__main__":
    args=parse_args()
    atom_pairs=get_atom_pairs(args)
    print(atom_pairs)
    
    

