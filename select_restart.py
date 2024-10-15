import argparse
from ase.io import read,write
import numpy as np
def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("-i","--trj",nargs="?",help="Path to the trajectory",default="trj.xyz")
    parser.add_argument("-r","--reference",nargs="?",help="Path to the reference structure",default="ref.xyz")
    parser.add_argument("-o","--output",nargs="?",help="Path to the output file",default="restart.xyz")
    parser.add_argument("--atom_pairs",nargs="+",help="Atom pairs to consider for the selection of frames (example: --atom_pairs \"0 1\" \"2 3\")",default=[[0,1],[2,3]])
    parser.add_argument("-d","--debug",action="store_true",help="Enable debug mode (show warnings)")
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

def get_distance_vector(atoms,atom_pairs):
    distance_vector=[]
    for pair in atom_pairs:
        distance_vector.append(atoms.get_distance(pair[0],pair[1]))
    return np.array(distance_vector)

def get_distance_matrix(trj,atom_pairs):
    distance_matrix=[]
    for atoms in trj:
        distance_vector=[]
        for pair in atom_pairs:
            distance_vector.append(atoms.get_distance(pair[0],pair[1]))
        distance_matrix.append(distance_vector)
    return np.array(distance_matrix)

def get_restart_idx(ref_distance_vector, trj_distance_matrix):
    distances = np.linalg.norm(trj_distance_matrix - ref_distance_vector, axis=1)
    print(0,distances[0])
    print(np.argmin(distances),np.min(distances))
    return np.argmin(distances)

if __name__=="__main__":
    args=parse_args()
    print(args)

    trj=read(args.trj,":")
    ref=read(args.reference)

    atom_pairs=get_atom_pairs(args)
    ref_distance_vector=get_distance_vector(ref,atom_pairs)
    print(ref_distance_vector)

    trj_distance_matrix=get_distance_matrix(trj,atom_pairs)
    print(trj_distance_matrix)

    restart_idx=get_restart_idx(ref_distance_vector,trj_distance_matrix)
    print(restart_idx)

    restart=trj[restart_idx]
    write(args.output,restart)

