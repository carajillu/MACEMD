import argparse

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--initial_state",nargs="?",help="Path to the optimized structure of the initial state",default="initial.xyz")
    parser.add_argument("--final_state",nargs="?",help="Path to the optimized structure of the final state",default="final.xyz")
    parser.add_argument("--fwd_trj",nargs="?",help="Path to the forward trajectory",default="fwd.xyz")
    parser.add_argument("--bwd_trj",nargs="?",help="Path to the backward trajectory",default="bwd.xyz")
    parser.add_argument("--atom_pairs",nargs="+",help="Atom pairs to consider for the selection of frames (example: --atom_pairs \"0 1\" \"2 3\")",default=[[0,1],[2,3]])
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (show warnings)")
    args=parser.parse_args()
    return args

def get_atom_pairs(args):
    atom_pairs=[]
    for pair in args.atom_pairs:
        pair=pair.split()
        pair=[int(i) for i in pair]
        atom_pairs.append(pair)
    return atom_pairs


if __name__=="__main__":
    args=parse_args()
    atom_pairs=get_atom_pairs(args)
    print(atom_pairs)