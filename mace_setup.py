import argparse
from ase.io import read, write
import numpy as np
from aseMolec import anaAtoms as aa
import random


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--ref', nargs="?", help="XYZ file containing the reference energies (default=energy-pos-1.xyz)",default="energy-pos-1.xyz")
    parser.add_argument('-p','--pos', nargs="?", help="XYZ file containing MD positions (default=nvt-pos-1.xyz)",default="nvt-pos-1.xyz")
    parser.add_argument('-f','--frc', nargs="?", help="XYZ file containing MD forces (default=nvt-frc-1.xyz)",default="nvt-frc-1.xyz")
    parser.add_argument('-o','--output', nargs="?", help="Base name for the training and test sets",default="dataset.xyz")
    parser.add_argument('--split', nargs="?", type=float,help="Proportion of the dataset to be used as test set (default=0.1)",default=0.1)
    parser.add_argument('-c','--mace_config', nargs="?", help="MACE config file",default="mace.yml")
    args=parser.parse_args()
    return args 


def get_cell_vectors(cell_path):
    cell_vectors=[]
    with open(cell_path,"r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            line=line.split()
            vector=" ".join([line[2:11]])
            cell_vectors.append(vector)
    return cell_vectors

def prep_reference(ref_path):
    dataset=read(ref_path,":")
    for atoms in dataset:
        atoms.info["config_type"]="IsolatedAtom"
    return dataset

def prep_db(positions_path, forces_path):
    pos=read(positions_path,":")
    frc=read(forces_path,":")
    aa.wrap_molecs(pos, prog=False)
    for i in range(0,len(pos)):
        pos[i].arrays["frc"]=frc[i].arrays["positions"]
    return pos 

def split_dataset(pos, split):
    random.shuffle(pos)
    threshold_idx=int(len(pos)*split)
    test=pos[0:threshold_idx]
    training=pos[threshold_idx:]
    return training, test


def train_mace(config_file_path):
    return 0
if __name__=="__main__":

    args=parse()
    print(args)

    reference=prep_reference(args.ref)
    db=prep_db(args.pos,args.frc)
    
    training_set,test_set=split_dataset(db,args.split)
    training_set_with_ref=reference+training_set

    write(filename=f"{args.output}_test.xyz",images=test_set,format="extxyz")
    write(filename=f"{args.output}_training.xyz",images=training_set_with_ref,format="extxyz")
