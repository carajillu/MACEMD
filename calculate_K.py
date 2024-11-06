import numpy as np
import argparse
from ase.io import read

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--trajectories",nargs="+",help="Path to the trajectories")
    parser.add_argument("--atom0",type=int,help="First atom",default=0)
    parser.add_argument("--atom1",type=int,help="Second atom",default=1)
    parser.add_argument("--R_0",type=float,help="Reference distance",default=0.3)
    parser.add_argument("--delta_R",type=float,help="deltaR",default=1)
    args=parser.parse_args()
    return args

def S_on(m):
    if m < 0:
        return 0
    elif 0 <= m <= 1:
        return 3*m**4 - 2*m**6
    else:
        return 1
   
def S_off(m):
    if m < 0:
        return 1
    elif 0 <= m <= 1:
        return 3*(m-1)**4 - 2*(m-1)**6
    else:
        return 0
    
def stack_traj(traj_paths):
    trj=[]
    for path in traj_paths:
        trj=trj+read(path,":")
    return trj
    
if __name__ == "__main__":
    args=parse_args()
    print(args)

    trj=stack_traj(args.trajectories)

    sum_S_on=0
    sum_S_off=0
    for frame in trj:
        r=frame.get_distance(args.atom0,args.atom1)
        m=(r-args.R_0)/args.delta_R
        S_on_val=S_on(m)
        S_off_val=S_off(m)
        sum_S_on+=S_on_val
        sum_S_off+=S_off_val
    print(f"Sum S_on = {sum_S_on}, Sum S_off = {sum_S_off}")
    K=sum_S_on/sum_S_off
    print(f"K = {K}")