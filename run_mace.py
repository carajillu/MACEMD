import argparse
import numpy as np
import sys
import os
import yaml
import glob
import importlib
import multiprocessing
import torch

# Set the start method to 'spawn'
multiprocessing.set_start_method('spawn', force=True)

# Set the default tensor dtype and device to float32 and cuda if available
if torch.cuda.is_available():
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda')

from ase.io import read
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
foundation_models=["mace_off","mace_anicc","mace_mp"]

md_module=importlib.import_module("ase.md")
dynamics_classes=md_module.__all__[1:] # Should we modify the ASE API so that types of md are classified in ensembles?


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("-c","--config",nargs="?",help="Path to the config file",default="example.yml")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (show warnings)")
    args=parser.parse_args()
    return args

def parse_yml(yml_path):
    with open(yml_path,"r") as f:
        config=yaml.load(f,Loader=yaml.FullLoader)
    return config

def read_structure(structure_path):
    atoms=read(structure_path)
    atoms.cell=[10,10,10]
    atoms.pbc=True
    atoms.center()
    return atoms

def load_calculator(device_name,mace_config):
    model_module=importlib.import_module(f"mace.calculators")
    if mace_config["model"] in foundation_models:
        model_class=getattr(model_module,mace_config["model"])
        calculator=model_class(model=mace_config["model_path"],device=device_name)
    else:
        model_class=getattr(model_module,"MACECalculator")
        calculator=model_class(model_path=mace_config["model"],device=device_name)
    return calculator

def load_dynamics(atoms,mdconfig):
    dynamics=mdconfig["dynamics"]
    if dynamics['class'] not in dynamics_classes:
        raise ValueError(f"Invalid dynamics class: {dynamics['class']}")
    dynamics_class=getattr(md_module,dynamics['class'])
    if isinstance(mdconfig["parameters"], dict):
       dyn = dynamics_class(atoms, timestep=mdconfig["timestep"], **mdconfig["parameters"])
    else:
        dyn = dynamics_class(atoms, timestep=mdconfig["timestep"])
    return dyn

def run_dyn(dyn,nsteps,stride):
    root_dir=os.getcwd()
    os.makedirs(f"{dyn.atoms.symbols}",exist_ok=True)
    os.chdir(f"{dyn.atoms.symbols}")
    def print_md_snapshot(): #that has to go somewhere else
        filename=f"{dyn.atoms.symbols}.trj.xyz"
        dyn.atoms.write(filename,append=True)
    dyn.attach(print_md_snapshot,interval=stride)
    dyn.attach(MDLogger(dyn, dyn.atoms, 'md.log', header=True, stress=False,
               peratom=True, mode="a"), interval=stride)
    print(f"Running {dyn.atoms.symbols} MD simulation on device: {dyn.atoms.calc.device}")
    dyn.run(steps=nsteps)
    os.chdir(root_dir)
        
def process_structure(structure_path, device_name, config):
    atoms = read_structure(structure_path)
    calculator = load_calculator(device_name, config["mace"])
    atoms.calc = calculator
    # Set initial velocities
    if config["md"]["temperature_K"] is not None:
       MaxwellBoltzmannDistribution(atoms, temperature_K=config["md"]["temperature_K"])
    else:
       print("No temperature provided, using the default temperature of 300 K")
       MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    # Load dynamics object
    dyn = load_dynamics(atoms, config["md"])
    
    run_dyn(dyn, config["md"]["nsteps"], config["md"]["stride"])

def main():
    args=parse_args()

    #read the config file
    config=parse_yml(args.config)
    print(config)


    #get the list of paths to the initial structures
    structure_path_list=glob.glob(config["initial_structures"]+"/*.xyz")
    nstructures=len(structure_path_list)

    #get the list of device names
    device_names=config["mace"]["devices"]
    ndevices=len(device_names)

    # Create a pool of worker processes
    with torch.multiprocessing.Pool(processes=ndevices) as pool:
        # Create a list of arguments for each structure
        args_list = [(structure_path, device_names[i % ndevices], config) 
                     for i, structure_path in enumerate(structure_path_list)] 
        # Map the process_structure function to the pool of workers
        pool.starmap(process_structure, args_list)
        pool.close()
        pool.join()



if __name__=="__main__":
      main()