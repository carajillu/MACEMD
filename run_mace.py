import argparse
import numpy as np
import sys
import shutil
import os
import yaml
import glob
import importlib
import multiprocessing
import torch
import time
import warnings
warnings.filterwarnings("ignore")
import subprocess

#Set number of openmp threads to 1 and let MPI handle multi-threading
os.environ["OMP_NUM_THREADS"] = "1"

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
    parser.add_argument("--restart", action="store_true", help="Restart the simulation from the last step")
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
    if mace_config['model'] in foundation_models:
        model_class=getattr(model_module,mace_config['model'])
        calculator=model_class(model=mace_config['model_path'],device=device_name)
    else:
        model_class=getattr(model_module,"MACECalculator")
        calculator=model_class(model_paths=[mace_config['model_path']],device=device_name)
    return calculator

def load_dynamics(atoms,mdconfig):
    dynamics=mdconfig['dynamics']
    if dynamics['class'] not in dynamics_classes:
        raise ValueError(f"Invalid dynamics class: {dynamics['class']}")
    dynamics_class=getattr(md_module,dynamics['class'])
    if isinstance(mdconfig['parameters'], dict):
       dyn = dynamics_class(atoms, timestep=mdconfig['timestep'], **mdconfig['parameters'])
    else:
        dyn = dynamics_class(atoms, timestep=mdconfig['timestep'])
    return dyn

def run_dyn(system_name, dyn, nsteps, stride, restart=False):
    print(f"Starting run_dyn for {system_name}")    
    max_snapshots = int(nsteps/stride) + 1
    if restart:
        try:
            trj_file = f"{system_name}.trj.xyz"
            print(f"Attempting to read restart file: {trj_file}")
            try:
                rst_atoms = read(trj_file, ":")
            except Exception as e:
                print(f"file {trj_file} seems corrupted. Attempting to read up to the second last snapshot")
                with open(trj_file,"r") as f:
                    lines=f.readlines()
                    n_atoms=int(lines[0].split()[0])
                    nsnapshots=int(len(lines)/(n_atoms+2))
                    # get the elements of newfile corresponding to all snapshots except the last.
                    newfile=lines[:(nsnapshots)*(n_atoms+2)]
                    # write the newfile to a new file
                with open(trj_file,"w") as f:
                    f.writelines(newfile)
                    # read the newfile
                rst_atoms=read(trj_file, ":")
            nsnapshots = len(rst_atoms)
            print(f"Read {nsnapshots} snapshots from restart file")
            if nsnapshots >= max_snapshots:
                print(f"Simulation of {system_name} is complete. Skipping.")
                return
            else:
                print(f"Restarting {system_name} from snapshot {nsnapshots}")
                nsteps = nsteps - nsnapshots * stride
                dyn.atoms.set_positions(rst_atoms[-1].get_positions())
                dyn.atoms.set_velocities(rst_atoms[-1].get_velocities())
                dyn.atoms.set_cell(rst_atoms[-1].get_cell())
                dyn.atoms.set_pbc(rst_atoms[-1].get_pbc())
        except Exception as e:
            print(f"Error reading restart file: {e}. Starting new simulation.")
    
    print(f"Starting MD simulation for {system_name} on device: {dyn.atoms.calc.device}")
    try:
        dyn.run(steps=nsteps)
        print(f"Completed MD simulation for {system_name}")
    except Exception as e:
        print(f"Error during MD simulation for {system_name}: {e}")
    
    
    print(f"Finished run_dyn for {system_name}")

def process_structure(structure_path, device_name, config,restart=False):
    atoms = read_structure(structure_path)
    calculator = load_calculator(device_name, config['mace'])
    atoms.calc = calculator
    # Set initial velocities
    if isinstance(config['md']['parameters'], dict) and "temperature_K" in config['md']['parameters']:
       MaxwellBoltzmannDistribution(atoms, temperature_K=config['md']['parameters']['temperature_K'])
    else:
       print("No temperature provided, using the default temperature of 300 K")
       MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    # Load dynamics object

    system_name=os.path.basename(structure_path).split(".")[0]
    root_dir = os.getcwd()
    os.makedirs(f"{system_name}", exist_ok=True)
    os.chdir(f"{system_name}")
    print(f"Changed directory to {os.getcwd()}")

    print("Creating dynamics object")
    dyn = load_dynamics(atoms, config['md'])
    print(f"Attaching loggers and snapshot writers")
    
    time_tracker=create_time_tracker(system_name)
    dyn.attach(time_tracker, interval=config['md']['stride'])

    if isinstance(config['cp2k'], dict):
        cp2k_input_name=f"../{config['cp2k']['input']}"
        cp2k_str=""
        print(f"Reading CP2K input file: {cp2k_input_name}")
        with open(cp2k_input_name,"r") as f:
            for line in f:
                cp2k_str+=line
                line=line.split()
                if len(line)==0:
                    continue
                if line[0]=="PROJECT":
                    config['cp2k']['project_name']=line[1]
                if "COORD_FILE_NAME" in line:
                    print(f"Found COORD_FILE_NAME: {line}")
                    config['cp2k']['coord_file_name']=line[1]
        config['cp2k']['input_str']=cp2k_str
        print(config['cp2k'])

        cp2k_run=create_run_cp2k(dyn,config['cp2k'])
        dyn.attach(cp2k_run, interval=config['md']['stride'])

    print_md_snapshot=create_print_md_snapshot(system_name, dyn)
    dyn.attach(print_md_snapshot, interval=config['md']['stride'])

    dyn.attach(MDLogger(dyn, dyn.atoms, 'md.log', header=True, stress=False,
               peratom=True, mode="a"), interval=config['md']['stride'])
    
    run_dyn(system_name, dyn, config['md']['nsteps'], config['md']['stride'],restart)
    os.chdir(root_dir)
    del atoms, calculator, dyn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


'''
Functions that are called during the simulation
'''
def create_print_md_snapshot(system_name, dyn):
    def print_md_snapshot():
        dyn.atoms.wrap()
        filename = f"{system_name}.trj.xyz"
        dyn.atoms.write(filename, append=True)
        filename = f"{system_name}.trj.pdb"
        dyn.atoms.write(filename, append=True)
    return print_md_snapshot

def create_time_tracker(system_name):
    def time_tracker():
        with open("time.log", "a") as f:
            f.write(f"{time.ctime()}\n")
    return time_tracker

def create_run_cp2k(dyn,cp2k_config):
    print("create_run_cp2k")
    def run_cp2k():
        # Get the MACE energy and forces
        mace_energy=dyn.atoms.calc.get_potential_energy()
        mace_forces=dyn.atoms.calc.get_forces()
        # Run a CP2K calculation
        os.makedirs("cp2k_files",exist_ok=True)
        os.chdir("cp2k_files")
        
        with open("cp2k.in","w") as f:
            f.write(cp2k_config['input_str'])
        print(f"Writing coordinates to {cp2k_config['coord_file_name']}")
        dyn.atoms.wrap()
        dyn.atoms.write(cp2k_config['coord_file_name'],append=False)

        subprocess.run(["mpirun", "-np", str(cp2k_config['nprocs']), cp2k_config['exe'], "-i", "cp2k.in", "-o", "cp2k.out"],check=True)
        #subprocess.run(' '.join([cp2k_config['exe'], "-i", "cp2k.in"]),shell=True,check=True)
        # Read the CP2K energy and forces
        cp2k_atoms=read(f"{cp2k_config['project_name']}-pos-1.xyz")
        cp2k_energy=cp2k_atoms.info['E']
        cp2k_atoms=read(f"{cp2k_config['project_name']}-frc-1.xyz")
        cp2k_forces=cp2k_atoms.arrays['positions']
        os.chdir("..")
        #cleanup the cp2k files
        #shutil.rmtree("cp2k_files") # we need to keep the cp2k wavefunction restart files
        # Compare the MACE and CP2K energies
        if (abs(mace_energy-cp2k_energy)>cp2k_config['energy_tol']) or \
           (not np.allclose(mace_forces, cp2k_forces, atol=cp2k_config['force_tol'])):
            print(f"CP2K Energy: {cp2k_energy}, MACE Energy: {mace_energy}")
            print(f"CP2K Forces: {cp2k_forces}, MACE Forces: {mace_forces}")
            print(f"Force mismatch between MACE and CP2K")
            dyn.atoms.info['E']=cp2k_energy
            dyn.atoms.arrays['frc']=cp2k_forces
            dyn.atoms.write("cp2k_snaps.xyz",append=True)
    return run_cp2k
            

def main():
    args=parse_args()

    #read the config file
    config=parse_yml(args.config)
    print(config)


    #get the list of paths to the initial structures
    structure_path_list=glob.glob(config['initial_structures']+"/*.xyz")
    nstructures=len(structure_path_list)

    #get the list of device names
    device_names=config['mace']['devices']
    ndevices=len(device_names)

    # Create a pool of worker processes
    with torch.multiprocessing.Pool(processes=ndevices) as pool:
        # Create a list of arguments for each structure
        args_list = [(structure_path, device_names[i % ndevices], config,args.restart) 
                     for i, structure_path in enumerate(structure_path_list)] 
        # Map the process_structure function to the pool of workers
        pool.starmap(process_structure, args_list)
        pool.close()
        pool.join()



if __name__=="__main__":
      main()