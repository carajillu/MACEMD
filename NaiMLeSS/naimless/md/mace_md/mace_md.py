import argparse
import yaml
from typing import Dict, Any
from mace import calculators
from ase import Atoms
from ase.io import read
import ase.md as md
import os
import importlib
import time
import numpy as np
foundation_models=["mace_off","mace_anicc","mace_mp"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Configure MACE MD settings")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to the configuration file")
    parser.add_argument("-i", "--input", default=None, help="Path to the input file")
    parser.add_argument("-r", "--restart", action="store_true", help="Restart the simulation from the last step")
    return parser.parse_args()

def load_yml(yaml_file: str) -> Dict[str, Any]:
    """
    Parse the YAML configuration file and extract the MACE MD settings.

    Args:
        yaml_file (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Dictionary containing the MACE MD configuration.

    Raises:
        FileNotFoundError: If the YAML file is not found.
        yaml.YAMLError: If there's an error parsing the YAML file.
        KeyError: 'mace_md' section not found in the YAML file.
    """
    try:
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"The configuration file '{yaml_file}' was not found.")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing the YAML file: {e}")
    
    if 'md' not in config:
        raise KeyError("The 'md' key was not found in the YAML file.")
    if 'mace_md' not in config['md']:
        raise KeyError("The 'mace_md' key was not found under the 'md' section in the YAML file.")
    
    mace_config = config['md']['mace_md']

    return mace_config

def check_config(mace_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check and set default values for the MACE MD dictionary.

    Args:
        mace_config (Dict[str, Any]): Dictionary containing MACE MD configuration.

    Returns:
        Dict[str, Any]: Validated and updated MACE MD configuration.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    # Set default values and check for required fields
    mace_config.setdefault('computing', {})
    mace_config['computing'].setdefault('devices', ['cpu'])
    if isinstance(mace_config['computing']['devices'],str):
        mace_config['computing']['devices']=[mace_config['computing']['devices']]

    mace_config.setdefault('model', 'mace-off')
    mace_config.setdefault('model_path', 'small')
    mace_config.setdefault('dynamics', {})
    
    dynamics = mace_config['dynamics']
    dynamics.setdefault('class', 'VelocityVerlet')
    dynamics.setdefault('timestep', 1.0)
    dynamics.setdefault('parameters', {})

    mace_config.setdefault('nsteps', 100)
    mace_config.setdefault('stride', 1)
    
    return mace_config

def create_water_molecule() -> Atoms:
    """
    Create and return an ASE Atoms object representing a water molecule.
    This will be passed to the MACE calculator when called as a script. For testing purposes.
    Returns:
        Atoms: ASE Atoms object of a water molecule.
    """
    water = Atoms('H2O',
                  positions=[[0, 0, 0],
                             [0.758602, 0.585882, 0],
                             [-0.758602, 0.585882, 0]],
                  cell=[10, 10, 10],
                  pbc=True)
    water.center()
    water.info['name'] = 'water'
    return water

def return_calculator(mace_config: Dict[str, Any], device_name: str) -> Any:
    """
    Return the MACE calculator based on the provided configuration.

    Args:
        mace_config (Dict[str, Any]): Dictionary containing MACE MD configuration.

    Returns:
        Any: MACE calculator object.
    """
    
    if mace_config['model'] in foundation_models:
        model_class=getattr(calculators,mace_config['model'])
        calculator=model_class(model=mace_config['model_path'],device=device_name)
    else:
        model_class=getattr(calculators,"MACECalculator")
        calculator=model_class(model_path=mace_config['model'],device=device_name)

    return calculator 

def return_dynamics(mace_config: Dict[str, Any], atoms: Atoms) -> Any:
    """
    Return the MACE dynamics based on the provided configuration.

    Args:
        mace_config (Dict[str, Any]): Dictionary containing MACE MD configuration.

    Returns:
        Any: MACE dynamics object.
    """
    try:
        dynamics_class = getattr(md, mace_config['dynamics']['class'])
    except AttributeError:
        raise ValueError(f"Invalid dynamics class: {mace_config['dynamics']['class']}")
    

    dyn=dynamics_class(atoms=atoms,timestep=mace_config['dynamics']['timestep'],\
                       **mace_config['dynamics']['parameters'])
    
    return dyn

def restart_md(dyn: Any, mace_config: Dict[str, Any]) -> None:
    """
    Restart the MD simulation from the last step.
    """
    nsteps=mace_config['nsteps']
    system_name = dyn.atoms.info["name"]
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
        max_snapshots = mace_config['nsteps']//mace_config['stride']
        print(f"Read {nsnapshots} snapshots from restart file")
        if nsnapshots >= max_snapshots:
            print(f"Simulation of {system_name} is complete. Skipping.")
            nsteps=0
        else:
            print(f"Restarting {system_name} from snapshot {nsnapshots}")
            nsteps = mace_config['nsteps'] - nsnapshots * mace_config['stride']
            dyn.atoms.set_positions(rst_atoms[-1].get_positions())
            dyn.atoms.set_velocities(rst_atoms[-1].get_velocities())
            dyn.atoms.set_cell(rst_atoms[-1].get_cell())
            dyn.atoms.set_pbc(rst_atoms[-1].get_pbc())
    except Exception as e:
        print(f"Error reading restart file: {e}. Starting new simulation.")
        nsteps=mace_config['nsteps']
    return dyn,nsteps

def run_md(dyn: Any, mace_config: Dict[str, Any], restart: bool = False) -> None:
    """
    Run the MD simulation.

    Args:
        dyn (Any): MACE dynamics object.
        mace_config (Dict[str, Any]): Dictionary containing MACE MD configuration.
        restart (bool): Whether to restart the simulation from the last step.
    """
    root_name=dyn.atoms.info["name"]
    os.makedirs(root_name,exist_ok=True)
    os.chdir(root_name)
    if restart:
        print(f"Restarting simulation of {root_name}")
        dyn,nsteps = restart_md(dyn,mace_config)
    else:
        print(f"Starting new simulation of {root_name} in device {dyn.atoms.calc.device}")
        nsteps=mace_config['nsteps']
    dyn.run(steps=nsteps)
    os.chdir('..')


def create_print_md_snapshot(system_name, dyn):
    def print_md_snapshot():
        filename = f"{system_name}.trj.xyz"
        dyn.atoms.wrap()
        dyn.atoms.write(filename, append=True)
        filename = f"{system_name}.trj.pdb"
        dyn.atoms.write(filename, append=True)
    return print_md_snapshot

def create_time_tracker(system_name):
    initial_time = time.time()
    def time_tracker():
        elapsed_time = time.time() - initial_time
        with open("time.log", "a") as f:
            f.write(f"{time.ctime()} {elapsed_time:.2f}\n")
    return time_tracker

def create_run_qm(qm_config,dyn):
    def run_qm():
        qm_module = importlib.import_module(qm_config["module"])
        calc=qm_module.get_calculator(qm_config)
        with calc as calc:
            #for key in calc.parameters:
            #    print(f"{key}: {calc.parameters[key]}")
            atoms=dyn.atoms
            atoms.wrap()
            try:
                E=calc.get_potential_energy(atoms)
                F=calc.get_forces(atoms)
            except Exception as e:
                print(f"QM calculation failed with error: {e}. Setting energy and forces to zero.")
                E=0
                F=np.zeros_like(atoms.get_forces())
        dyn.atoms.info['E']=E
        dyn.atoms.arrays['frc']=F
    return run_qm

def main(atoms: Atoms, mace_config: Dict[str, Any], qm_config: Any = None, restart: bool = False,device_id:int=0) -> None:
    """
    Main function to run the MD simulation.

    Args:
        atoms (Atoms): ASE Atoms object.
        mace_config (Dict[str, Any]): Dictionary containing MACE MD configuration.
        restart (bool): Whether to restart the simulation from the last step.
    """
    atoms.calc=return_calculator(mace_config, mace_config['computing']['devices'][device_id])
    dyn = return_dynamics(mace_config, atoms)
    #attach QM calls (needs to go first because it adds info to dyn.atoms)
    if qm_config is not None:
        run_qm = create_run_qm(qm_config,dyn)
        dyn.attach(run_qm, interval=mace_config['stride'])
    #attach snapshot printing
    print_md_snapshot = create_print_md_snapshot(f"{atoms.info['name']}", dyn)
    dyn.attach(print_md_snapshot, interval=mace_config['stride'])
    #attach time tracking
    time_tracker = create_time_tracker(f"{atoms.info['name']}.time.log")
    dyn.attach(time_tracker, interval=mace_config['stride'])

    run_md(dyn,mace_config,restart=restart)

if __name__ == "__main__":
    #from __utils__ import *

    args = parse_args()
    try:
        mace_config = load_yml(args.config)
        mace_config = check_config(mace_config)
    except (FileNotFoundError, yaml.YAMLError, KeyError, ValueError) as e:
        print(f"Error: {e}")
    
    print(mace_config)
    if args.input is not None:
        atoms = read(args.input)
        atoms.info["name"]=os.path.basename(args.input).split(".")[0]
        atoms.center()
        atoms.cell=[10,10,10]
        atoms.pbc=True
    else:
        atoms = create_water_molecule()
    
    if args.restart:
        os.makedirs(atoms.info["name"],exist_ok=True)
        os.chdir(atoms.info["name"])
        atoms.write(f"{atoms.info['name']}.trj.xyz")
        os.chdir('..')
   
    main(atoms,mace_config,qm_config=None,restart=args.restart)

#else:
#    #from .__utils__ import *
