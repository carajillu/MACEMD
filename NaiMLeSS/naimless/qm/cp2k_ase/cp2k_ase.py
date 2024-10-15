import argparse
import yaml
from typing import Dict, Any
import os
from ase import Atoms
from ase.calculators.cp2k import CP2K
import platform
import numpy as np
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Run CP2K with ASE")
    parser.add_argument("-c", "--config", type=str, help="Path to the YAML configuration file")
    return parser.parse_args()

def load_yml(yaml_file: str) -> Dict[str, Any]:
    """
    Parse the YAML configuration file and extract the system settings.

    Args:
        yaml_file (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Dictionary containing the system configuration.

    Raises:
        FileNotFoundError: If the YAML file is not found.
        yaml.YAMLError: If there's an error parsing the YAML file.
        KeyError: If the 'system' key is not found in the YAML file.
    """
    try:
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"The configuration file '{yaml_file}' was not found.")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing the YAML file: {e}")
    if 'qm' not in config:
        raise KeyError("The 'qm' key was not found in the YAML file.")
    if 'cp2k_ase' not in config['qm']:
        raise KeyError("The 'cp2k_ase' key was not found under the 'qm' section in the YAML file.")
    
    qm_config = config['qm']['cp2k_ase']
    return qm_config

def check_config(qm_config: Dict[str, Any]) -> Dict[str, Any]:
    qm_config.setdefault('bin', 'cp2k.psmp')
    if not os.path.exists(qm_config['bin']):
        raise FileNotFoundError(f"The executable '{qm_config['bin']}' does not exist.")
    if not os.access(qm_config['bin'], os.X_OK):
        raise PermissionError(f"The executable '{qm_config['bin']}' is not executable.")
    
    qm_config.setdefault('lib', '/path/to/libcp2k.dylib')
    if not os.path.exists(qm_config['lib']):
        raise FileNotFoundError(f"The library '{qm_config['lib']}' does not exist.")
    
    qm_config.setdefault('force_eval', 'feval.in')
    if not os.path.exists(qm_config['force_eval']):
        raise FileNotFoundError(f"The force_eval file '{qm_config['force_eval']}' does not exist.")
    if not os.access(qm_config['force_eval'], os.R_OK):
        raise PermissionError(f"The force_eval file '{qm_config['force_eval']}' is not accessible.")

    qm_config.setdefault('computing', {})
    qm_config['computing'].setdefault('omp_num_threads', 1)
    qm_config['computing'].setdefault('mpi_num_processes', 4)

    return qm_config

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

def get_calculator(qm_config: Dict[str, Any]) -> CP2K:
    lib_dir = os.path.dirname(qm_config['lib'])
    
    # Set library path for both Linux and macOS
    if platform.system() == "Darwin":  # macOS
        os.environ['DYLD_LIBRARY_PATH'] = f"{lib_dir}:{os.environ.get('DYLD_LIBRARY_PATH', '')}"
    else:  # Linux and others
        os.environ['LD_LIBRARY_PATH'] = f"{lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"

    # Create a wrapper script for cp2k_shell
    wrapper_script = f"""#!/bin/bash
export DYLD_LIBRARY_PATH={lib_dir}:$DYLD_LIBRARY_PATH
export LD_LIBRARY_PATH={lib_dir}:$LD_LIBRARY_PATH
{qm_config['bin']} --shell "$@"
"""

    # Write the wrapper script to a file
    with open('cp2k_shell', 'w') as f:
        f.write(wrapper_script)

    # Make the wrapper script executable
    os.chmod('cp2k_shell', 0o755)
    
    if qm_config['computing']['mpi_num_processes'] > 1:
        cmd = (f"env OMP_NUM_THREADS={qm_config['computing']['omp_num_threads']} "
               f"mpirun -np {qm_config['computing']['mpi_num_processes']} "
               f"./cp2k_shell")
    else:       
        cmd = f"env OMP_NUM_THREADS={qm_config['computing']['omp_num_threads']} ./cp2k_shell"
    
    #print(cmd)
    with open(f"../{qm_config['force_eval']}", 'r') as file:
        inp = file.read()
    CP2K.command = cmd
    return CP2K(inp=inp)
    
    
    try:
        calculator=CP2K(command=cmd)
        #calculator.parameters.clear()
        #calculator.parameters.inp=inp # Need to be able to set the input file
        calculator.parameters.charge=-1
        calculator.parameters.multiplicity=1
    except Exception as e:
        raise e
    return calculator

def get_cp2k_energy(cp2k: CP2K, atoms: Atoms) -> float:
    with cp2k as calc:
        E=calc.get_potential_energy(atoms)
    return E

def get_cp2k_forces(cp2k: CP2K, atoms: Atoms) -> np.ndarray:
    with cp2k as calc:
        F=calc.get_forces(atoms)
    return F
    
def main(qm_config: Dict[str, Any], atoms: Atoms):
    cp2k=get_calculator(qm_config)
    E=get_cp2k_energy(cp2k,atoms)
    print(E)
    F=get_cp2k_forces(cp2k,atoms)
    print(F)

if __name__ == "__main__":
   from __utils__ import *
   args = parse_args()
   qm_config = load_yml(args.config)
   qm_config = check_config(qm_config)
   atoms=create_water_molecule()
   main(qm_config, atoms)
else:
   from .__utils__ import *
