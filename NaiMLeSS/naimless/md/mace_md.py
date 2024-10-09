import argparse
import yaml
from typing import Dict, Any
from mace import calculators
from ase import Atoms
import ase.md as md
import os
from copy import deepcopy
foundation_models=["mace_off","mace_anicc","mace_mp"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Configure MACE MD settings")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to the configuration file")
    return parser.parse_args()

def check_yaml(yaml_file: str) -> Dict[str, Any]:
    """
    Parse the YAML configuration file and extract the MACE MD settings.

    Args:
        yaml_file (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Dictionary containing the MACE MD configuration.

    Raises:
        FileNotFoundError: If the YAML file is not found.
        yaml.YAMLError: If there's an error parsing the YAML file.
        KeyError: If the 'md' or 'mace' keys are not found in the YAML file.
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
    if 'mace' not in config['md']:
        raise KeyError("The 'mace' key was not found under the 'md' section in the YAML file.")

    mace_config = config['md']['mace']
    mace_config.setdefault('computing', {})
    mace_config['computing'].setdefault('devices', ['cpu'])

    mace_config.setdefault('model', 'mace-off')
    mace_config.setdefault('model_path', 'small')
    mace_config.setdefault('dynamics', {})
    mace_config['dynamics'].setdefault('class', 'Langevin')
    mace_config['dynamics'].setdefault('timestep', 1.0)
    mace_config['dynamics'].setdefault('parameters', {})

    mace_config.setdefault('nsteps', 100)
    mace_config.setdefault('stride', 1)
    mace_config.setdefault('logfile', 'md.log')

    return mace_config

def check_dictionary(mace_config: Dict[str, Any]) -> Dict[str, Any]:
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
    
    # Validate fields
    if not isinstance(mace_config['computing']['devices'], list):
        raise ValueError("'devices' must be a list")
    
    if not isinstance(dynamics['timestep'], (int, float)) or dynamics['timestep'] <= 0:
        raise ValueError("'timestep' must be a positive number")
    
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
        print(dynamics_class)
    except AttributeError:
        raise ValueError(f"Invalid dynamics class: {mace_config['dynamics']['class']}")
    

    dyn=dynamics_class(atoms=atoms,timestep=mace_config['dynamics']['timestep'],\
                       **mace_config['dynamics']['parameters'])
    
    return dyn

def run_md(dyn: Any, nsteps: int) -> None:
    root_name=dyn.atoms.info["name"]
    os.makedirs(root_name,exist_ok=True)
    os.chdir(root_name)
    dyn.run(steps=nsteps)
    os.chdir('..')

def main():
    args = parse_args()
    try:
        mace_config = check_yaml(args.config)
        mace_config = check_dictionary(mace_config)
    except (FileNotFoundError, yaml.YAMLError, KeyError, ValueError) as e:
        print(f"Error: {e}")
    
    print(mace_config)

    atoms = create_water_molecule()
    atoms.calc=return_calculator(mace_config, mace_config['computing']['devices'][0])
    dyn = return_dynamics(mace_config, atoms)
    run_md(dyn,mace_config['nsteps'])

if __name__ == "__main__":
    main()