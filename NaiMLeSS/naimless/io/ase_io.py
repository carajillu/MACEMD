import yaml
from ase.io import read
from ase.cell import Cell
from ase.atoms import Atoms
import os
from typing import Dict, List, Union

import yaml
from typing import Dict, Any

def check_yaml(yaml_file: str) -> Dict[str, Any]:
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

    if 'io' not in config:
        raise KeyError("The 'io' key was not found in the YAML file.")
    if 'ase_io' not in config['io']:
        raise KeyError("The 'ase_io' key was not found under the 'io' section in the YAML file.")
    
    return config['io']['ase_io']

def check_dictionary(system: Dict[str, any]) -> Dict[str, any]:
    """
    Check and set default values for the system dictionary.

    Args:
        system (Dict[str, any]): Dictionary containing system configuration.

    Returns:
        Dict[str, any]: Validated and updated system configuration.

    Raises:
        ValueError: If initial_structures is invalid.
    """
    # Set default values
    system.setdefault('pbc', True)
    system.setdefault('cellvectors', [10, 10, 10])
    
    if isinstance(system.get('initial_structures'), str):
        system['initial_structures'] = [system['initial_structures']]
    elif not isinstance(system.get('initial_structures'), list):
        raise ValueError("initial_structures must be a string or a list of strings")
    
    return system

def read_structure(structure_path: str, system: Dict[str, any]) -> Atoms:
    """
    Read a structure file and apply system settings.

    Args:
        structure_path (str): Path to the structure file.
        system (Dict[str, any]): Dictionary containing system configuration.

    Returns:
        Atoms: ASE Atoms object with applied system settings.
    """
    atoms = read(structure_path)
    atoms.pbc = system['pbc']
    atoms.cell = system['cellvectors']
    atoms.center()
    basename = os.path.basename(structure_path).split('.')[0]
    atoms.info["name"] = basename
    return atoms

def read_structures(system: Dict[str, any]) -> List[Atoms]:
    """
    Read all structure files specified in the system configuration.

    Args:
        system (Dict[str, any]): Dictionary containing system configuration.

    Returns:
        List[Atoms]: List of ASE Atoms objects.

    Raises:
        FileNotFoundError: If a specified structure file is not found.
    """
    structures = []
    for path in system['initial_structures']:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Structure file not found: {path}")
        structures.append(read_structure(path, system))
    return structures

def main(yaml_file: str) -> List[Atoms]:
    """
    Main function to read the YAML configuration and process all structures.

    Args:
        yaml_file (str): Path to the YAML configuration file.

    Returns:
        List[Atoms]: List of processed ASE Atoms objects.
    """
    system = check_yaml(yaml_file)
    system = check_dictionary(system)
    structures = read_structures(system)
    for i, atoms in enumerate(structures):
        print(f"Read structure {i+1} from {system['initial_structures'][i]}")
        print(f"PBC: {atoms.pbc}")
        print(f"Cell: {atoms.cell}")
    return structures

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Read structures and setup from YAML file")
    parser.add_argument("-c","--config", help="Path to the YAML configuration file",default="example.yml")
    args = parser.parse_args()
    
    main(args.config)