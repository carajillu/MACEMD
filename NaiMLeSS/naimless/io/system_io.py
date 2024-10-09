import yaml
from ase.io import read
from ase.cell import Cell
from ase.atoms import Atoms
import os
from typing import Dict, List, Union

def check_yaml(yaml_file: str) -> Dict[str, any]:
    """
    Parse the YAML configuration file and extract the system settings.

    Args:
        yaml_file (str): Path to the YAML configuration file.

    Returns:
        Dict[str, any]: Dictionary containing the system configuration.

    Raises:
        ValueError: If the celltype is unsupported or if initial_structures is invalid.
    """
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    
    system = config.get('system', {})
    
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