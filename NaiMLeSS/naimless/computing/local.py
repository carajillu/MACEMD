import argparse
import yaml
from typing import Dict, Any

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Configure local computing settings")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to the configuration file")
    return parser.parse_args()

def check_yaml(yaml_file: str) -> Dict[str, Any]:
    """
    Parse the YAML configuration file and extract the local computing settings.

    Args:
        yaml_file (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Dictionary containing the local computing configuration.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    
    local_config = config.get('computing', {}).get('local', {})
    
    # Set default values and check for required fields
    local_config.setdefault('devices', ['cpu'])
    local_config.setdefault('omp_num_threads', 1)
    local_config.setdefault('mpi_num_processes', 1)
    
    # Convert single string device to a list
    if isinstance(local_config['devices'], str):
        local_config['devices'] = [local_config['devices']]
    elif not isinstance(local_config['devices'], list):
        raise ValueError("'devices' must be a string or a list of strings")
    
    if not isinstance(local_config['omp_num_threads'], int) or local_config['omp_num_threads'] < 1:
        raise ValueError("'omp_num_threads' must be a positive integer")
    
    if not isinstance(local_config['mpi_num_processes'], int) or local_config['mpi_num_processes'] < 1:
        raise ValueError("'mpi_num_processes' must be a positive integer")
    
    return local_config

def main():
    args = parse_args()
    local_config = check_yaml(args.config)
    print("Local computing configuration:")
    print(f"Devices: {local_config['devices']}")
    print(f"OpenMP threads: {local_config['omp_num_threads']}")
    print(f"MPI processes: {local_config['mpi_num_processes']}")

if __name__ == "__main__":
    main()
