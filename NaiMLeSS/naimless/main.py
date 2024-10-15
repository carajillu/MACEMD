import argparse
import glob
import os
import torch
import multiprocessing
import yaml
import importlib
import warnings
import sys
warnings.filterwarnings("ignore")

multiprocessing.set_start_method('spawn', force=True)

def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments including config file path,
        parallel execution flag, and restart flag.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", nargs="?", help="Path to the config file", default="config.yaml")
    parser.add_argument("-p", "--parallel", action="store_true", help="Run the simulation in parallel")
    parser.add_argument("--restart", action="store_true", help="Restart the simulation from the last step")
    args = parser.parse_args()
    return args

def load_yml(path):
    """
    Parse the YAML configuration file.

    Args:
        path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        KeyError: If 'io' or 'md' keys are not found in the YAML file.
    """
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def check_config(config):
    """
    Check and process the configuration dictionary.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        dict: Processed configuration dictionary with imported modules.

    Raises:
        ValueError: If a section is empty or has more than one key.
        ModuleNotFoundError: If a required module is not found.
    """
    if 'io' not in config:
        raise KeyError("The 'io' key was not found in the YAML file.")
    if 'md' not in config:
        raise KeyError("The 'md' key was not found in the YAML file.")
    #if 'ml' not in config:
    #    raise KeyError("The 'ml' key was not found in the YAML file.")
    #if 'qm' not in config:
    #      raise KeyError("The 'qm' key was not found in the YAML file.")

    for section in config.keys():
        if len(config[section].keys()) == 0:
            raise ValueError(f"The {section} section is empty.")
        elif len(config[section].keys()) > 1:
            raise ValueError(f"The {section} section has more than one key. Only one key is allowed per outer section")
        try:
            section_module = importlib.import_module(f"naimless.{section}")
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f"The {section} module was not found.")
        
        key = list(config[section].keys())[0]
        try:
            key_module = importlib.import_module(f"naimless.{section}.{key}")
            config[section][key]["module"] = f"naimless.{section}.{key}"
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f"The {key} module was not found under the {section} section.")
        config[section] = config[section][key]
        config[section] = key_module.check_config(config[section])
    return config

def run_md(structure_path, device_name, config, restart=False,device_id=0):
    """
    Run a single molecular dynamics simulation.

    Args:
        structure_path (str): Path to the input structure file.
        device_name (str): Name of the device to run the simulation on.
        config (dict): Configuration dictionary.
        restart (bool, optional): Whether to restart the simulation. Defaults to False.
    """
    io_module = importlib.import_module(config["io"]["module"])
    atoms = io_module.read_structure(structure_path, config["io"])

    md_module = importlib.import_module(config["md"]["module"])
    md_module.main(atoms, config["md"], config["qm"], restart=restart,device_id=device_id)

def md_parallel_batch(config, restart=False):
    """
    Run molecular dynamics simulations in parallel.

    Args:
        config (dict): Configuration dictionary.
        restart (bool, optional): Whether to restart the simulations. Defaults to False.
    """
    ndevices = len(config["md"]["computing"]["devices"])
    print(f"Number of devices: {ndevices}")
    device_names = config["md"]["computing"]["devices"]
    print(f"Devices: {device_names}")
    structure_path_list = config["io"]["initial_structures"]
    print(f"Number of structures: {len(structure_path_list)}")
    with torch.multiprocessing.Pool(processes=ndevices) as pool:
        args_list = [(structure_path, device_names[i % ndevices], config, restart,i) 
                     for i, structure_path in enumerate(structure_path_list)]
        #print(args_list)
        pool.starmap(run_md, args_list)

def main():
    """
    Main function to run the molecular dynamics simulation(s).
    
    Parses arguments, reads and processes the configuration, and runs the simulation(s)
    either in parallel or serial mode based on the provided arguments.
    """
    args = parse_args()
    config = load_yml(args.config)
    config = check_config(config)
    print(config)

    if args.parallel:
        md_parallel_batch(config, restart=args.restart)
    else:  
        for structure_path in config["io"]["initial_structures"]:
            run_md(structure_path, config["md"]["computing"]["devices"][0], config, restart=args.restart)

if __name__ == "__main__":
    main()