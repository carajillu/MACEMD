import argparse
import glob
import os
import torch
from naimless.io.system_io import parse_yml, read_structure
from naimless.md.mace_calculator import setup_md, attach_md_loggers
from naimless.qm.cp2k import create_run_cp2k
from naimless.md.mace_calculator import run_dyn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", nargs="?", help="Path to the config file", default="config.yaml")
    parser.add_argument("--restart", action="store_true", help="Restart the simulation from the last step")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (show warnings)")
    args = parser.parse_args()
    return args

def process_structure(structure_path, device_name, config, restart=False):
    atoms = read_structure(structure_path)
    dyn = setup_md(atoms, config)
    
    system_name = os.path.basename(structure_path).split(".")[0]
    root_dir = os.getcwd()
    os.makedirs(f"{system_name}", exist_ok=True)
    os.chdir(f"{system_name}")
    
    attach_md_loggers(dyn, system_name, config)
    
    if isinstance(config['cp2k'], dict):
        cp2k_run = create_run_cp2k(dyn, config['cp2k'])
        dyn.attach(cp2k_run, interval=config['md']['stride'])
    
    run_dyn(system_name, dyn, config['md']['nsteps'], config['md']['stride'], restart)
    os.chdir(root_dir)
    del atoms, dyn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def main():
    args = parse_args()
    config = parse_yml(args.config)
    
    structure_path_list = glob.glob(config['initial_structures'] + "/*.xyz")
    device_names = config['mace']['devices']
    
    with torch.multiprocessing.Pool(processes=len(device_names)) as pool:
        args_list = [(structure_path, device_names[i % len(device_names)], config, args.restart) 
                     for i, structure_path in enumerate(structure_path_list)]
        pool.starmap(process_structure, args_list)

if __name__ == "__main__":
    main()