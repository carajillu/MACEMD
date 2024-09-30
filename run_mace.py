import argparse
import pymp
import yaml
import sys
import os
from ase.io import read
from ase import units
import glob
import importlib
import random
import torch
foundation_models=["mace_off","mace_anicc","mace_mp"]
# Get the list of possible dynamics classes that can be impored from ase.md. Store them in a list. Use dir() to get all the classes in the module.
md_module=importlib.import_module("ase.md")
dynamics_classes=[]
for name in dir(md_module):
    if "__" not in name:
        dynamics_classes.append(name)
print(dynamics_classes)

def get_units(config):
    units_dict=config["units"]
    for key,value in units_dict.items():
        units.set_default_units(key,value)
    return
def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("-c","--config",nargs="?",help="Path to the config file",default="example.yml")
    args=parser.parse_args()
    return args

def parse_yml(yml_path):
    with open(yml_path,"r") as f:
        config=yaml.load(f,Loader=yaml.FullLoader)
    return config

def read_initial_structures(init_struct_dir):
    '''
    Reads all xyz files present in the initial_structures directory and returns a list of atoms objects
    '''
    os.chdir(init_struct_dir)
    initial_structures=[]
    for file in glob.glob("*.xyz"):
        print(file)
        atoms=read(file)
        initial_structures.append(atoms)
    return initial_structures

def add_model(model_name,devices,model_path):
    '''
    If the model is a foundation model, use the ASE calculator interface to add the model
    If not, load the model from its file and use the ase.calculators.mace.MACECalculator interface.
    The device should be specified every time. If there is a list of devices, the different alculatoers should be spread evenly among the available devices.
    use importlib
    '''
    model_instances=[]
    model_module=importlib.import_module(f"mace.calculators")
    if model_name in foundation_models:
        model_class=getattr(model_module,model_name)
    else:
        model_class=getattr(model_module,"MACECalculator")
    
    if model_name in foundation_models:
        for i in range(len(devices)):
            calculator=model_class(model=model_path,device=devices[i])
            model_instances.append(calculator)
    else:
        for i in range(len(devices)):
            calculator=model_class(model_path=model_path,device=devices[i])
            model_instances.append(calculator)
    return model_instances

def main():
    #read the config file
    args=parse_args()
    config=parse_yml(args.config)
    print(config)

    #read the initial structures
    init_struct_dir=config["initial_structures"]
    initial_structures=read_initial_structures(init_struct_dir)
    print(initial_structures)

    #read the model
    model_name=config["mace"]["model"]
    devices=config["mace"]["devices"]
    model_path=config["mace"]["model_path"]
    model_instances=add_model(model_name,devices,model_path)
    print(model_instances)

    # Assign calculators evenly to the structures
    for i in range(len(initial_structures)):
        initial_structures[i].set_calculator(model_instances[i%len(model_instances)])
        print(initial_structures[i],initial_structures[i].get_calculator().device)
    
    #Organize structures into batches by device
    device_batches={}
    for structure in initial_structures:
        device=structure.get_calculator().device
        if device not in device_batches:
            device_batches[device]=[]
        device_batches[device].append(structure)
    print(device_batches)
    sys.exit()
 





    #Get MD parameters
    dynamics=config["md"]["dynamics"]
    if dynamics not in dynamics_classes:
        raise ValueError(f"Invalid dynamics class: {dynamics}")
    md_module=importlib.import_module(f"ase.md.{dynamics}")
    dynamics_class=getattr(md_module,dynamics.title())
    print(dynamics_class)

    timestep=config["md"]["timestep"]
    friction=config["md"]["friction"]
    temperature=config["md"]["temperature"]
    print(f"Timestep: {timestep}, Friction: {friction}, Temperature: {temperature}")
    #sys.exit()
    
    
    
    
            

if __name__=="__main__":
    main()