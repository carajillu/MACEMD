import argparse
import pymp
import yaml
import sys
import os
from ase.io import read
from ase import units
import glob
import importlib
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

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
    root_dir=os.getcwd()
    os.chdir(init_struct_dir)
    initial_structures=[]
    for file in glob.glob("*.xyz"):
        print(file)
        atoms=read(file)
        atoms.cell=[10,10,10]
        initial_structures.append(atoms)
    os.chdir(root_dir)
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
    for dev in device_batches:
        print(f"Device: {dev}, Structures: {device_batches[dev]}")

    #Get MD parameters
    dynamics=config["md"]["dynamics"]
    if dynamics['class'] not in dynamics_classes:
        raise ValueError(f"Invalid dynamics class: {dynamics['class']}")
    md_module=importlib.import_module(f"ase.md.{dynamics['module']}")
    dynamics_class=getattr(md_module,dynamics['class'])
    print(dynamics_class)

    #Set the velocities
    for atoms in initial_structures:
        MaxwellBoltzmannDistribution(atoms, temperature_K=config["md"]["parameters"]["temperature_K"])
    
    #Run MD in parallel for each device
    root_dir=os.getcwd()
    with pymp.Parallel(len(device_batches)) as p:
        for j in p.range(len(device_batches)):
            dev=list(device_batches.keys())[j]
            for i in range(len(device_batches[dev])):
                dyn=dynamics_class(device_batches[dev][i],timestep=config["md"]["timestep"],**config["md"]["parameters"])
                os.makedirs(f"{dyn.atoms.symbols}",exist_ok=True)
                os.chdir(f"{dyn.atoms.symbols}")
                def print_md_snapshot(): #that has to go somewhere else
                    filename=f"{dyn.atoms.symbols}.trj.xyz"
                    dyn.atoms.write(filename,append=True)
                dyn.attach(print_md_snapshot,interval=config["md"]["stride"])
                dyn.attach(MDLogger(dyn, dyn.atoms, 'md.log', header=False, stress=False,
                           peratom=True, mode="a"), interval=config["md"]["stride"])
                nsteps=config["md"]["nsteps"]
                dyn.run(steps=nsteps)
                os.chdir(root_dir)
    
    
    
    
    
            

if __name__=="__main__":
    main()