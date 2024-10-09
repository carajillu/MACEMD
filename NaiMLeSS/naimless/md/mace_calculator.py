import importlib
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

foundation_models = ["mace_off", "mace_anicc", "mace_mp"]

def load_calculator(device_name, mace_config):
    model_module = importlib.import_module("mace.calculators")
    if mace_config['model'] in foundation_models:
        model_class = getattr(model_module, mace_config['model'])
        calculator = model_class(model=mace_config['model_path'], device=device_name)
    else:
        model_class = getattr(model_module, "MACECalculator")
        calculator = model_class(model_path=mace_config['model'], device=device_name)
    return calculator

def load_dynamics(atoms, mdconfig):
    md_module = importlib.import_module("ase.md")
    dynamics_classes = md_module.__all__[1:]
    dynamics = mdconfig['dynamics']
    if dynamics['class'] not in dynamics_classes:
        raise ValueError(f"Invalid dynamics class: {dynamics['class']}")
    dynamics_class = getattr(md_module, dynamics['class'])
    if isinstance(mdconfig['parameters'], dict):
        dyn = dynamics_class(atoms, timestep=mdconfig['timestep'], **mdconfig['parameters'])
    else:
        dyn = dynamics_class(atoms, timestep=mdconfig['timestep'])
    return dyn

def setup_md(atoms, config):
    calculator = load_calculator(config['mace']['devices'][0], config['mace'])
    atoms.calc = calculator
    
    if isinstance(config['md']['parameters'], dict) and "temperature_K" in config['md']['parameters']:
        MaxwellBoltzmannDistribution(atoms, temperature_K=config['md']['parameters']['temperature_K'])
    else:
        print("No temperature provided, using the default temperature of 300 K")
        MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    
    dyn = load_dynamics(atoms, config['md'])
    return dyn

def attach_md_loggers(dyn, system_name, config):
    dyn.attach(MDLogger(dyn, dyn.atoms, 'md.log', header=True, stress=False,
               peratom=True, mode="a"), interval=config['md']['stride'])
