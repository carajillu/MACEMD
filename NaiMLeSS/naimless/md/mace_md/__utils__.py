import time
import importlib
import numpy as np

def create_print_md_snapshot(system_name, dyn):
    def print_md_snapshot():
        filename = f"{system_name}.trj.xyz"
        dyn.atoms.write(filename, append=True)
    return print_md_snapshot

def create_time_tracker(system_name):
    initial_time = time.time()
    def time_tracker():
        elapsed_time = time.time() - initial_time
        with open("time.log", "a") as f:
            f.write(f"{time.ctime()} {elapsed_time:.2f}\n")
    return time_tracker

def create_run_qm(qm_config,dyn):
    def run_qm():
        qm_module = importlib.import_module(qm_config["module"])
        calc=qm_module.get_calculator(qm_config)
        with calc as calc:
            for key in calc.parameters:
                print(f"{key}: {calc.parameters[key]}")
            atoms=dyn.atoms
            try:
                E=calc.get_potential_energy(atoms)
                F=calc.get_forces(atoms)
            except Exception as e:
                print(f"QM calculation failed with error: {e}. Setting energy and forces to zero.")
                E=0
                F=np.zeros_like(atoms.get_forces())
        dyn.atoms.info['E']=E
        dyn.atoms.arrays['frc']=F
    return run_qm