import os
import subprocess
from ase.io import read
import numpy as np
def create_run_cp2k(dyn, cp2k_config):
    def run_cp2k():
        mace_energy = dyn.atoms.calc.get_potential_energy()
        mace_forces = dyn.atoms.calc.get_forces()
        
        os.makedirs("cp2k_files", exist_ok=True)
        os.chdir("cp2k_files")
        
        with open("cp2k.in", "w") as f:
            f.write(cp2k_config['input_str'])
        print(f"Writing coordinates to {cp2k_config['coord_file_name']}")
        dyn.atoms.write(cp2k_config['coord_file_name'])

        subprocess.run(["mpirun", "-np", str(cp2k_config['nprocs']), cp2k_config['exe'], "-i", "cp2k.in", "-o", "cp2k.out"], check=True)
        
        cp2k_atoms = read(f"{cp2k_config['project_name']}-pos-1.xyz")
        cp2k_energy = cp2k_atoms.info['E']
        cp2k_atoms = read(f"{cp2k_config['project_name']}-frc-1.xyz")
        cp2k_forces = cp2k_atoms.arrays['positions']
        os.chdir("..")
        
        if (abs(mace_energy - cp2k_energy) > cp2k_config['energy_tol']) or \
           (not np.allclose(mace_forces, cp2k_forces, atol=cp2k_config['force_tol'])):
            print(f"CP2K Energy: {cp2k_energy}, MACE Energy: {mace_energy}")
            print(f"CP2K Forces: {cp2k_forces}, MACE Forces: {mace_forces}")
            print(f"Force mismatch between MACE and CP2K")
            dyn.atoms.info['E'] = cp2k_energy
            dyn.atoms.arrays['frc'] = cp2k_forces
            dyn.atoms.write("cp2k_snaps.xyz", append=True)
    return run_cp2k
