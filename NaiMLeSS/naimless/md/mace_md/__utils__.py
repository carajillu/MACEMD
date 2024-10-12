import time

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