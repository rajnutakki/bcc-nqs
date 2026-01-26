import os
import shutil
import sys


def remove_files(root_dir=None):
    if root_dir is None:
        root_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Cleaning up files in {root_dir}...")
    try:
        os.remove(root_dir + "/opt1.log")
        os.remove(root_dir + "/opt.json")
        os.remove(root_dir + "/opt.mpack")
    except FileNotFoundError:
        pass
    try:
        os.remove(root_dir + "/opt2.log")
        os.remove(root_dir + "/opt3.log")
    except FileNotFoundError:
        pass
    try:
        shutil.rmtree(root_dir + "/checkpoint0")
        shutil.rmtree(root_dir + "/checkpoint1")
        shutil.rmtree(root_dir + "/checkpoint2")
        shutil.rmtree(root_dir + "/checkpoint3")
        shutil.rmtree(root_dir + "/checkpoint4")
    except FileNotFoundError:
        pass
    try:
        shutil.rmtree(root_dir + "/post")
        os.remove(root_dir + "/post.json")
    except FileNotFoundError:
        pass
    try:
        os.remove(root_dir + "/post_exp.json")
        os.remove(root_dir + "/post_exp_samples.npy")
    except FileNotFoundError:
        pass
    try:
        os.remove(root_dir + "/expectation_values.json")
        os.remove(root_dir + "/expectation_values_samples.npy")
    except FileNotFoundError:
        pass
    try:
        os.remove(root_dir + "/exp_stage2.json")
        os.remove(root_dir + "/exp_stage2_samples.npy")
    except FileNotFoundError:
        pass
    try:
        shutil.rmtree(root_dir + "/vstate")
        shutil.rmtree(root_dir + "/vstate1")
        shutil.rmtree(root_dir + "/vstate2")
        shutil.rmtree(root_dir + "/vstate3")
        shutil.rmtree(root_dir + "/vstate4")
        shutil.rmtree(root_dir + "/vstate5")
    except FileNotFoundError:
        pass
    try:
        shutil.rmtree(root_dir + "/increase_size")
    except FileNotFoundError:
        pass

    print("Finished cleaning")


if __name__ == "__main__":
    remove_files(sys.argv[1] if len(sys.argv) > 1 else None)
