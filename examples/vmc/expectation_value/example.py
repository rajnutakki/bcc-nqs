# Compute the expectation values of observables after running an optimization protocol
from spin_vmc.expectation_value.expectation_value import compute
import os

os.system(
    "uv run ../../../packages/spin_vmc/optimization/run.py --config ../optimization/config_vit.yaml"
)  # first run an optimization

# After running an optimization, dirname is the output folder of that
dirname = "."
# Mc parameters for computing the expectation value, need not be the same as optimization, e.g using more samples
n_samples_per_chain = 100
n_chains = 16
n_discard_per_chain = 0
chunk_size = 1600
res = compute(
    dirname=dirname,
    n_samples_per_chain=n_samples_per_chain,
    n_chains=n_chains,
    n_discard_per_chain=n_discard_per_chain,
    chunk_size=chunk_size,
    observables=("energy", "mz"),
    save_type="opt",
)
print(res)

os.system("uv run ../cleaner.py $PWD")  # clean up files
