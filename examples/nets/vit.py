# Example of using the ViT with a FT head on a 3D lattice, using the BCCHeisenberg system
import netket as nk
import netket.experimental as nke
from nets.net import ViTNd
from vmc.system import BCCHeisenberg

# System
extent = (4, 4, 4)
J = (1, 1)
q = (0, 0, 0)
system = BCCHeisenberg(lattice_shape=extent, J=J, q=q)
# ViT
depth = 2
d_model = 12
heads = 6
output_head = "FT"
expansion_factor = 2
attention_kernel_shape = (4, 4, 4)  # receptive field of factored attention
network = ViTNd(
    depth=depth,
    d_model=d_model,
    heads=heads,
    output_head=output_head,
    expansion_factor=expansion_factor,
    system=system,
    q=q,
    kernel_shape=attention_kernel_shape,
)

# Optimization
lr = 1e-2
diag_shift = 1e-4
r = 1e-6
n_samples = 100
chunk_size = n_samples
symm_stage = 0  # specify symmetrization stage: 0 = translation, 1 = space group, 2 = space group and spin parity
sampler = system.sampler_t(system.hilbert_space)
optimizer = nk.optimizer.Sgd(learning_rate=lr)
SR_solver = nk.optimizer.solver.pinv_smooth(rtol=r, rtol_smooth=r)
vstate = nk.vqs.MCState(
    sampler,
    model=system.symmetrizing_functions[symm_stage](network.network),
    n_samples=n_samples,
    chunk_size=chunk_size,
)
gs = nke.driver.VMC_SR(
    hamiltonian=system.hamiltonian,
    optimizer=optimizer,
    diag_shift=diag_shift,
    chunk_size_bwd=chunk_size,
    variational_state=vstate,
    linear_solver_fn=SR_solver,
    use_ntk=True,
    on_the_fly=False,
)
log = nk.logging.RuntimeLog()
gs.run(
    n_iter=10,
    out=log,
)
