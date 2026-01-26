# Example of using the ViT with a FT head on a 3D lattice
import netket as nk
import netket.experimental as nke
import nk_extensions
from nets.net.ViT import FT as ViTFT
from nets.blocks.patching import Patching

# System
extent = (4, 4, 4)  # 4 x 4 x 4 unit cells
graph = nk_extensions.graph.BCC_cubic(
    extent=extent, pbc=True, max_neighbor_order=2, tetragonal_distortion=1.0
)  # J1-J2 BCC with 2-site basis
hilb = nk.hilbert.Spin(s=1 / 2, N=graph.n_nodes, total_sz=0)
ham = nk.operator.Heisenberg(hilbert=hilb, graph=graph, J=(1, 1))
# ViT
# With a small network here, to run quickly, see TODO for network parameters used
num_layers = 2
d_model = 12
heads = 6
plattice_shape = extent  # extent of lattice after patching
patches = Patching(graph, output_dim=1)  # automatically takes 2-site unit cell as patch
expansion_factor = 2
attention_kernel_shape = (4, 4, 4)  # receptive field of factored attention
q = (0, 0, 0)  # momentum in units of pi
network = ViTFT(
    num_layers=num_layers,
    d_model=d_model,
    heads=heads,
    plattice_shape=plattice_shape,
    expansion_factor=expansion_factor,
    extract_patches=patches.extract_patches,
    q=q,
    compute_positions=patches.compute_positions,
    kernel_shape=attention_kernel_shape,
)

# Optimization
lr = 1e-2
diag_shift = 1e-4
r = 1e-6
n_samples = 100
chunk_size = n_samples
sampler = nk.sampler.MetropolisExchange(hilbert=hilb, graph=graph, d_max=1)
optimizer = nk.optimizer.Sgd(learning_rate=lr)
SR_solver = nk.optimizer.solver.pinv_smooth(rtol=r, rtol_smooth=r)
vstate = nk.vqs.MCState(
    sampler, model=network, n_samples=n_samples, chunk_size=chunk_size
)
gs = nke.driver.VMC_SR(
    hamiltonian=ham,
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
