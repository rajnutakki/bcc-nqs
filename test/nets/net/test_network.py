import pytest

import numpy as np
import jax
from nets.net import ViTNd, load
from nets.utils.tree import variables_are_equal
from vmc.system import BCCHeisenberg

system = BCCHeisenberg(lattice_shape=(4, 4, 4), J=(1.0, 1.1, 1.2))
N = system.graph.n_nodes

test_input = (
    pytest.param(
        ViTNd(
            depth=4,
            d_model=36,
            heads=6,
            output_head="Vanilla",
            expansion_factor=2,
            system=system,
        ),
        id="ViTNd",
    ),
)


@pytest.mark.parametrize("network", test_input)
def test_save_load(network, tmpdir):
    file_name = str(tmpdir) + "/bla.json"
    prefix = "network"
    sample = np.ones((1, N))
    variables1 = network.network.init(jax.random.PRNGKey(0), sample)
    network.save(file_name, prefix, write_mode="w+")
    new_network = load(file_name, system, prefix)
    variables2 = new_network.network.init(jax.random.PRNGKey(0), sample)
    assert variables_are_equal(variables1, variables2)
