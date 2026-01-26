import pytest
from vmc.system import BCCHeisenberg
from nets.net import ViTNd
from vmc.optimization.protocols import MCProtocol

# Network
depth = 2
features = 6
kernel_width = 3
expansion_factor = 2
output_depth = 1
# Protocol
MC_protocol_dict = {
    "samples_per_rank": 16,
    "n_chains_per_rank": 8,
    "discard_fraction": 0.0,
    "iters": [2],
    "lr": [1e-3],
    "lr_factor": [0.5],
    "diag_shift": [1e-2],
    "diag_shift_factor": [1e-4],
    "chunk_size": [16],
    "save_every": 2,
    "momentum": 0.9,
    "checkpoint": 1,
    "post_iters": 0,
    "save_base": "",
    "save_num": 0,
    "time_it": 0,
    "show_progress": 1,
    "double_precision": 1,
    "sweep_factor": 1,
    "seed": 5,
    "solver": "cholesky",
    "load_stage": 0,
    "load_base": "",
    "norm_constraint_factor": None,
    "proj_reg": None,
}

threed_systems = (BCCHeisenberg(lattice_shape=(4, 2, 2), J=(1, 2)),)

MC_test_input = [
    (
        ViTNd(
            depth=2,
            d_model=12,
            heads=6,
            output_head="FT",
            expansion_factor=2,
            q=(0, 0, 0),
            system=system,
        ),
        system,
    )
    for system in threed_systems
]


def describe_test_case(network, system):
    net_type = type(network).__name__
    sys_type = type(system).__name__
    out_head = getattr(network, "output_head", getattr(network, "net_type", ""))
    return f"{net_type}_{out_head}_{sys_type}"


@pytest.mark.parametrize(
    "input", MC_test_input, ids=[describe_test_case(n, s) for n, s in MC_test_input]
)
def test_MCprotocol_nosymm(input, tmp_path):
    network = input[0]
    system = input[1]
    MC_protocol_dict["n_symmetry_stages"] = 1
    MC_protocol_dict["save_base"] = str(tmp_path)
    print("Running", network, system, MC_protocol_dict)
    protocol = MCProtocol(system, network, MC_protocol_dict)
    protocol.run()


@pytest.mark.parametrize(
    "input", MC_test_input, ids=[describe_test_case(n, s) for n, s in MC_test_input]
)
def test_MCprotocol_symm(input, tmp_path):
    network = input[0]
    system = input[1]
    n = 2
    MC_protocol_dict["n_symmetry_stages"] = n
    MC_protocol_dict["iters"] = n * MC_protocol_dict["iters"]
    MC_protocol_dict["lr"] = n * MC_protocol_dict["lr"]
    MC_protocol_dict["lr_factor"] = n * MC_protocol_dict["lr_factor"]
    MC_protocol_dict["diag_shift"] = n * MC_protocol_dict["diag_shift"]
    MC_protocol_dict["diag_shift_factor"] = n * MC_protocol_dict["diag_shift_factor"]
    MC_protocol_dict["chunk_size"] = n * MC_protocol_dict["chunk_size"]
    MC_protocol_dict["save_base"] = str(tmp_path)
    print("Running", network, system, MC_protocol_dict)
    protocol = MCProtocol(system, network, MC_protocol_dict)
    protocol.run()
