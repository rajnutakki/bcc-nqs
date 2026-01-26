import pytest
from nets.net.ViT.body import Encoder
import netket as nk
import numpy as np
import jax


@pytest.mark.parametrize(
    "graph_type, ndim, extent_list, kernel_shape_fn",
    [
        # 1D Hypercube
        ("Hypercube", 1, [8], lambda k: [k]),
        # 2D Hypercube
        ("Hypercube", 2, [8, 8], lambda k: [k, k]),
        # 2D Grid
        ("Grid", 2, [8, 4], lambda k: [2 * k, k]),
        # 3D Hypercube
        ("Hypercube", 3, [8, 8, 8], lambda k: [k, k, k]),
        # 3D Grid
        ("Grid", 3, [8, 4, 4], lambda k: [2 * k, k, k]),
    ],
)
def test_translational_equivariance(graph_type, ndim, extent_list, kernel_shape_fn):
    d = 12
    L = extent_list[0]

    if graph_type == "Hypercube":
        graph = nk.graph.Hypercube(length=L, n_dim=ndim, pbc=True)
    elif graph_type == "Grid":
        graph = nk.graph.Grid(extent=tuple(extent_list), pbc=True)
    else:
        raise ValueError(f"Unsupported graph type: {graph_type}")

    for k in np.arange(graph.extent[-1], 1, -1):
        encoder = Encoder(
            num_layers=2,
            d_model=d,
            h=6,
            plattice_shape=tuple(graph.extent),
            kernel_shape=kernel_shape_fn(k),
        )

        run_translational_equivariance_test(encoder, graph, d)


def run_translational_equivariance_test(encoder, graph, d):
    Np = graph.n_nodes
    test_input = np.random.random(size=(d, Np))  # (d, Np)

    # Apply all translations
    translated_input = graph.translation_group() @ test_input  # (Nt, d, Np)
    translated_input = translated_input.transpose(0, 2, 1)  # (Nt, Np, d)

    vars = encoder.init(jax.random.PRNGKey(2), translated_input[0].reshape(1, Np, d))
    output = encoder.apply(vars, translated_input)  # (Nt, Np, d)

    # Output of unshifted input
    t0_out = output[0]  # (Np, d)

    # Translate the unshifted output using the group
    translated_output = graph.translation_group() @ t0_out.T  # (Nt, d, Np)
    translated_output = translated_output.transpose(0, 2, 1)  # (Nt, Np, d)

    # Check for equivariance
    assert np.allclose(translated_output, np.array(output), atol=1e-10), (
        "Output not equivariant"
    )

    # Check for invariance under translation of total features (sum over nodes)
    for i in range(output.shape[0]):
        assert np.allclose(
            np.sum(output[i], axis=0), np.sum(output[0], axis=0), atol=1e-10
        ), "Output not invariant"


def test_masking():
    # TODO 2d  and 3d version of this
    ndim = 1
    d = 12
    Np = 10
    k = 3
    graph = nk.graph.Hypercube(length=Np, n_dim=ndim, pbc=True)

    encoder = Encoder(
        num_layers=2,
        d_model=d,
        h=6,
        plattice_shape=tuple(graph.extent),
        kernel_shape=ndim * [k],
    )

    i = 2  # Central site to test
    input1 = np.random.random(size=(1, Np, d))
    input2 = input1.copy()

    # Modify input2 outside the kernel region around i
    input2[:, : i - k // 2, :] += 0.5
    input2[:, i + k // 2 + 1 :, :] += 0.5

    vars = encoder.init(jax.random.PRNGKey(2), input1)
    output1 = encoder.apply(vars, input1)
    output2 = encoder.apply(vars, input2)

    # Check that kernel output is unchanged
    assert np.allclose(
        output1[:, i - k // 2 : i + k // 2 + 1, :],
        output2[:, i - k // 2 : i + k // 2 + 1, :],
        atol=1e-6,
    ), "Output within the kernel should be invariant to changes outside the kernel."

    # Check that outputs outside the kernel *do* differ
    if i - k // 2 > 0:
        assert not np.allclose(
            output1[:, : i - k // 2, :], output2[:, : i - k // 2, :]
        ), "Output before kernel should change when input changes."

    if i + k // 2 + 1 < Np:
        assert not np.allclose(
            output1[:, i + k // 2 + 1 :, :], output2[:, i + k // 2 + 1 :, :]
        ), "Output after kernel should change when input changes."
