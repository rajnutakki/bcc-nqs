from netket.graph import Lattice
from netket.utils.group import PointGroup
from netket.utils.group import cubic
from nk_extensions.group.pg_utils import extract_valid_point_group
from typing import Sequence
import numpy as np


def BCC_cubic(
    extent: Sequence[int],
    pbc: bool = True,
    max_neighbor_order: int = 1,
    point_group: PointGroup | None = None,
    tetragonal_distortion: float = 1.0,
    **kwargs,
) -> Lattice:
    """
    The body-centered cubic lattice using the conventional cubic unit cell.
    This lattice has a two-site basis.
    """
    # Check input
    assert len(extent) == 3
    extent = np.array(extent)
    # Cubic Bravais lattice
    basis_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, tetragonal_distortion]])
    # 2 site basis
    site_offsets = np.array(
        [
            [0, 0, 0],
            [0.5, 0.5, tetragonal_distortion / 2],
        ]
    )

    if (
        point_group is None
    ):  # determine which symmetries of full point group are symmetries of this cluster
        temp_graph = Lattice(
            basis_vectors=basis_vectors,
            extent=extent,
            site_offsets=site_offsets,
            pbc=pbc,
            **kwargs,
        )
        point_group = extract_valid_point_group(temp_graph, cubic.Oh())

    return Lattice(
        basis_vectors=basis_vectors,
        extent=extent,
        pbc=pbc,
        site_offsets=site_offsets,
        max_neighbor_order=max_neighbor_order,
        point_group=point_group,
    )
