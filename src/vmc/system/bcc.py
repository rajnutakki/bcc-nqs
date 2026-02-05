# 2D spin models
import netket as nk
import nk_extensions
import numpy as np
import argparse
import warnings

from collections.abc import Sequence
from vmc.system.base import SpinSystem, BaseSystem
from netket.utils.group import PermutationGroup, Identity
from netket.graph.space_group import Translation
from netket.utils.types import Array
from netket.nn.blocks import SymmExpSum
from nets.blocks.sign import SignNet, SignHelper, SignRule, DoubleSignNet
from nets.blocks import FlipExpSum
from functools import partial
from nk_extensions.group.translations import translation_group_from_axis_translations
from typing import Optional
import jax.numpy as jnp
from netket.utils import HashableArray


class BCCHeisenberg(SpinSystem):
    """
    Heisenberg model on a bcc lattice
    """

    name = "BCCHeisenberg"
    n_basis = 2  # number of sites in the unit cell

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--lattice_shape",
            type=int,
            action="append",
            required=True,
            help="Shape of the cubic unit cell",
        )
        parser.add_argument(
            "--J",
            type=float,
            action="append",
            required=True,
            help="Sequence of J values",
        )
        parser.add_argument(
            "--sign_rule",
            type=int,
            default=0,
            help="Define a SignNet with J1 sign rule (1) or J2 sign rule (2)",
        )
        parser.add_argument(
            "--tetragonal_distortion",
            type=float,
            default=1.0,
            help="Tetragonal distortion of the lattice in z direction, ie c/a ratio",
        )
        parser.add_argument(
            "--little_group_id",
            type=int,
            default=None,
            help="If specified, symmetrize in this irrep of the little group",
        )
        parser.add_argument(
            "--spin_flip_symmetric",
            type=int,
            default=1,
            help="Whether to symmetrize as spin flip symmetric or not",
        )

    @staticmethod
    def read_arguments(args: argparse.Namespace):
        # args.patch_shape and args.q is defined in the nqs_nets.net.wrappers ViTNd
        return (
            args.lattice_shape,
            tuple(args.J),
            args.patch_shape,
            args.sign_rule,
            args.tetragonal_distortion,
            args.q,
            args.little_group_id,
            args.spin_flip_symmetric,
        )

    def __init__(
        self,
        lattice_shape: tuple[int, int, int],
        J: Sequence[float],
        patch_shape: Optional[tuple[int]] = None,
        sign_rule: int = 0,
        tetragonal_distortion: float = 1.0,
        q: Array = np.array([0, 0, 0]),
        little_group_id: Optional[int] = None,
        spin_flip_symmetric: bool = True,
        sz_sector: float = 0,
    ):
        """
        Args:
            lattice_shape: extent of the lattice in units of the lattice vectors (different lattice vectors if simple_cubic = True or False)
            J: J values of the Heisenberg model ordered by distance on the lattice
            patch_shape: shape of the patches used in the network, in units of the unit cell, e.g if simple_cubic = False the unit cell has a single site and a (2,1,1) patch_shape means
                         taking the patch as two sites along a1, one along a2 and one along a3.
            sign_rule: Use as an ansatz S(x)\Psi(x) where S(x) gives a sign according to the J1 term (1) J2 term (2) of J1-J2 BCC Heisenberg model. 0 means no sign rule is used.
            tetragonal_distortion: c/a ratio of the lattice
            q: Momentum sector of space group
            little_group_id: If specified, symmetrize in this irrep of the little group, otherwise defaults to invariant irreps of all space group symmetries
            spin_flip_symmetric: Whether to symmetrize in the spin-flip symmetric (True) or spin-flip antisymmetric (False) sector
            sz_sector: Total s^z sector of the Hilbert space to work in
        """
        super().__init__(
            N=self.n_basis * int(np.prod(lattice_shape)), sz_sector=sz_sector
        )
        self.J = J
        self.lattice_shape = lattice_shape
        self.sign_rule = sign_rule
        self.patch_shape = patch_shape
        self.tetragonal_distortion = tetragonal_distortion
        self.q = np.pi * np.array(q)
        self.name = "BCCHeisenberg"
        self.little_group_id = little_group_id
        self.spin_flip_symmetric = spin_flip_symmetric

        self.graph = nk_extensions.graph.BCC_cubic(
            extent=lattice_shape,
            pbc=True,
            max_neighbor_order=len(J),
            tetragonal_distortion=self.tetragonal_distortion,
        )  # the graph which is the Hamiltonian is defined on
        print(f"Using simple cubic BCC lattice with lattice shape {self.graph.extent}")

        # Get full translation group and intra-patch translations
        self.translation_group = self.graph.translation_group()
        # Translations of the unit cell within the patch
        self.intra_patch_translations = translation_group_from_axis_translations(
            self.graph, n=(1, 1, 1), max_translations=self.patch_shape
        )
        # Translations within the unit cell if a multi-site unit cell (ie simple cubic)
        displacement_vector = self.graph.site_offsets[1] - self.graph.site_offsets[0]
        trans_perm = self.graph.id_from_position(
            self.graph.positions - displacement_vector
        )
        intra_unitcell_translations = PermutationGroup(
            [
                Identity(),
                Translation(
                    inverse_permutation_array=trans_perm,
                    displacement=displacement_vector,
                ),
            ],
            degree=self.graph.n_nodes,
        )
        self.intra_patch_translations = (
            self.intra_patch_translations @ intra_unitcell_translations
        )
        self.translation_group = self.translation_group @ intra_unitcell_translations
        assert len(self.translation_group) == self.graph.n_nodes, (
            "Translation group does not have the correct number of elements"
        )
        # Point group symmetries
        if little_group_id is None:  # symmetrize over invariant sectors
            print(
                "No little group irrep specified, symmetrizing over all invariant sectors"
            )
            self.point_group = self.graph.point_group()  # need to use point group of full graph, as symmetrized over full translation group!
            print("Number of point group symmetries: ", len(self.point_group))
            self.symmetrizing_functions = (
                lambda net: net,  # translationally symmetrized up to patches
                lambda net: SymmExpSum(  # translations within the patch
                    net, self.intra_patch_translations
                ),
                lambda net: SymmExpSum(
                    net, self.intra_patch_translations @ self.point_group
                ),  # Full point group
                lambda net: FlipExpSum(
                    SymmExpSum(net, self.intra_patch_translations @ self.point_group)
                ),
            )
        else:
            self.space_group = self.graph.space_group()
            self.little_group = self.graph.point_group(
                self.space_group.little_group(self.q)
            )
            elements, characters = self.space_group.little_group_irreps_readable(
                self.q, full=True
            )
            little_group_characters = characters[little_group_id]
            print(f"Momentum sector k = {self.q}")
            print(f"Using little group irrep {little_group_id}:")
            print("Little group elements: \n", elements)
            print("Little group characters: \n", little_group_characters)
            little_group_characters = HashableArray(little_group_characters)
            self.symmetrizing_functions = (
                lambda net: net,  # to momentum sector
                lambda net: SymmExpSum(  # little group sector specified by characters
                    net, self.little_group, characters=little_group_characters
                ),
                lambda net: FlipExpSum(
                    SymmExpSum(
                        net, self.little_group, characters=little_group_characters
                    ),
                    symmetrize=spin_flip_symmetric,
                ),  # S^z parity and little group
            )

        if min(self.lattice_shape) < 3:
            warnings.warn(
                "Using BCC lattice with side length <= 2, sites may be connected to each other twice due to periodic boundaries."
            )
            self.custom_edges = nk_extensions.graph.utils.edges_from_graph_positions(
                self.graph
            )
            self.hamiltonian = nk_extensions.operator.heisenberg.heisenberg_edges(
                hilbert=self.hilbert_space, edges=self.custom_edges, J=J
            )
        else:
            self.hamiltonian = nk.operator.Heisenberg(
                hilbert=self.hilbert_space, graph=self.graph, J=self.J
            )

        self.hamiltonian_name = "Heisenberg"
        self.sampler_t = partial(
            nk.sampler.MetropolisExchange, graph=self.graph, d_max=2
        )

        # Sign rule
        self.j1_xsublattice = (0,)  # a sites of nn BCC lattice
        self.sign_helperj1 = SignHelper(
            graph=self.graph, x_sublattices=self.j1_xsublattice
        )
        if np.isclose(tetragonal_distortion, 1.0):
            # J2 joins two independent simple cubic sublattices
            self.j2_xsites1 = tuple(
                [
                    i
                    for i, site in enumerate(self.graph.sites)
                    if site.basis_coord[-1] == 0
                    and np.sum(site.basis_coord[:-1]) % 2 == 0
                ]
            )  # one sublattice of simple_cubic sublattice
            self.j2_xsites2 = tuple(
                [
                    i
                    for i, site in enumerate(self.graph.sites)
                    if site.basis_coord[-1] == 1
                    and np.sum(site.basis_coord[:-1]) % 2 == 0
                ]
            )  # one sublattice of other simple_cubic sublattice
        else:  # Jab and Jc sign rule
            self.j2_xsites1 = tuple(
                [
                    site.id
                    for site in self.graph.sites
                    if (site.basis_coord[0] + site.basis_coord[1]) % 2 == 0
                    and site.basis_coord[-1] == 0
                ]
            )
            self.j2_xsites2 = tuple(
                [
                    site.id
                    for site in self.graph.sites
                    if (site.basis_coord[0] + site.basis_coord[1]) % 2 == 0
                    and site.basis_coord[-1] == 1
                ]
            )

        self.sign_helperj2_1 = SignHelper(graph=self.graph, x_sites=self.j2_xsites1)
        self.sign_helperj2_2 = SignHelper(graph=self.graph, x_sites=self.j2_xsites2)

        if self.sign_rule == 1:  # J1 sign rule
            self.sign_net = lambda net: SignNet(
                logpsi=net,
                sign_type=SignRule,
                compute_nx=self.sign_helperj1.compute_nx,
                dtype=jnp.complex128,
            )
        elif self.sign_rule == 2:  # J2 sign rule
            self.sign_net = lambda net: DoubleSignNet(
                logpsi=net,
                sign_type=SignRule,
                compute_nx1=self.sign_helperj2_1.compute_nx,
                compute_nx2=self.sign_helperj2_2.compute_nx,
                dtype=jnp.complex128,
            )

        self.q = tuple(np.round(self.q / np.pi, decimals=2))

    def name_and_arguments_to_dict(self):
        return {
            "name": self.name,
            "lattice_shape": self.lattice_shape,
            "J": self.J,
            "patch_shape": self.patch_shape,
            "sign_rule": self.sign_rule,
            "tetragonal_distortion": self.tetragonal_distortion,
            "q": self.q,
            "little_group_id": self.little_group_id,
            "spin_flip_symmetric": self.spin_flip_symmetric,
            "sz_sector": self.sz_sector,
        }


systems = {"BCCHeisenberg": BCCHeisenberg}


def from_dict(arg_dict: dict):
    """
    Return the system specified by the dictionary
    """
    try:
        system = systems[str(arg_dict["name"])]
        del arg_dict["name"]
    except KeyError:
        system = systems[str(arg_dict["Name"])]
        del arg_dict["Name"]
    try:
        return system(**arg_dict)
    except TypeError:
        del arg_dict["simple_cubic"]
        return system(**arg_dict)


def load(file_name: str, prefix: str = None):
    """
    Return the system specified by the dictionary, dict[prefix], contained in
    the json file file_filename
    """
    arg_dict = BaseSystem.argument_loader(file_name, prefix)
    loaded_system = from_dict(arg_dict)
    return loaded_system
