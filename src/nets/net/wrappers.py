# Wrap the neural networks to define parameters and save
from .base_wrapper import NetBase
import argparse
from nets.net import ViT
from nets.blocks.patching import Patching


class ViTNd(NetBase):
    nets = {
        "Vanilla": ViT.ndim,
        "FT": ViT.FT,
        "Positive": ViT.Positive,
        "FTReal": ViT.FTReal,
    }

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--depth", type=int, required=True, help="Number of encoder layers"
        )
        parser.add_argument(
            "--d_model",
            type=int,
            required=True,
            help="Model dimension (number of features)",
        )
        parser.add_argument(
            "--heads",
            type=int,
            required=True,
            help="Number of heads in attention mechanism",
        )
        parser.add_argument(
            "--output_head",
            type=str,
            required=True,
            help="Which output head to use (Vanilla / LayerSum / RBMnoLayer / noLayer)",
        )
        parser.add_argument(
            "--expansion_factor",
            type=int,
            required=True,
            help="Factor to expand model dimension in feedforward block",
        )
        parser.add_argument(
            "--q",
            type=float,
            action="append",
            required=False,
            help="Momentum of quantum state in units of pi",
        )
        parser.add_argument(
            "--kernel_shape",
            type=int,
            action="append",
            required=False,
            help="Shape of attention matrix in space",
        )
        parser.add_argument(
            "--patch_shape",
            type=int,
            action="append",
            required=False,
            help="Shape of patches",
        )
        parser.add_argument(
            "--sign_net",
            type=int,
            required=False,
            default=0,
            help="Whether to use wrap with a sign net which implements a sign rule",
        )

    @staticmethod
    def read_arguments(args: argparse.Namespace):
        return {
            "depth": args.depth,
            "d_model": args.d_model,
            "heads": args.heads,
            "output_head": args.output_head,
            "expansion_factor": args.expansion_factor,
            "q": args.q,
            "kernel_shape": args.kernel_shape,
            "patch_shape": args.patch_shape,
            "sign_net": args.sign_net,
        }

    def __init__(
        self,
        depth: int,
        d_model: int,
        heads: int,
        output_head: str,
        expansion_factor: int,
        system,
        q: tuple[float] = (0, 0),
        kernel_shape: tuple[int] = None,
        patch_shape: tuple[int] = None,
        sign_net: bool = False,
    ):
        self.name = "ViTNd"
        # Extract lattice information for patching
        patches = Patching(system.graph, output_dim=1, patch_shape=patch_shape)
        # Network properties
        self.patch_shape = patch_shape
        self.patches = patches
        self.depth = depth
        self.d_model = d_model
        self.heads = heads
        self.output_head = output_head
        self.expansion_factor = expansion_factor
        if kernel_shape is None:
            self.kernel_shape = kernel_shape
        else:
            self.kernel_shape = tuple(kernel_shape)
        if q is None:
            self.q = q
        else:
            self.q = tuple(q)
        self.sign_net = sign_net
        if self.output_head == "FT" or self.output_head == "FTReal":
            assert len(self.q) == system.graph.ndim
            print(f"Momentum sector = {self.q}")
            self.network = self.nets[self.output_head](
                num_layers=self.depth,
                d_model=self.d_model,
                heads=self.heads,
                plattice_shape=patches.plattice_shape,
                extract_patches=patches.extract_patches,
                expansion_factor=expansion_factor,
                q=self.q,
                kernel_shape=self.kernel_shape,
                compute_positions=patches.compute_positions,
                transl_invariant=True,
            )
        else:
            self.network = self.nets[self.output_head](
                num_layers=self.depth,
                d_model=self.d_model,
                heads=self.heads,
                plattice_shape=patches.plattice_shape,
                extract_patches=patches.extract_patches,
                kernel_shape=self.kernel_shape,
                expansion_factor=expansion_factor,
                transl_invariant=True,
            )
        # Sign rule
        if (
            self.sign_net
        ):  # assumes complex network (doesnt make sense to use with real positive)
            print("Using sign rule helper network")
            self.network = system.sign_net(self.network)

    def name_and_arguments_to_dict(self):
        """
        Convert the arguments for __init__ (except system) to a dictionary
        """
        arg_dict = {
            "name": self.name,
            "depth": self.depth,
            "d_model": self.d_model,
            "heads": self.heads,
            "output_head": self.output_head,
            "expansion_factor": self.expansion_factor,
            "q": self.q,
            "kernel_shape": self.kernel_shape,
            "patch_shape": self.patch_shape,
            "sign_net": self.sign_net,
        }
        return arg_dict


networks = {"ViTNd": ViTNd}


def from_dict(arg_dict: dict, system, network_name="ConvNext"):
    """
    Return the wrapped network specified by the dictionary
    """
    try:
        network = networks[str(arg_dict["name"])]
        del arg_dict["name"]
    except KeyError:  # compatibility with old versions where it wasnt saved
        network = networks[network_name]
        arg_dict["net_type"] = arg_dict["output_head"]
        del arg_dict["output_head"]
    try:
        return network(**arg_dict, system=system)
    except TypeError:
        del arg_dict["gutzwiller"]
        return network(**arg_dict, system=system)


def load(file_name: str, system, prefix: str = None):
    """
    Return the wrapped network specified by the dictionary, dict[prefix], contained in
    the json file file_filename
    """
    arg_dict = NetBase.argument_loader(file_name, prefix)
    loaded_network = from_dict(arg_dict, system)
    return loaded_network
