import json
import vmc.system.bcc as sys
import nets.net.wrappers as net
from netket.sampler import Sampler
import nets.utils.serialize as netsaver
import nqxpack
import os
from netket.vqs import VariationalState
import netket as nk
import warnings
from typing import Sequence
from vmc.optimization.utils import add_module, remove_module


def save(system, network, fname: str, save_type: str = "post", **kwargs):
    kwarg_dict = kwargs
    system_dict = system.name_and_arguments_to_dict()
    net_dict = network.name_and_arguments_to_dict()
    save_dict = {save_type: kwarg_dict, "system": system_dict, "network": net_dict}
    with open(fname, "w+") as f:
        json.dump(save_dict, f)


def load(fname: str, save_type: str = "post"):
    with open(fname, "r") as f:
        load_dict = json.load(f)
    try:
        min_index = int(load_dict[save_type]["min_index"])
    except KeyError:
        min_index = None
    try:
        symmetry_stage = int(load_dict[save_type]["symmetry_stage"])
    except KeyError:
        symmetry_stage = None
    try:
        n_chains_opt = int(load_dict[save_type]["n_chains"])
    except KeyError:
        n_chains_opt = None
    system = sys.load(fname, "system")
    network = net.load(fname, system, "network")
    network = network.network
    return min_index, system, network, symmetry_stage, n_chains_opt


def save_sampler(
    sampler: Sampler,
    save_base: str = "",
    folder: str = "vstate",
    sname: str = "sampler",
):
    """
    Save the provided sampler
    fname: name of the file to save the vstate to, saved to {save_base}{folder}/fname.nk
    """
    nqxpack.save(object=sampler, path=f"{save_base}{folder}/{sname}.nk")


def load_sampler(fname: str = "sampler.nk"):
    """
    Load the sampler from a file
    fname: name of the file to load the vstate from, loaded from fname
    symm_stage: starting symmetry stage to initialize the vstate in
    """
    print(f"Loading sampler from {fname} ...")
    sampler = nqxpack.load(fname)
    print(f"Loaded sampler: {sampler}")
    return sampler


def save_vstate(
    vstate: VariationalState,
    sampler: Sampler,
    save_base: str = "",
    folder: str = "vstate",
    vname: str = "vars",
    sname: str = "sampler",
):
    """
    Save the provided vstate by saving its variables to {save_base}{folder}/fname.mpack
    and sampler to {save_base}{folder}/sname.nk
    """
    os.makedirs(save_base + folder, exist_ok=True)
    print(f"Saving vstate to {save_base}{folder}/ ...")
    netsaver.save_variables(
        mpack_name=f"{save_base}{folder}/{vname}.mpack", vstate=vstate
    )
    save_sampler(sampler=sampler, save_base=save_base, folder=folder, sname=sname)


def load_vstate(
    sampler: Sampler,
    nets: Sequence,
    samples_per_rank: int,
    seed: int,
    n_discard_per_chain: int,
    chunk_size: int,
    save_base: str = "",
    load_base: str = "",
    load_stage: int = -1,
    vname: str = "vars",
    sname: str = "sampler",
    stage_diff: int = 0,
    new_sampler: bool = False,
) -> tuple[VariationalState, Sampler]:
    """
    Load a saved vstate, by loading its variables from {load_base}fname.mpack
    and sampler from {load_base}sname.nk.
    Resets the vstate to its zeroth symmetrization stage, whilst loading in with parameters
    formatted according to load_stage
    """
    print(f"Loading vstate from {load_base} ...")
    load_sampler_str = save_base + load_base + "/" + sname + ".nk"
    if not new_sampler:  # Load in sample from file
        try:
            sampler = load_sampler(load_sampler_str)
        except:  # noqa: E722
            warnings.warn("Error reloading sampler, defaulting to new sampler")
    load_vstate = nk.vqs.MCState(
        sampler,
        model=nets[load_stage - 1],
        n_samples_per_rank=samples_per_rank,
        seed=seed,
        sampler_seed=seed,
        n_discard_per_chain=n_discard_per_chain,
        chunk_size=chunk_size,
    )
    try:
        netsaver.load_variables(
            save_base + load_base + "/" + vname + ".mpack", load_vstate
        )
    except ValueError:
        raise ValueError(
            "Error loading vstate, check load_stage is correctly specified (should be value of last finished symmetry stage, with index starting from 1)"
        )

    resume_vstate = nk.vqs.MCState(
        sampler,
        model=nets[load_stage - 1 + stage_diff],
        n_samples_per_rank=samples_per_rank,
        seed=seed,
        sampler_seed=seed,
        n_discard_per_chain=n_discard_per_chain,
        chunk_size=chunk_size,
    )
    if stage_diff == 0:
        return load_vstate, sampler
    elif stage_diff > 0:
        new_params = add_module(
            old_params=load_vstate.variables["params"],
            new_params=resume_vstate.variables["params"],
        )
    else:
        new_params = remove_module(
            old_params=load_vstate.variables["params"],
            new_params=resume_vstate.variables["params"],
        )

    resume_vstate.variables = {"params": new_params}
    return resume_vstate, sampler
