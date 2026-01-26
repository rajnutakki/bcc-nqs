import vmc.utils.serialize as serialize
import netket as nk
import json
import jax
import numpy as np
import jax.numpy as jnp
from typing import Sequence, Optional
from nets.utils.serialize import load_variables
from vmc.expectation_value.observables import ObservableParser, compute_phases
from vmc.optimization.utils import add_module
import inspect


def get_init_args(obj):
    sig = inspect.signature(obj.__class__.__init__)
    params = list(sig.parameters.keys())[1:]  # skip 'self'
    init_args = {k: getattr(obj, k) for k in params if hasattr(obj, k)}
    return init_args


def compute(
    dirname: str,
    n_samples_per_chain: int,
    n_chains: int,
    n_discard_per_chain: int,
    chunk_size: int,
    observables: Sequence[str],  # Names of observables to compute
    save_type: str,
    save_file_name: str = "expectation_values",
    final_stage: Optional[int] = None,
    symmetry_stage: Optional[int] = None,
    little_group_id: Optional[int] = None,
    spin_flip_symmetric: Optional[bool] = None,
):
    """
    Load in the system, network (net_name) and vstate of the minimum energy state in {dirname}/post/checkpoint, then compute the expectation value of
    of the operators with the parameters provided. Results are saved to {dirname}/expectation_values.json.
    Returns the results_dict of form {"operator_name": expectation_value,...}
    """
    final_stage = 0 if final_stage is None else final_stage

    if dirname == "":
        json_path = f"{save_type}.json"
        if save_type == "post":
            post_path = "post"
        else:
            post_path = ""

    else:
        json_path = dirname + f"/{save_type}.json"
        if save_type == "post":
            post_path = dirname + "/post"
        else:
            post_path = dirname

    min_index, system, network, load_symmetry_stage, _ = serialize.load(
        fname=json_path, save_type=save_type
    )
    # Check if we need to change some symmetrization stage arguments of system
    # print("Loaded system:", vars(system))
    if little_group_id is not None or spin_flip_symmetric is not None:
        new_args = get_init_args(system)
        if little_group_id is not None:
            print(
                f"Changing little_group_id from {system.little_group_id} to {little_group_id}"
            )
            new_args["little_group_id"] = little_group_id
        if spin_flip_symmetric is not None:
            print(
                f"Changing spin_flip_symmetric from {system.spin_flip_symmetric} to {spin_flip_symmetric}"
            )
            new_args["spin_flip_symmetric"] = spin_flip_symmetric
        system = type(system)(**new_args)
        # print("Modified system:", vars(system))
    networks = [f(network) for f in system.symmetrizing_functions]

    if symmetry_stage is None or symmetry_stage == 0:
        symmetry_stage = load_symmetry_stage
        vars_path2 = f"{post_path}/vstate/vars.mpack"
    else:
        vars_path2 = f"{post_path}/vstate{symmetry_stage}/vars.mpack"

    if save_type == "post":
        vars_path1 = f"{post_path}/vars{min_index}.mpack"
    else:
        vars_path1 = f"{post_path}/{save_type}.mpack"

    print(f"Loading network of symmetry stage {symmetry_stage}...")
    print(networks[symmetry_stage - 1])
    sampler = system.sampler_t(system.hilbert_space, n_chains=n_chains)
    vstate = nk.vqs.MCState(sampler, model=networks[symmetry_stage - 1])
    try:
        vstate = load_variables(vars_path1, vstate)
    except FileNotFoundError:
        print(
            f"No mpack found at {vars_path1}, attempting to load from {vars_path2} instead..."
        )
        vstate = load_variables(vars_path2, vstate)
    while symmetry_stage < final_stage:
        print(f"Increasing symmetry stage to {symmetry_stage + 1}...")
        new_vstate = nk.vqs.MCState(sampler, model=networks[symmetry_stage])
        new_params = add_module(
            old_params=vstate.variables["params"],
            new_params=new_vstate.variables["params"],
        )
        new_vstate.variables = {"params": new_params}
        vstate = new_vstate
        symmetry_stage += 1

    observables = ObservableParser(observables, system)
    vstate.n_samples = n_samples_per_chain * n_chains
    vstate.n_discard_per_chain = n_discard_per_chain
    vstate.chunk_size = chunk_size
    print(f"vstate.chunk_size = {vstate.chunk_size}")
    print(f"vstate.n_chains = {vstate.sampler.n_chains}")
    print(f"vstate.n_samples = {vstate.n_samples}")
    print(f"vstate.n_discard_per_chain = {vstate.n_discard_per_chain}")
    if chunk_size:
        vstate.chunk_size = chunk_size
    results_dict = {
        "n_chains": vstate.sampler.n_chains,
        "n_samples": vstate.n_samples,
        "n_discard_per_chain": vstate.n_discard_per_chain,
    }
    for name, operator in observables.operators.items():
        print(f"Computing expectation value for {name}...")
        result = vstate.expect(operator.to_jax_operator())
        result_dict = (
            result.__dict__
        )  # convert all of the attributes and their values to a dictionary
        # Convert to types compatible with json
        for key, value in result_dict.items():
            if isinstance(value, jax.Array):
                result_dict[key] = float(
                    np.real(complex(value))
                )  # cannot go directly from jax.Array with complex dtype to float, so take real part
        results_dict[name] = result_dict
    print("observables.compute_signs", observables.compute_signs)
    # Compute phase information
    n_chunks = (
        vstate.n_samples // vstate.chunk_size
        if vstate.chunk_size is not None
        else vstate.n_samples
    )
    if observables.compute_signs:
        samples = vstate.sample()
        samples = samples.reshape((-1, system.graph.n_nodes))  # flatten samples
        phases, j1_signs, j2_signs = (
            np.zeros((samples.shape[0],)),
            np.zeros((samples.shape[0],)),
            np.zeros((samples.shape[0],)),
        )
        for i_chunk in range(n_chunks):
            (
                phases[i_chunk * chunk_size : (i_chunk + 1) * chunk_size],
                j1_signs[i_chunk * chunk_size : (i_chunk + 1) * chunk_size],
                j2_signs[i_chunk * chunk_size : (i_chunk + 1) * chunk_size],
            ) = compute_phases(
                samples[i_chunk * chunk_size : (i_chunk + 1) * chunk_size],
                vstate.log_value(
                    samples[i_chunk * chunk_size : (i_chunk + 1) * chunk_size]
                ),
                system,
            )

        results_dict["phases"] = phases.tolist()
        results_dict["j1_signs"] = j1_signs.tolist()
        results_dict["j2_signs"] = j2_signs.tolist()

    if dirname != "":
        save_file_name = "/" + save_file_name

    save_file = f"{dirname}{save_file_name}"

    # Save all results
    with open(save_file + ".json", "a") as f:
        json.dump(results_dict, f)

    jnp.save(save_file + "_samples.npy", vstate.samples)

    return results_dict
