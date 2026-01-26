import glob
import netket as nk
import flax
import re
from collections.abc import Sequence
import json
import numpy as np
import os
from vmc.data.funcs import get_ievav, rel_err


def get_matching(dir: str, match_str: str):
    """
    Return list of paths to all files in dir containing match_str
    """
    return glob.glob(f"{dir}*{match_str}*")


def get_int(filename: str, prefix: str) -> int:
    with open(filename) as file:
        contents = file.read()
        match = re.search(rf"{prefix} (\d+)", contents)
        if match:
            number = int(match.group(1))
            return number
        else:
            print("Number not found")


def get_mpack_paths(dir: str):
    """
    Return list of paths to all .mpack files in dir, netket variational states are saved as .mpack at the end of optimization
    """
    return glob.glob(dir + "*.mpack")


def read_from_str(
    params: dict, key: str, match_str: str, return_type=float, end_str=","
):
    """
    Read the value in the string at params[key] = string, which comes immediately after match_str
    """
    value = params[key]
    start_index = value.index(match_str) + len(match_str)
    end_index = start_index + value[start_index:].index(end_str)
    number = return_type(value[start_index:end_index])
    return number


def load_vstate(mpack_name: str, network, sampler) -> nk.vqs.VariationalState:
    var_state = nk.vqs.MCState(sampler, model=network)
    with open(mpack_name, "rb") as f:
        variables = flax.serialization.from_bytes(var_state.variables, f.read())
    var_state.parameters = variables
    return var_state


def get_indices(param_seq: Sequence, key: str, match_str: str):
    """
    Get all i for which params_seq[i][key] == match_str
    """
    return [i for i, param in enumerate(param_seq) if param[key] == match_str]


def sort_indices(param_seq: Sequence, sort_key: str):
    """
    Return a list of the indices of param_seq sorted according to the value of param_seq[i][sort_key]
    """
    return [
        i
        for i, _ in sorted(
            enumerate([p[sort_key] for p in param_seq]), key=lambda x: x[1]
        )
    ]


def get_results(base: str, nums: tuple = None, opt_min=1, opt_max=100):
    if nums is None:
        files = [base]
    else:
        files = [base + f"{num}/" for num in nums]
    opt_is = [
        "",
    ] + [i for i in range(opt_min, opt_max + 1)]
    print(files)
    # print([[file+f"opt{i}.log" for i in opt_is] for file in files])
    # logs = [[get_matching(file+f"opt{i}", "log") for i in opt_is] for file in files]
    logs = [
        [file + f"opt{i}.log" for i in opt_is if os.path.exists(file + f"opt{i}.log")]
        for file in files
    ]
    # print(logs)
    results = [[json.load(open(log)) for log in logs_i] for logs_i in logs]
    if len(files) == 1:
        files = files[0]
        results = results[0]
    # print(files)
    return results, files


def get_spinspin(res: dict, Ni: int, Nj: int, string="SzSz"):
    S = np.zeros((Ni, Nj))
    S_error = np.zeros((Ni, Nj))
    if string[-1] != "t":
        for i in range(Ni):
            for j in range(i + 1):
                S[i, j] = res[f"{string}(({i}, {j}))"]["mean"]
                S[j, i] = res[f"{string}(({i}, {j}))"]["mean"]
                S_error[i, j] = res[f"{string}(({i}, {j}))"]["error_of_mean"]
                S_error[j, i] = res[f"{string}(({i}, {j}))"]["error_of_mean"]
        assert np.allclose(S, S.T), f"{string} matrix is not symmetric"

    elif string[-1] == "t":
        for i in range(Ni):
            for j in range(Nj):
                S[i, j] = res[f"{string}(({i}, {j}))"]["mean"]
                S_error[i, j] = res[f"{string}(({i}, {j}))"]["error_of_mean"]

    S_error = np.nan_to_num(S_error, nan=0.0)
    # assert np.allclose(np.diag(S),0.25), "i,i correlations not 0.25 (for S = 1/2)"
    return S, S_error


def load_data(path, Ni, step, max_steps, index_fn, opt_max=2):
    data = np.zeros((Ni, max_steps + 1), dtype="object")
    files = np.zeros((Ni, max_steps + 1), dtype="object")
    for i in range(Ni):
        data[i, :], files[i, :] = get_results(
            path, nums=[index_fn(i, j) for j in max_steps + 1], opt_max=opt_max
        )
    return data, files


def get_mins(data, N):
    min_energies, min_vscores, energy_std, vscore_std, min_variances = (
        [],
        [],
        [],
        [],
        [],
    )
    for i in range(data.shape[0]):
        min_energy = 0
        for j, results in enumerate(data[i]):
            if len(results) != 0:
                try:
                    iters, energy, vscore, acceptance, variance = get_ievav(
                        results[-1], N, True
                    )  # takes from the final results, i.e last stage
                except TypeError:
                    print("Error for ", i, j)
                energy = energy / (4 * N)
                final_energy = energy[-50:].mean()
                if final_energy < min_energy:
                    min_energy = final_energy
                    min_energy_std = energy[-50:].std()
                    min_vscore = vscore[-50:].mean()
                    min_vscore_std = vscore[-50:].std()
                    min_variance = variance[-50:].mean()

        min_energies.append(min_energy)
        min_vscores.append(min_vscore)
        energy_std.append(min_energy_std)
        vscore_std.append(min_vscore_std)
        min_variances.append(min_variance)

    return (
        np.array(min_energies),
        np.array(min_vscores),
        np.array(energy_std),
        np.array(vscore_std),
        np.array(min_variances),
    )


def get_minimum(
    data,
    groups,
    index_fn,
    N,
    n_iters,
    group_labels=None,
    iters_per_file=None,
    ed_energies=None,
):
    energies = np.zeros((len(groups), 2))
    energy_mean = np.zeros((len(groups), 2))
    vscores = np.zeros((len(groups),))
    vscore_mean = np.zeros((len(groups), 2))
    min_runs = np.zeros((len(groups),))
    relative_errors = np.zeros((len(groups),))
    runs = np.full((len(groups), len(data[0]), 5, n_iters), np.nan)
    for i in range(len(groups)):
        print(f"group = {groups[i]}")
        if group_labels is not None:
            print(group_labels[i])
        min_energy = 0
        vscore_min = None
        k_energy = []
        k_vscore = []
        for j, results in enumerate(data[i]):
            try:
                result = results[-1]
                iters, energy, vscore, acceptance, variance = get_ievav(result, N, True)
                energy = energy / (4 * N)
                runs[i, j, :, : len(iters)] = np.stack(
                    (iters, energy, vscore, acceptance, variance), axis=0
                )
                print(
                    f"run = {j} (total run = {index_fn(i, j)}):, <E> = {energy[-50:].mean()}+-{energy[-50:].std()}, <V> = {vscore[-50:].mean()}+-{vscore[-50:].std()}, <A> = {acceptance[-50:].mean()}"
                )
                if ed_energies is not None:
                    rel_error = rel_err(energy[-50:].mean(), ed_energies[i])
                    print(f"Relative error = {rel_error:.3e}")
                final_energy = energy[-50:].mean()
                if abs(final_energy) < 10 and abs(vscore[-50:].mean()) < 10:
                    k_energy.append(final_energy)
                    k_vscore.append(vscore[-50:].mean())
                    if final_energy < min_energy:
                        min_energy = final_energy
                        min_std = energy[-50:].std()
                        vscore_min = vscore[-50:].mean()
                        relative_errors[i] = (
                            rel_err(min_energy, ed_energies[i])
                            if ed_energies is not None
                            else np.nan
                        )
                        min_runs[i] = index_fn(i, j)

            except IndexError:
                print(f"run = {j} (total run = {index_fn(i, j)}): No data")
                continue
            except ValueError:
                print(f"run = {j} (total run = {index_fn(i, j)}): Mismatched lengths")
                continue

        energy_mean[i] = np.mean(k_energy), np.std(k_energy)
        vscore_mean[i] = np.mean(k_vscore), np.std(k_vscore)

        print(f"Mean energy = {np.mean(k_energy):.5f} +- {np.std(k_energy):.5f}")
        print(f"Mean vscore = {np.mean(k_vscore):.5f} +- {np.std(k_vscore):.5f}")

        energies[i] = min_energy, min_std
        vscores[i] = vscore_min
    print("Min runs", min_runs)
    return energies, energy_mean, vscores, vscore_mean, min_runs, runs, relative_errors
