import netket as nk
import flax


def load_variables(mpack_name: str, vstate) -> nk.vqs.VariationalState:
    with open(mpack_name, "rb") as f:
        variables = flax.serialization.from_bytes(vstate.variables, f.read())
    vstate.variables = variables
    return vstate


def load_variables_tree(mpack_name: str, target_tree):
    with open(mpack_name, "rb") as f:
        variables = flax.serialization.from_bytes(target_tree, f.read())
    return variables


def save_variables(mpack_name: str, vstate):
    with open(mpack_name, "wb") as f:
        f.write(flax.serialization.to_bytes(vstate.variables))
