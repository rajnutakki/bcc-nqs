import json
import os
from collections.abc import Sequence


def write_attributes_json(filename: str, group_name: str, **kwargs):
    if os.path.exists(filename):
        updating_json = json.load(open(filename))
    else:
        updating_json = {}

    new_json = {}
    for key, val in kwargs.items():
        new_json[f"{group_name}/{key}"] = str(val)

    updating_json.update(new_json)
    json.dump(updating_json, open(filename, mode="w"))


def write_json(filename: str, gnames: Sequence, objs: Sequence):
    for gname, obj in zip(gnames, objs):
        write_attributes_json(filename, gname, **vars(obj))
