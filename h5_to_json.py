import h5py
import json
import numpy as np


def convert_h5_to_json(h5_file, json_file):

    def recursive_convert(group):
        data = {}
        for key, item in group.items():
            if isinstance(item, h5py.Dataset):
                data[key] = item[()].tolist()
            elif isinstance(item, h5py.Group):
                data[key] = recursive_convert(item)
        return data

    with h5py.File(h5_file, 'r') as file:
        data = recursive_convert(file)

    with open(json_file, 'w') as file:
        json.dump(data, file)


h5_file = '/home/jsmoon/outputs/haiper-fountain/matches-patch2pix_pairs-sfm.h5'
json_file = '/home/jsmoon/outputs/haiper-fountain/matches-patch2pix_pairs-sfm.json'

convert_h5_to_json(h5_file, json_file)