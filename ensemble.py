import os
import h5py
from hloc.utils.io import find_pair
from hloc.utils.parsers import names_to_pair
import numpy as np


def get_keypoints(hfile, name):
    dset = hfile[name]['keypoints']
    return dset.__array__()


def get_matches(hfile, name0: str, name1: str) -> tuple[np.ndarray]:
    pair = names_to_pair(name0, name1)
    matches = hfile[pair]['matches0'].__array__()
    return matches


def merge_keypoints(files: list[str]):

    base = os.path.dirname(files[0])
    with h5py.File(f'{base}/merged_features.h5', 'w') as f_out:

        all_keys = set()
        for filepath in files:
            with h5py.File(filepath, 'r') as f:
                all_keys.update(f.keys())

        for image in all_keys:
            # create a new group in the output file for this image
            grp = f_out.create_group(image)

            keypoints_list = []
            scores_list = []  # List to store keypoints' scores
            image_size_list = []

            for features_filepath in files:
                with h5py.File(features_filepath, 'r') as f_f:
                    # Collect keypoints, descriptors, scores data and image_size
                    try:
                        keypoints_list.append(
                            f_f[image]['keypoints'].__array__())
                    except:
                        print(
                            f"no keypoint for the given image: {image}, {features_filepath}"
                        )

            # Concatenate keypoints, descriptors, and scores
            keypoints = np.concatenate(keypoints_list, axis=0)

            # Here we assume 'image_size' is a single array for each image, so we just take the last one.

            grp.create_dataset('keypoints', data=keypoints)


def merge_matches(matches_files: list[str], features_files: list[str],
                  pairs_path):
    # Open a new file for writing
    base = os.path.dirname(matches_files[0])
    with h5py.File(f'{base}/merged_matches.h5', 'w') as f_out:

        with open(str(pairs_path), 'r') as f:
            pairs = [p.split() for p in f.readlines()]

        for pair in pairs:
            # create a new group in the output file for this pair
            grp = f_out.create_group(f"{pair[0]}_{pair[1]}")

            matches_list = []
            scores_list = []
            offset = 0  # initialize offset

            for matches_filepath, features_filepath in zip(
                    matches_files, features_files):
                with h5py.File(matches_filepath,
                               'r') as m_f, h5py.File(features_filepath,
                                                      'r') as f_f:
                    try:
                        matches = get_matches(m_f, pair[0], pair[1])

                        # For valid matches, we add the offset to update the indices.
                        matches[matches != -1] += offset

                        matches_list.append(matches)
                    except:
                        print("no matching for the given pair. continue.")

                    # Update offset for next file
                    offset += f_f[pair[1]]['keypoints'].shape[0]
            if matches_list:
                # concatenate all matches and scores from all files for this pair
                matches = np.concatenate(matches_list, axis=0)

                # Create the 'matches0' and 'matching_scores0' datasets in this group
                grp.create_dataset('matches0', data=matches)
