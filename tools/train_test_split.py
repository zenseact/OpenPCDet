import json
import os
import random


def progress(list_in: list, freq: int = 100, desc: str = ""):
    """Generator for printing progress.

    Args:
        list_in (list): list to loop over
        freq (int, optional): Frequency for printing progress. Defaults to 100.
        desc (str, optional): Description to print with progress.. Defaults to "".

    Yields:
        [type]: [description]
    """
    total = len(list_in)
    for i, el in enumerate(list_in):
        if i % freq == 0:
            print(f"{desc}: {i+1}/{total}")
        yield el


def get_files_from_anno_idx(dataset_path: str, idx: int) -> dict:
    """Given a dataset path and sequence idx, returns paths to all relevant files.

    Args:
        dataset_path (str): path to root of dataset
        idx (int): index of sequence of interest

    Returns:
        dict: Dict containing all relevant files for sequence index idx.
    """

    def get_files_from_subfolder(path: str, idx: int, ext: str = None) -> list:
        """Given path, finds folder corresponding to idx and returns all files contained in folder.

        Args:
            path (str): Path where subfolders are located.
            idx (int): index of sequence of interest
            ext (str, optional): File extension. All files if not specified. Defaults to None.

        Returns:
            [list]: list of files (complete path)
        """
        folders = [fold for fold in os.listdir(path) if ".DS_Store" not in fold]
        try:
            folder = [folds for folds in folders if int(folds.split("_")[0]) == idx][0]
        except:
            print("WARNING: Missing data for folder")
            print(path)
            print("and idx")
            print(idx)
            return []

        files = os.listdir(os.path.join(path, folder))

        if ext is not None:
            files = [file for file in files if file.endswith(ext)]

        return [os.path.join(path, folder, file) for file in files]

    def get_date_and_vehicle_str(path: str, idx: int):
        """Finds vehicle name and date for sequence idx.

        Args:
            path (str): path to base this on. Should contain subfolders, such as the folder blurred_images.
            idx (int): seqeunce idx

        Returns:
            str, str: date_str and vehicle name
        """
        folders = [fold for fold in os.listdir(path) if ".DS_Store" not in fold]
        folder = [folds for folds in folders if int(folds.split("_")[0]) == idx][0]
        date_str = folder.split("_")[1].split("T")[0]
        vehicle_str = os.listdir(os.path.join(path, folder))[0].split("_")[0]
        return date_str, vehicle_str

    anno_path = os.path.join(dataset_path, "annotations_kitti/dynamic_objects")
    lidar_dets_path = os.path.join(dataset_path, "detections", "lidar")
    camera_dets_path = os.path.join(dataset_path, "detections", "camera")
    calib_path = os.path.join(dataset_path, "calibration")
    blurred_imgs_path = os.path.join(dataset_path, "blurred_images")
    dnat_imgs_path = os.path.join(dataset_path, "dnat_images")
    lidar_path = os.path.join(dataset_path, "lidar_data")
    range_lidar_path = os.path.join(dataset_path, "range_lidar_data")
    vehicle_data = os.path.join(dataset_path, "vehicle_data")
    oxts_data = os.path.join(dataset_path, "oxts_data")

    files = {}

    # Get annotation file
    anno = [
        file
        for file in os.listdir(anno_path)
        if f"{idx}.txt" == file.strip("0") or idx == 0 and len(file.strip("0")) == 4
    ]
    if len(anno):
        files["anno"] = [os.path.join(anno_path, anno[0])]
    else:
        files["anno"] = []

    # Get detections
    dets_idx_path = os.path.join(lidar_dets_path, str(idx))
    files["dets_lidar"] = (
        [os.path.join(dets_idx_path, file) for file in os.listdir(dets_idx_path)]
        if os.path.exists(dets_idx_path)
        else []
    )
    dets_idx_path = os.path.join(camera_dets_path, str(idx))
    files["dets_camera"] = (
        [os.path.join(dets_idx_path, file) for file in os.listdir(dets_idx_path)]
        if os.path.exists(dets_idx_path)
        else []
    )

    # Get images
    files["blurred_imgs"] = get_files_from_subfolder(blurred_imgs_path, idx, ".png")
    files["dnat_imgs"] = get_files_from_subfolder(dnat_imgs_path, idx, ".png")

    # Get lidar sweeps
    files["lidar"] = get_files_from_subfolder(lidar_path, idx, ".npy")
    files["lidar_range"] = get_files_from_subfolder(range_lidar_path, idx, ".npy")

    # Get vehicle data
    files["vehicle_data"] = get_files_from_subfolder(vehicle_data, idx, ".hdf5")

    # Get OXTS data
    files["oxts_data"] = get_files_from_subfolder(oxts_data, idx, ".hdf5")

    date_str, vehicle_str = get_date_and_vehicle_str(blurred_imgs_path, idx)
    files["calibration"] = [os.path.join(calib_path, f"{vehicle_str}_{date_str}.json")]

    return files


def make_split(dataset_path, anno_path, anno_filenames, test_set_size):
    # Find all files with non-empty annotations files
    anno_files_keep = []
    for file in progress(anno_filenames, desc="Splitting train/test"):
        with open(os.path.join(anno_path, file)) as f:
            lines = f.readlines()

        if not len(lines):
            continue

        anno_files_keep.append(file)

    # Randomly sample test set
    test_anno_filenames = random.sample(anno_files_keep, test_set_size)
    train_anno_filenames = list(set(anno_files_keep) - set(test_anno_filenames))

    assert len(test_anno_filenames) == len(set(test_anno_filenames))
    assert len(test_anno_filenames) + len(train_anno_filenames) == len(anno_files_keep)

    return train_anno_filenames, test_anno_filenames


def get_old_split(train_json, test_json):
    with open(train_json, "r") as f:
        train = json.load(f)

    with open(test_json, "r") as f:
        test = json.load(f)

    train_anno_filenames = [f"{str(id).zfill(6)}.txt" for id in train.keys()]
    test_anno_filenames = [f"{str(id).zfill(6)}.txt" for id in test.keys()]

    return train_anno_filenames, test_anno_filenames


#################### PARAMETERS TO CHANGE #######################
dataset_path = "path/to/dataset/root"
anno_path = os.path.join(dataset_path, "annotations_kitti/dynamic_objects")
test_set_size = 788

train_json = os.path.join(dataset_path, "train.json")
test_json = os.path.join(dataset_path, "test.json")
use_old_split = True
update_train = False
update_test = False
update_unlabeled = False
#################################################################


anno_filenames = [file for file in os.listdir(anno_path) if file.endswith(".txt")]

if use_old_split:
    train_anno_filenames, test_anno_filenames = get_old_split(train_json, test_json)
else:
    train_anno_filenames, test_anno_filenames = make_split(
        dataset_path, anno_path, anno_filenames, test_set_size
    )

unlabeled_idx = list(
    set(list(range(1, 6667)))
    - set([int(file[:-4]) for file in train_anno_filenames])
    - set([int(file[:-4]) for file in test_anno_filenames])
    - set([917])  # This sequence is missing for some reason
)

unlabeled_filenames = [f"{idx}.txt" for idx in unlabeled_idx]

if update_train:
    train_dict = {}
    for file in progress(
        train_anno_filenames, desc="Finding relevant files for train."
    ):
        stripped = file.strip("0")
        if len(stripped) == 4:
            stripped = "0.txt"

        idx = int(stripped.split(".")[0])
        train_dict[idx] = get_files_from_anno_idx(dataset_path, idx)

if update_test:
    test_dict = {}
    for file in progress(test_anno_filenames, desc="Finding relevant files for test."):
        idx = int(file.strip("0").split(".")[0])
        test_dict[idx] = get_files_from_anno_idx(dataset_path, idx)

if update_unlabeled:
    unlabeled_dict = {}
    for file in progress(
        unlabeled_filenames, desc="Finding relevant files for unlabeled."
    ):
        stripped = file.strip("0")
        if len(stripped) == 4:
            stripped = "0.txt"

        idx = int(file.strip("0").split(".")[0])
        unlabeled_dict[idx] = get_files_from_anno_idx(dataset_path, idx)

if update_train:
    with open(os.path.join(dataset_path, "train1.json"), "w") as f:
        json.dump(train_dict, f)

if update_test:
    with open(os.path.join(dataset_path, "test1.json"), "w") as f:
        json.dump(test_dict, f)

if update_unlabeled:
    with open(os.path.join(dataset_path, "unlabeled1.json"), "w") as f:
        json.dump(unlabeled_dict, f)
