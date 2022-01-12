import copy
import os

import numpy as np
import yaml
from torch.utils import data
from tqdm import tqdm

from ...utils import box_utils, common_utils
from ...utils.object3d_kitti import Object3d
from ..dataset import DatasetTemplate


class ZenseactDataset(DatasetTemplate):
    def __init__(
        self,
        dataset_cfg=None,
        class_names=None,
        training=True,
        root_path=None,
        logger=None,
    ):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
        )
        self.range_lidar_path = os.path.join(
            self.root_path, "range_lidar_data")
        self.anotation_path = os.path.join(
            self.root_path, "annotations_kitti", "dynamic_objects_lidar"
        )

        file_paths = os.listdir(self.range_lidar_path)
        annotation_paths = os.listdir(self.anotation_path)
        self.annotations = {}
        file_paths_with_annotations = []
        for fp in tqdm(file_paths, desc="Initializing dataset."):
            idx = fp.split("_")[0]
            anno_fp = os.path.join(self.anotation_path, f"{idx}.txt")
            if f"{idx}.txt" in annotation_paths:
                with open(anno_fp, "r") as file:
                    lines = file.readlines()

                # remove all files that have 0 annotations
                if not len(lines):
                    continue
                # remove all files that have only dont cares
                if not any(
                    [line.split()[0].lower() != "DontCare".lower()
                     for line in lines]
                ):
                    continue

                obj_list = [Object3d(line) for line in lines]
                for obj in obj_list:
                    x, y, _ = obj.loc
                    obj.loc[0] = y
                    obj.loc[1] = -x
                    obj.ry -= np.pi / 2

                annotations = {}
                annotations["name"] = np.array(
                    [obj.cls_type for obj in obj_list])
                annotations["truncated"] = np.array(
                    [obj.truncation for obj in obj_list])
                annotations["occluded"] = np.array(
                    [obj.occlusion for obj in obj_list])
                annotations["alpha"] = np.array(
                    [obj.alpha for obj in obj_list])
                annotations["bbox"] = np.concatenate(
                    [obj.box2d.reshape(1, 4) for obj in obj_list], axis=0
                )
                annotations["dimensions"] = np.array(
                    [[obj.l, obj.h, obj.w] for obj in obj_list]
                )  # lhw(camera) format
                annotations["location"] = np.concatenate(
                    [obj.loc.reshape(1, 3) for obj in obj_list], axis=0
                )
                annotations["rotation_y"] = np.array(
                    [obj.ry for obj in obj_list])
                annotations["score"] = np.array(
                    [obj.score for obj in obj_list])
                annotations["difficulty"] = np.array(
                    [obj.level for obj in obj_list], np.int32)

                num_objects = len(
                    [obj.cls_type for obj in obj_list if obj.cls_type != "DontCare"]
                )
                num_gt = len(annotations["name"])
                index = list(range(num_objects)) + \
                    [-1] * (num_gt - num_objects)
                annotations["index"] = np.array(index, dtype=np.int32)

                loc = annotations["location"]
                dims = annotations["dimensions"]
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                rots = annotations["rotation_y"]
                gt_boxes_lidar = np.concatenate(
                    [loc, l, w, h, rots[..., np.newaxis]], axis=1)
                annotations["gt_boxes_lidar"] = gt_boxes_lidar

                annotations = common_utils.drop_info_with_name(
                    annotations, name="DontCare")

                self.annotations.update({anno_fp: annotations})
                file_paths_with_annotations.append(fp)

        split_idx = int(len(file_paths_with_annotations) * 0.75)

        if training:
            self.file_paths = file_paths_with_annotations[:split_idx]
        else:
            self.file_paths = file_paths_with_annotations[split_idx:]

    @staticmethod
    def generate_prediction_dicts(
        batch_dict, pred_dicts, class_names, output_path=None
    ):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                "name": np.zeros(num_samples),
                "truncated": np.zeros(num_samples),
                "occluded": np.zeros(num_samples),
                "alpha": np.zeros(num_samples),
                "bbox": np.zeros([num_samples, 4]),
                "dimensions": np.zeros([num_samples, 3]),
                "location": np.zeros([num_samples, 3]),
                "rotation_y": np.zeros(num_samples),
                "score": np.zeros(num_samples),
                "boxes_lidar": np.zeros([num_samples, 7]),
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict["pred_scores"].cpu().numpy()
            pred_boxes = box_dict["pred_boxes"].cpu().numpy()
            pred_labels = box_dict["pred_labels"].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_boxes_camera = copy.deepcopy(pred_boxes)
        
            pred_boxes_img = np.ones(shape=(len(pred_boxes_camera), 4))
            # this is because of his stupid eval see removal of object with small bounding boxes in eval.py
            pred_boxes_img[:,3] = 50

            pred_dict["name"] = np.array(class_names)[pred_labels - 1]
            pred_dict["alpha"] = (
                -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0])
                + pred_boxes_camera[:, 6]
            )
            pred_dict["bbox"] = pred_boxes_img
            pred_dict["dimensions"] = pred_boxes_camera[:, 3:6]
            pred_dict["location"] = pred_boxes_camera[:, 0:3]
            pred_dict["rotation_y"] = pred_boxes_camera[:, 6]
            pred_dict["score"] = pred_scores
            pred_dict["boxes_lidar"] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict["frame_id"][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict["frame_id"] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ("%s.txt" % frame_id)
                with open(cur_det_file, "w") as f:
                    bbox = single_pred_dict["bbox"]
                    loc = single_pred_dict["location"]
                    dims = single_pred_dict["dimensions"]  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print(
                            "%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f"
                            % (
                                single_pred_dict["name"][idx],
                                single_pred_dict["alpha"][idx],
                                bbox[idx][0],
                                bbox[idx][1],
                                bbox[idx][2],
                                bbox[idx][3],
                                dims[idx][1],
                                dims[idx][2],
                                dims[idx][0],
                                loc[idx][0],
                                loc[idx][1],
                                loc[idx][2],
                                single_pred_dict["rotation_y"][idx],
                                single_pred_dict["score"][idx],
                            ),
                            file=f,
                        )

        return annos

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        folder_path = os.path.join(
            self.range_lidar_path, self.file_paths[index])
        lidar_range_file_names = [
            f for f in os.listdir(folder_path) if f.endswith(".npy")
        ]
        lidar_fn = lidar_range_file_names[len(lidar_range_file_names) // 2]
        lidar_fp = os.path.join(self.range_lidar_path,
                                self.file_paths[index], lidar_fn)

        pointcloud = np.load(lidar_fp, allow_pickle=True)
        # Note that the order here is becuase of differences in definition in Zenseact and OpenPcDet coordinate frames
        pointcloud = np.c_[
            pointcloud["y"],
            -pointcloud["x"],
            pointcloud["z"],
            pointcloud["intensity"] / 255.0,
        ]

        get_item_list = self.dataset_cfg.get("GET_ITEM_LIST", ["points"])

        file_index = self.file_paths[index].split("_")[0]
        input_dict = {"frame_id": file_index}

        if "points" in get_item_list:
            if self.dataset_cfg.FOV_POINTS_ONLY:
                x = pointcloud[:, 0]
                y = pointcloud[:, 1]
                fov_flag = abs(np.arctan2(y, x)) < np.deg2rad(65)
                pointcloud = pointcloud[fov_flag]

            input_dict["points"] = pointcloud

        anno_fp = os.path.join(self.anotation_path, f"{file_index}.txt")
        annotations = self.annotations[anno_fp]

        # TODO: MAKE SURE TO REMOVE THE ONES THAT DO NOT HAVE 3D PROPS.
        input_dict.update(
            {"gt_names": annotations["name"],
                "gt_boxes": annotations["gt_boxes_lidar"]}
        )

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    def evaluation(self, det_annos, class_names, **kwargs):

        from ..kitti.kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)

        eval_gt_annos = [copy.deepcopy(
            self.annotations[
                os.path.join(
                    self.anotation_path, f"{det['frame_id']}.txt")]
        )
            for det in det_annos

        ]

        
        for anno in eval_gt_annos:
            rot = anno["rotation_y"]
            loc = anno["location"]
            anno["rotation_y"] = -rot - np.pi/2
            anno["location"] = np.concatenate((-loc[:,1:2], -loc[:,2:3], loc[:,0:1]), axis=1)
            
        for det in eval_det_annos:
            rot = det["rotation_y"]
            loc = det["location"]
            dims = det["dimensions"]
            det["rotation_y"] = -rot - np.pi/2
            det["location"] = np.concatenate((-loc[:,1:2], -loc[:,2:3], loc[:,0:1]), axis=1)
            det["dimensions"] = dims[:, [0, 2, 1]]


        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
            eval_gt_annos, eval_det_annos, class_names
        )

        return ap_result_str, ap_dict


class ZenseactRangeDataset(DatasetTemplate):
    def __init__(
        self,
        dataset_cfg=None,
        class_names=None,
        training=True,
        root_path=None,
        logger=None,
    ):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
        )
        self.range_lidar_path = os.path.join(
            self.root_path, "range_lidar_data")

        file_paths = os.listdir(self.range_lidar_path)
        self.all_file_paths = []
        for fp in tqdm(file_paths, desc="Initializing dataset"): 
            folder_path = os.path.join(self.range_lidar_path, fp)
            self.all_file_paths += [
                os.path.join(folder_path, file) 
                for file in os.listdir(folder_path) 
                if file.endswith(".npy")
            ]
            
    def __len__(self):
        return len(self.all_file_paths)

    
    def __getitem__(self, index):

        pointcloud = np.load(self.all_file_paths[index], allow_pickle=True)
        # Note that the order here is becuase of differences in definition in Zenseact and OpenPcDet coordinate frames
        pointcloud = np.c_[
            pointcloud["y"],
            -pointcloud["x"],
            pointcloud["z"],
            pointcloud["intensity"] / 255.0,
        ]

        get_item_list = self.dataset_cfg.get("GET_ITEM_LIST", ["points"])

        file_index = self.all_file_paths[index].split("/")[-1][:-4]
        input_dict = {"frame_id": file_index}

        if "points" in get_item_list:
            if self.dataset_cfg.FOV_POINTS_ONLY:
                x = pointcloud[:, 0]
                y = pointcloud[:, 1]
                fov_flag = abs(np.arctan2(y, x)) < np.deg2rad(65)
                pointcloud = pointcloud[fov_flag]

            input_dict["points"] = pointcloud

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict