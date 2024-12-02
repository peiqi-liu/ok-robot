"""
 This file implements R3DSemanticDataset
    in Clip-fields (https://github.com/notmahi/clip-fields) project
 Most codes are adapted from:
    1. https://github.com/notmahi/clip-fields/blob/main/dataloaders/record3d.py

License:
MIT License

Copyright (c) 2024 Nur Muhammad "Mahi" Shafiullah

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import json
from pathlib import Path
from typing import List, Optional
from zipfile import ZipFile

import liblzfse
import numpy as np
import open3d as o3d
import tqdm
from PIL import Image
from quaternion import as_rotation_matrix, quaternion
from torch.utils.data import Dataset

from dataloaders.scannet_200_classes import CLASS_LABELS_200


class R3DSemanticDataset(Dataset):
    def __init__(
        self,
        path: str,
        custom_classes: Optional[List[str]] = CLASS_LABELS_200,
        subsample_freq = 1,
        x1 = None,
        y1 = None,
        x2 = None,
        y2 = None,
        z_offset = None,
    ):
        # x1 = -6.013387
        # y1 = 1.741216
        # x2 = -6.015079
        # y2 = 1.592178
        # z_offset = -1.406663
        # assert(x1 is None and y1 is None and x2 is None and y2 is None and z_offset is None) \
        #         or (x1 is not None and y1 is not None and x2 is not None and y2 is not None and z_offset is not None),\
        #         'x1, y1, x2, y2, z_offset would either be all None, or all not None'
        # if x1 is not None:
        #     x_offset = x1
        #     y_offset = y1
        #     theta_offset =  np.arctan2((y2 - y1), (x2 - x1))
        #     self.n2r_matrix = np.array([
        #         [np.cos(theta_offset), np.sin(theta_offset), 0, 0],
        #         [-np.sin(theta_offset), np.cos(theta_offset), 0, 0],
        #         [0, 0, 1, 0],
        #         [0, 0, 0, 1]
        #     ]) @ \
        #     np.array([
        #         [1, 0, 0, -x_offset],
        #         [0, 1, 0, -y_offset],
        #         [0, 0, 1, -z_offset],
        #         [0, 0, 0, 1]
        #     ])
        if path.endswith((".zip", ".r3d")):
            self._path = ZipFile(path)
        else:
            self._path = Path(path)

        if custom_classes:
            self._classes = CLASS_LABELS_200 + custom_classes
        else:
            self._classes = CLASS_LABELS_200
        self._classes = list(set(self._classes))
        print("The labels you use for OWL-ViT is ", str(self._classes))
        
        self._subsample_freq = subsample_freq
        self._reshaped_depth = []
        self._reshaped_conf = []
        self._depth_images = []
        self._rgb_images = []
        self._confidences = []

        self._metadata = self._read_metadata()
        self.global_xyzs = []
        # self.global_pcds = None
        self._load_data()
        self._reshape_all_depth_and_conf()
        self.calculate_all_global_xyzs()

    def _read_metadata(self):
        with self._path.open("metadata", "r") as f:
            metadata_dict = json.load(f)

        # Now figure out the details from the metadata dict.
        self.rgb_width = metadata_dict["w"]
        self.rgb_height = metadata_dict["h"]
        self.fps = metadata_dict["fps"]
        self.camera_matrix = np.array(metadata_dict["K"]).reshape(3, 3).T

        self.image_size = (self.rgb_width, self.rgb_height)
        self.poses = np.array(metadata_dict["poses"])
        self.init_pose = np.array(metadata_dict["initPose"])
        self.total_images = len(self.poses)

        self._id_to_name = {i: x for (i, x) in enumerate(self._classes)}

    def load_image(self, filepath):
        with self._path.open(filepath, "r") as image_file:
            return np.asarray(Image.open(image_file))

    def load_depth(self, filepath):
        with self._path.open(filepath, "r") as depth_fh:
            raw_bytes = depth_fh.read()
            decompressed_bytes = liblzfse.decompress(raw_bytes)
            depth_img: np.ndarray = np.frombuffer(decompressed_bytes, dtype=np.float32)

        if depth_img.shape[0] == 960 * 720:
            depth_img = depth_img.reshape((960, 720))  # For a FaceID camera 3D Video
        else:
            depth_img = depth_img.reshape((256, 192))  # For a LiDAR 3D Video
        return depth_img

    def load_conf(self, filepath):
        with self._path.open(filepath, "r") as depth_fh:
            raw_bytes = depth_fh.read()
            decompressed_bytes = liblzfse.decompress(raw_bytes)
            depth_img = np.frombuffer(decompressed_bytes, dtype=np.uint8)
        if depth_img.shape[0] == 960 * 720:
            depth_img = depth_img.reshape((960, 720))  # For a FaceID camera 3D Video
        else:
            depth_img = depth_img.reshape((256, 192))  # For a LiDAR 3D Video
        return depth_img

    def _load_data(self):
        assert self.fps  # Make sure metadata is read correctly first.
        for i in tqdm.tqdm(range(0, self.total_images, self._subsample_freq), desc="Loading data"):
            # Read up the RGB and depth images first.
            rgb_filepath = f"rgbd/{i}.jpg"
            depth_filepath = f"rgbd/{i}.depth"
            conf_filepath = f"rgbd/{i}.conf"

            depth_img = self.load_depth(depth_filepath)
            confidence = self.load_conf(conf_filepath)
            rgb_img = self.load_image(rgb_filepath)

            # Now, convert depth image to real world XYZ pointcloud.
            self._depth_images.append(depth_img)
            self._rgb_images.append(rgb_img)
            self._confidences.append(confidence)

    def _reshape_all_depth_and_conf(self):
        #for index in tqdm.trange(len(self.poses), desc="Upscaling depth and conf"):
        for index in tqdm.trange(len(self._depth_images), desc="Upscaling depth and conf"): 
            depth_image = self._depth_images[index]
            # Upscale depth image.
            pil_img = Image.fromarray(depth_image)
            reshaped_img = pil_img.resize((self.rgb_width, self.rgb_height))
            reshaped_img = np.asarray(reshaped_img)
            self._reshaped_depth.append(reshaped_img)

            # Upscale confidence as well
            confidence = self._confidences[index]
            conf_img = Image.fromarray(confidence)
            reshaped_conf = conf_img.resize((self.rgb_width, self.rgb_height))
            reshaped_conf = np.asarray(reshaped_conf)
            self._reshaped_conf.append(reshaped_conf)

    def get_global_xyz(self, index, depth_scale=1000.0, only_confident=True):
        reshaped_img = np.copy(self._reshaped_depth[index])
        # If only confident, replace not confident points with nans
        if only_confident:
            reshaped_img[self._reshaped_conf[index] != 2] = np.nan

        depth_o3d = o3d.geometry.Image(
            np.ascontiguousarray(depth_scale * reshaped_img).astype(np.float32)
        )
        rgb_o3d = o3d.geometry.Image(
            np.ascontiguousarray(self._rgb_images[index]).astype(np.uint8)
        )

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, convert_rgb_to_intensity=False
        )

        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=int(self.rgb_width),
            height=int(self.rgb_height),
            fx=self.camera_matrix[0, 0],
            fy=self.camera_matrix[1, 1],
            cx=self.camera_matrix[0, 2],
            cy=self.camera_matrix[1, 2],
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, camera_intrinsics
        )
        # Flip the pcd
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        extrinsic_matrix = np.eye(4)
        #qx, qy, qz, qw, px, py, pz = self.poses[index]
        qx, qy, qz, qw, px, py, pz = self.poses[index * self._subsample_freq]
        extrinsic_matrix[:3, :3] = as_rotation_matrix(quaternion(qw, qx, qy, qz))
        extrinsic_matrix[:3, -1] = [px, py, pz]
        pcd.transform(extrinsic_matrix)

        # Now transform everything by init pose.
        init_matrix = np.eye(4)
        qx, qy, qz, qw, px, py, pz = self.init_pose
        init_matrix[:3, :3] = as_rotation_matrix(quaternion(qw, qx, qy, qz))
        init_matrix[:3, -1] = [px, py, pz]
        pcd.transform(init_matrix)
        # pcd.transform([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        if hasattr(self, 'n2r_matrix'):
            pcd.transform(self.n2r_matrix)

        return pcd

    def calculate_all_global_xyzs(self, only_confident=True):
        if len(self.global_xyzs):
            return self.global_xyzs, self.global_pcds
        for i in tqdm.trange(len(self._depth_images), desc="Calculating global XYZs"):
            global_xyz_pcd = self.get_global_xyz(i, only_confident=only_confident)
            global_xyz = np.asarray(global_xyz_pcd.points)
            self.global_xyzs.append(global_xyz)
        #     if global_xyz_pcd is not None:
        #         global_xyz_pcd.voxel_down_sample(voxel_size = 0.5)
        #         if self.global_pcds is None:
        #             self.global_pcds = global_xyz_pcd
        #         else:
        #             self.global_pcds += global_xyz_pcd
        #     # self.global_pcds.append(global_xyz_pcd)
        # return self.global_xyzs, self.global_pcds

    def __len__(self):
        #return len(self.poses)
        return len(self._depth_images)

    def __getitem__(self, idx):
        result = {
            "xyz_position": self.global_xyzs[idx],
            "rgb": self._rgb_images[idx],
            "depth": self._reshaped_depth[idx],
            "conf": self._reshaped_conf[idx],
        }
        return result
