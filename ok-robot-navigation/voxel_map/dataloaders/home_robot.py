from torch.utils.data import Dataset
import pickle as pkl
from dataloaders.scannet_200_classes import CLASS_LABELS_200
import torch
import numpy as np


def get_inv_intrinsics(intrinsics):
    # return intrinsics.double().inverse().to(intrinsics)
    fx, fy, ppx, ppy = intrinsics[..., 0, 0], intrinsics[..., 1, 1], intrinsics[..., 0, 2], intrinsics[..., 1, 2]
    inv_intrinsics = torch.zeros_like(intrinsics)
    inv_intrinsics[..., 0, 0] = 1.0 / fx
    inv_intrinsics[..., 1, 1] = 1.0 / fy
    inv_intrinsics[..., 0, 2] = -ppx / fx
    inv_intrinsics[..., 1, 2] = -ppy / fy
    inv_intrinsics[..., 2, 2] = 1.0
    return inv_intrinsics


def apply_pose(xyz, pose):
    return (torch.einsum("na,nba->nb", xyz.to(pose), pose[..., :3, :3]) + pose[..., :3, 3]).to(xyz)


def apply_inv_intrinsics(xy, intrinsics):
    inv_intrinsics = get_inv_intrinsics(intrinsics)
    xyz = torch.cat((xy, torch.ones_like(xy[..., :1])), dim=-1)
    return torch.einsum("na,nba->nb", xyz.to(inv_intrinsics), inv_intrinsics)

def get_xyz_coordinates_from_xy(depth, xy, pose, intrinsics):
    xyz = apply_inv_intrinsics(xy, intrinsics)
    xyz = xyz * depth[:, None]
    xyz = apply_pose(xyz, pose)
    return xyz

def get_xyz_coordinates(depth, mask, pose, intrinsics):

    bsz, _, height, width = depth.shape
    flipped_mask = ~mask

    # Associates poses and intrinsics with XYZ coordinates.
    batch_inds = torch.arange(bsz, device=mask.device)
    batch_inds = batch_inds[:, None, None, None].expand_as(mask)[~mask]
    intrinsics = intrinsics[batch_inds]
    pose = pose[batch_inds]

    # Gets the depths for each coordinate.
    depth = depth[flipped_mask]

    # Gets the pixel grid.
    xs, ys = torch.meshgrid(
        torch.arange(width, device=depth.device),
        torch.arange(height, device=depth.device),
        indexing="xy",
    )
    xy = torch.stack([xs, ys], dim=-1)[None, :, :].repeat_interleave(bsz, dim=0)
    xy = xy[flipped_mask.squeeze(1)]

    return get_xyz_coordinates_from_xy(depth, xy, pose, intrinsics)

class HomeRobotDataset(Dataset):
    def __init__(
        self,
        path,
        custom_classes = CLASS_LABELS_200,
        subsample_freq = 1,
    ):
        with open(path, 'rb') as f:
            data = pkl.load(f)

        if custom_classes:
            self._classes = custom_classes
        else:
            self._classes = CLASS_LABELS_200
        self._classes = list(set(self._classes))
        print("The labels you use for OWL-ViT is ", str(self._classes))
        self._id_to_name = {i: x for (i, x) in enumerate(self._classes)}
        
        self.global_xyzs = []
        self._rgb_images = []
        self._reshaped_depth = []
        self._reshaped_conf = []
        self._subsample_freq = subsample_freq
        for i in range(len(data['rgb'])):
            rgb = data['rgb'][i].to(torch.uint8)
            depth = data['depth'][i]
            pose = data['camera_poses'][i]
            intr = data['camera_K'][i]
            mask = (depth < 0.3) | (depth > 3.0)
            conf = torch.empty(mask.shape, dtype = torch.int).fill_(2)
            conf[mask] = 1
            xyzs = get_xyz_coordinates(depth.unsqueeze(0).unsqueeze(0), mask.unsqueeze(0).unsqueeze(0), pose.unsqueeze(0), intr.unsqueeze(0))
            #print(xyzs.shape)
            #print(torch.where(conf == 2)[0].shape)
            self.global_xyzs.append(xyzs)
            self._rgb_images.append(rgb)
            self._reshaped_depth.append(depth)
            self._reshaped_conf.append(conf)
        self.image_size = (self._rgb_images[0].shape[0], self._rgb_images[0].shape[1])

    def __len__(self):
        return len(self._rgb_images)

    def __getitem__(self, idx):
        result = {
            "xyz_position": self.global_xyzs[idx],
            "rgb": self._rgb_images[idx],
            "depth": self._reshaped_depth[idx],
            "conf": self._reshaped_conf[idx],
        }
        return result
