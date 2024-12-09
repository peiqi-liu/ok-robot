# from a_star.data_util import get_posed_rgbd_dataset
from a_star.dataset_class import R3DDataset, HomeRobotDataset
import torch
from PIL import Image
from matplotlib import pyplot as plt
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as V
from voxel import VoxelizedPointcloud
from voxel_map_localizer import VoxelMapLocalizer
import sys
sys.path.append('voxel_map')
from dataloaders.scannet_200_classes import CLASS_LABELS_200

# from segment_anything import sam_model_registry, SamPredictor
from mobile_sam import sam_model_registry, SamPredictor
from ultralytics import YOLO, SAM, YOLOWorld
from transformers import AutoProcessor, OwlViTForObjectDetection
import clip
from transformers import AutoProcessor, AutoModel
from torchvision import transforms

import os
import wget
import time

import clip
import numpy as np
import tqdm
import torch.nn.functional as F
import pandas as pd

import cv2

from home_robot.perception.encoders import ClipEncoder
from home_robot.mapping.instance import InstanceMemory, Instance, InstanceView

# def get_inv_intrinsics(intrinsics):
#     # return intrinsics.double().inverse().to(intrinsics)
#     fx, fy, ppx, ppy = intrinsics[..., 0, 0], intrinsics[..., 1, 1], intrinsics[..., 0, 2], intrinsics[..., 1, 2]
#     inv_intrinsics = torch.zeros_like(intrinsics)
#     inv_intrinsics[..., 0, 0] = 1.0 / fx
#     inv_intrinsics[..., 1, 1] = 1.0 / fy
#     inv_intrinsics[..., 0, 2] = -ppx / fx
#     inv_intrinsics[..., 1, 2] = -ppy / fy
#     inv_intrinsics[..., 2, 2] = 1.0
#     return inv_intrinsics


# def apply_pose(xyz, pose):
#     return (torch.einsum("na,nba->nb", xyz.to(pose), pose[..., :3, :3]) + pose[..., :3, 3]).to(xyz)


# def apply_inv_intrinsics(xy, intrinsics):
#     inv_intrinsics = get_inv_intrinsics(intrinsics)
#     xyz = torch.cat((xy, torch.ones_like(xy[..., :1])), dim=-1)
#     return torch.einsum("na,nba->nb", xyz.to(inv_intrinsics), inv_intrinsics)

# def get_xyz_coordinates_from_xy(depth, xy, pose, intrinsics):
#     xyz = apply_inv_intrinsics(xy, intrinsics)
#     xyz = xyz * depth[:, None]
#     xyz = apply_pose(xyz, pose)
#     return xyz

# def get_xyz_coordinates(depth, mask, pose, intrinsics):

#     bsz, _, height, width = depth.shape
#     flipped_mask = ~mask

#     # Associates poses and intrinsics with XYZ coordinates.
#     batch_inds = torch.arange(bsz, device=mask.device)
#     batch_inds = batch_inds[:, None, None, None].expand_as(mask)[~mask]
#     intrinsics = intrinsics[batch_inds]
#     pose = pose[batch_inds]

#     # Gets the depths for each coordinate.
#     depth = depth[flipped_mask]

#     # Gets the pixel grid.
#     xs, ys = torch.meshgrid(
#         torch.arange(width, device=depth.device),
#         torch.arange(height, device=depth.device),
#         indexing="xy",
#     )
#     xy = torch.stack([xs, ys], dim=-1)[None, :, :].repeat_interleave(bsz, dim=0)
#     xy = xy[flipped_mask.squeeze(1)]

#     return get_xyz_coordinates_from_xy(depth, xy, pose, intrinsics)

def get_xyz_coordinates(depth, pose, intrinsic):

    _, height, width = depth.shape

    # Gets the pixel grid.
    xs, ys = torch.meshgrid(
        torch.arange(width, device=depth.device),
        torch.arange(height, device=depth.device),
        indexing="xy",
    )

    x = (xs - intrinsic[0, 2]) / intrinsic[0, 0]
    y = (ys - intrinsic[1, 2]) / intrinsic[1, 1]

    # Depth array should be the same shape as x and y
    z = depth[0]

    # Prepare camera coordinates
    camera_coords = torch.stack((x * z, y * z, z, torch.ones_like(z)), axis=-1)

    # Prepare pose matrix for broadcasting
    # Transform to world coordinates using the pose matrix
    world_coords = camera_coords @ pose.T

    # Return world coordinates (excluding the homogeneous coordinate)
    return world_coords[..., :3]

class SemanticMemoryEval:
    def __init__(self,  
        detection = 'yolo', 
        vlencoder = 'siglip',
        sam_config = 'vit_b',
        device = 'cuda',
        min_depth = 0.25,
        max_depth = 2.0,
        pcd_path: str = None,
    ):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.obs_count = 0
        self.detection = detection
        self.vlencoder = vlencoder
        # If cuda is not available, then device will be forced to be cpu
        if not torch.cuda.is_available():
            device = 'cpu'
        self.device = device
        self.pcd_path = pcd_path
        self.create_vision_model()

    def eval_texts(self, texts):
        return self.voxel_map_localizer.find_alignment_for_A(texts)

    def create_vision_model(self):
        if self.vlencoder == 'clip':
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/16", device=self.device)
            self.clip_model.eval()
        else:
            self.clip_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(self.device)
            self.clip_preprocess = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
            self.clip_model.eval()
        if self.detection != 'maskclip':
            if not os.path.exists('sam_vit_b_01ec64.pth'):
                wget.download('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth', out = 'sam_vit_b_01ec64.pth')
            sam = sam_model_registry['vit_b'](checkpoint='sam_vit_b_01ec64.pth')
            # sam = sam_model_registry['vit_t'](checkpoint='/data/peiqi/weight/mobile_sam.pt')
            self.mask_predictor = SamPredictor(sam)
            self.mask_predictor.model = self.mask_predictor.model.eval().to(self.device)
            if self.detection == 'owl':
                self.texts = [['a photo of ' + text for text in CLASS_LABELS_200]]
                self.owl_processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
                self.owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").eval().to(self.device)
            else:
                self.yolo_model = YOLOWorld('yolov8s-worldv2.pt')
                self.yolo_model.set_classes(list(CLASS_LABELS_200))
        if self.vlencoder == 'clip':
            self.voxel_map_localizer = VoxelMapLocalizer(device = self.device, siglip = False)
        else:
            self.voxel_map_localizer = VoxelMapLocalizer(device = 'cpu', siglip = True)
        if self.pcd_path is not None:
            print('Loading old semantic memory')
            self.voxel_map_localizer.voxel_pcd = torch.load(self.pcd_path)
            print('Finish loading old semantic memory')

    def forward_one_block(self, resblocks, x):
        q, k, v = None, None, None
        y = resblocks.ln_1(x)
        y = F.linear(y, resblocks.attn.in_proj_weight, resblocks.attn.in_proj_bias)
        N, L, C = y.shape
        y = y.view(N, L, 3, C//3).permute(2, 0, 1, 3).reshape(3*N, L, C//3)
        y = F.linear(y, resblocks.attn.out_proj.weight, resblocks.attn.out_proj.bias)
        q, k, v = y.tensor_split(3, dim=0)
        v += x
        v = v + resblocks.mlp(resblocks.ln_2(v))

        return v

    def extract_mask_clip_features(self, x, image_shape):
        with torch.no_grad():
            x = self.clip_model.visual.conv1(x)
            N, L, H, W = x.shape
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            x = torch.cat([self.clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
            x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
            x = self.clip_model.visual.ln_pre(x)
            x = x.permute(1, 0, 2)
            for idx in range(self.clip_model.visual.transformer.layers):
                if idx == self.clip_model.visual.transformer.layers - 1:
                    break
                x = self.clip_model.visual.transformer.resblocks[idx](x)
            x = self.forward_one_block(self.clip_model.visual.transformer.resblocks[-1], x)
            x = x[1:]
            x = x.permute(1, 0, 2)
            x = self.clip_model.visual.ln_post(x)
            x = x @ self.clip_model.visual.proj
            feat = x.reshape(N, H, W, -1).permute(0, 3, 1, 2)
        feat = F.interpolate(feat, image_shape, mode = 'bilinear', align_corners = True)
        feat = F.normalize(feat, dim = 1)
        return feat.permute(0, 2, 3, 1)
    
    def run_mask_clip(self, rgb, mask, world_xyz):
        # This code verify whether image is BGR, if it is RGB, then you should transform images into BGR

        # cv2.imwrite('debug.jpg', np.asarray(transforms.ToPILImage()(rgb), dtype = np.uint8))
        with torch.no_grad():
            if self.device == 'cpu':
                input = self.clip_preprocess(transforms.ToPILImage()(rgb)).unsqueeze(0).to(self.device)
            else:
                input = self.clip_preprocess(transforms.ToPILImage()(rgb)).unsqueeze(0).to(self.device).half()
            features = self.extract_mask_clip_features(input, rgb.shape[-2:])[0].cpu()

        # Let MaskClip do segmentation, the results should be reasonable but do not expect it to be accurate

        # text = clip.tokenize(["a keyboard", "a human"]).to(self.device)
        # image_vis = np.array(rgb.permute(1, 2, 0))
        # cv2.imwrite('clean_' + str(self.obs_count) + '.jpg', cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))
        # with torch.no_grad():
        #     text_features = self.clip_model.encode_text(text)
        #     text_features = F.normalize(text_features, dim = -1)
        #     output = torch.argmax(features.float() @ text_features.T.float().cpu(), dim = -1)
        # segmentation_color_map = np.zeros(image_vis.shape, dtype=np.uint8)
        # segmentation_color_map[np.asarray(output) == 0] = [0, 255, 0]
        # image_vis = cv2.addWeighted(image_vis, 0.7, segmentation_color_map, 0.3, 0)
        # cv2.imwrite("seg" + str(self.obs_count) + ".jpg", cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))
            
        valid_xyz = world_xyz[~mask]
        features = features[~mask]
        valid_rgb = rgb.permute(1, 2, 0)[~mask]
        if len(valid_xyz) != 0:
            self.add_to_voxel_pcd(valid_xyz, features, valid_rgb)

    def run_yolo_sam_siglip(self, rgb, mask, world_xyz):
        with torch.no_grad():
            results = self.yolo_model.predict(rgb.permute(1,2,0)[:, :, [2, 1, 0]].numpy(), conf=0.15, verbose=False)
            xyxy_tensor = results[0].boxes.xyxy
            if len(xyxy_tensor) == 0:
                return

            self.mask_predictor.set_image(rgb.permute(1,2,0).numpy())
            bounding_boxes = torch.stack(sorted(xyxy_tensor, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse = True), dim = 0)
            transformed_boxes = self.mask_predictor.transform.apply_boxes_torch(bounding_boxes.detach().to(self.device), rgb.shape[-2:])
            masks, _, _= self.mask_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False
            )
            masks = masks[:, 0, :, :].cpu()
            
            # Debug code, visualize all bounding boxes and segmentation masks

            image_vis = np.asarray(rgb.permute(1, 2, 0))
            segmentation_color_map = np.zeros(image_vis.shape, dtype=np.uint8)
            for idx, box in enumerate(bounding_boxes):
                tl_x, tl_y, br_x, br_y = box
                tl_x, tl_y, br_x, br_y = tl_x.item(), tl_y.item(), br_x.item(), br_y.item()
                cv2.rectangle(image_vis, (int(tl_x), int(tl_y)), (int(br_x), int(br_y)), (255, 0, 0), 2)
            image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR) 
            for vis_mask in masks:
                segmentation_color_map[vis_mask.detach().cpu().numpy()] = [0, 255, 0]
            image_vis = cv2.addWeighted(image_vis, 0.7, segmentation_color_map, 0.3, 0)
            cv2.imwrite("debug/seg" + str(self.obs_count) + ".jpg", image_vis)
    
            crops = []
            if self.vlencoder == 'clip':
                for box in bounding_boxes:
                    tl_x, tl_y, br_x, br_y = box
                    crops.append(self.clip_preprocess(transforms.ToPILImage()(rgb[:, max(int(tl_y), 0): min(int(br_y), rgb.shape[1]), max(int(tl_x), 0): min(int(br_x), rgb.shape[2])])))
                features = self.clip_model.encode_image(torch.stack(crops, dim = 0).to(self.device))
            else:
                for box in bounding_boxes:
                    tl_x, tl_y, br_x, br_y = box
                    crops.append(rgb[:, max(int(tl_y), 0): min(int(br_y), rgb.shape[1]), max(int(tl_x), 0): min(int(br_x), rgb.shape[2])])
                inputs = self.clip_preprocess(images = crops, padding="max_length", return_tensors="pt").to(self.device)
                features = self.clip_model.get_image_features(**inputs)
            features = F.normalize(features, dim = -1).cpu()

        for idx, (sam_mask, feature) in enumerate(zip(masks.cpu(), features.cpu())):
            valid_mask = torch.logical_and(~mask, sam_mask)
            valid_xyz = world_xyz[valid_mask]
            if valid_xyz.shape[0] == 0:
                continue
            feature = feature.repeat(valid_xyz.shape[0], 1)
            valid_rgb = rgb.permute(1, 2, 0)[valid_mask]
            self.add_to_voxel_pcd(valid_xyz, feature, valid_rgb)
    
    def run_owl_sam_clip(self, rgb, mask, world_xyz):
        with torch.no_grad():
            inputs = self.owl_processor(text=self.texts, images=rgb, return_tensors="pt")
            for input in inputs:
                inputs[input] = inputs[input].to(self.device)
            outputs = self.owl_model(**inputs)
            target_sizes = torch.Tensor([rgb.size()[-2:]]).to(self.device)
            results = self.owl_processor.post_process_object_detection(outputs=outputs, threshold=0.15, target_sizes=target_sizes)
            if len(results[0]['boxes']) == 0:
                return

            self.mask_predictor.set_image(rgb.permute(1,2,0).numpy())
            bounding_boxes = torch.stack(sorted(results[0]['boxes'], key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse = True), dim = 0)
            transformed_boxes = self.mask_predictor.transform.apply_boxes_torch(bounding_boxes.detach().to(self.device), rgb.shape[-2:])
            masks, _, _= self.mask_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False
            )
            masks = masks[:, 0, :, :].cpu()
            
            # Debug code, visualize all bounding boxes and segmentation masks

            image_vis = np.asarray(rgb.permute(1, 2, 0))
            segmentation_color_map = np.zeros(image_vis.shape, dtype=np.uint8)
            for idx, box in enumerate(bounding_boxes):
                tl_x, tl_y, br_x, br_y = box
                tl_x, tl_y, br_x, br_y = tl_x.item(), tl_y.item(), br_x.item(), br_y.item()
                cv2.rectangle(image_vis, (int(tl_x), int(tl_y)), (int(br_x), int(br_y)), (255, 0, 0), 2)
            image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR) 
            for vis_mask in masks:
                segmentation_color_map[vis_mask.detach().cpu().numpy()] = [0, 255, 0]
            image_vis = cv2.addWeighted(image_vis, 0.7, segmentation_color_map, 0.3, 0)
            cv2.imwrite("debug/seg" + str(self.obs_count) + ".jpg", image_vis)
    
            crops = []
            if self.vlencoder == 'clip':
                for box in bounding_boxes:
                    tl_x, tl_y, br_x, br_y = box
                    crops.append(self.clip_preprocess(transforms.ToPILImage()(rgb[:, max(int(tl_y), 0): min(int(br_y), rgb.shape[1]), max(int(tl_x), 0): min(int(br_x), rgb.shape[2])])))
                features = self.clip_model.encode_image(torch.stack(crops, dim = 0).to(self.device))
            else:
                for box in bounding_boxes:
                    tl_x, tl_y, br_x, br_y = box
                    crops.append(rgb[:, max(int(tl_y), 0): min(int(br_y), rgb.shape[1]), max(int(tl_x), 0): min(int(br_x), rgb.shape[2])])
                inputs = self.clip_preprocess(images = crops, padding="max_length", return_tensors="pt").to(self.device)
                features = self.clip_model.get_image_features(**inputs)
            features = F.normalize(features, dim = -1).cpu()
            
            # Debug code, let the clip select bounding boxes most aligned with a text query, used to check whether clip embeddings for
            # bounding boxes are reasonable

            # text = clip.tokenize(["a coco cola"]).to(self.device)

            # with torch.no_grad():
            #     text_features = self.clip_model.encode_text(text)
            #     text_features = F.normalize(text_features, dim = -1)
            #     i = torch.argmax(features.float() @ text_features.T.float().cpu()).item()
            # image_vis = np.array(rgb.permute(1, 2, 0))
            # segmentation_color_map = np.zeros(image_vis.shape, dtype=np.uint8)
            # cv2.imwrite('clean_' + str(self.obs_count) + '.jpg', cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))
            # tl_x, tl_y, br_x, br_y = bounding_boxes[i]
            # tl_x, tl_y, br_x, br_y = tl_x.item(), tl_y.item(), br_x.item(), br_y.item()
            # cv2.rectangle(image_vis, (int(tl_x), int(tl_y)), (int(br_x), int(br_y)), (255, 0, 0), 2)
            # image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR) 
            # for vis_mask in masks:
            #     segmentation_color_map[vis_mask.detach().cpu().numpy()] = [0, 255, 0]
            # image_vis = cv2.addWeighted(image_vis, 0.7, segmentation_color_map, 0.3, 0)
            # cv2.imwrite("seg" + str(self.obs_count) + ".jpg", image_vis)


        for idx, (sam_mask, feature) in enumerate(zip(masks.cpu(), features.cpu())):
            valid_mask = torch.logical_and(~mask, sam_mask)
            valid_xyz = world_xyz[valid_mask]
            if valid_xyz.shape[0] == 0:
                continue
            feature = feature.repeat(valid_xyz.shape[0], 1)
            valid_rgb = rgb.permute(1, 2, 0)[valid_mask]
            self.add_to_voxel_pcd(valid_xyz, feature, valid_rgb)
    
    def add_to_voxel_pcd(self, valid_xyz, feature, valid_rgb, weights = None, threshold = 0.85):
        selected_indices = torch.randperm(len(valid_xyz))[:int((1 - threshold) * len(valid_xyz))]
        if len(selected_indices) == 0:
            return
        if valid_xyz is not None:
            valid_xyz = valid_xyz[selected_indices]
        if feature is not None:
            feature = feature[selected_indices]
        if valid_rgb is not None:
            valid_rgb = valid_rgb[selected_indices]
        if weights is not None:
            weights = weights[selected_indices]
        self.voxel_map_localizer.add(points = valid_xyz, 
                                features = feature,
                                rgb = valid_rgb,
                                weights = weights)


    def process_rgbd_images(self, rgb, mask, world_xyz):
        self.obs_count += 1
        rgb = (rgb * 255).to(torch.uint8)

        if self.detection == 'owl':
            self.run_owl_sam_clip(rgb, mask, world_xyz)
        elif self.detection == 'maskclip':
            self.run_mask_clip(rgb, mask, world_xyz)
        else:
            self.run_yolo_sam_siglip(rgb, mask, world_xyz)


def eval_semantic(dataset, csv_file):
    semanticEval = SemanticMemoryEval(detection = 'yolo')
    for i in tqdm.tqdm(dataset):
        rgb, depth, mask, intrinsics, pose = i
        _, w, h = rgb.shape
        # depth = depth.unsqueeze(0)
        mask = mask.unsqueeze(0)
        # point_mask = torch.empty(mask.shape, dtype=mask.dtype).fill_(False)
        # intrinsics = intrinsics.unsqueeze(0)
        # pose = pose.unsqueeze(0)
        # batch_xyz = get_xyz_coordinates(depth.to(pose), point_mask, pose, intrinsics).reshape(w, h, -1)
        batch_xyz = get_xyz_coordinates(depth.to(pose), pose, intrinsics)
        semanticEval.process_rgbd_images(rgb, mask[0, 0], batch_xyz)
    
    annotations = pd.read_csv(csv_file)
    labels = list(annotations['query'].values)
    xyzs = torch.stack([torch.from_numpy(annotations['x'].values), torch.from_numpy(annotations['y'].values), torch.from_numpy(annotations['z'].values)], dim = -1)
    afford = torch.from_numpy(annotations['affordance'].values)
    pred_xyzs = semanticEval.eval_texts(labels)
    return pred_xyzs, xyzs, labels, afford, len(torch.where(torch.linalg.norm((pred_xyzs - xyzs), dim = -1) <= afford)[0])

def run_owl_sam(rgb, mask, owl_model, owl_processor, mask_predictor, texts, device):
    with torch.no_grad():
        inputs = owl_processor(text=texts, images=rgb, return_tensors="pt")
        for input in inputs:
            inputs[input] = inputs[input].to(device)
        outputs = owl_model(**inputs)
        target_sizes = torch.Tensor([rgb.size()[-2:]]).to(device)
        results = owl_processor.post_process_object_detection(outputs=outputs, threshold=0.2, target_sizes=target_sizes)
        if len(results[0]['boxes']) == 0:
            return None

        mask_predictor.set_image(rgb.permute(1,2,0).numpy())
        bounding_boxes = torch.stack(sorted(results[0]['boxes'], key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse = True), dim = 0)
        transformed_boxes = mask_predictor.transform.apply_boxes_torch(bounding_boxes.detach().to(device), rgb.shape[-2:])
        masks, _, _= mask_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False
        )
        masks = masks[:, 0, :, :].cpu()

        # image_vis = np.array(rgb.permute(1, 2, 0))
        # segmentation_color_map = np.zeros(image_vis.shape, dtype=np.uint8)
        # for idx, box in enumerate(bounding_boxes):
        #     tl_x, tl_y, br_x, br_y = box
        #     tl_x, tl_y, br_x, br_y = tl_x.item(), tl_y.item(), br_x.item(), br_y.item()
        #     cv2.rectangle(image_vis, (int(tl_x), int(tl_y)), (int(br_x), int(br_y)), (255, 0, 0), 2)
        # image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR) 
        # for vis_mask in masks:
        #     segmentation_color_map[vis_mask.detach().cpu().numpy()] = [0, 255, 0]
        # image_vis = cv2.addWeighted(image_vis, 0.7, segmentation_color_map, 0.3, 0)
        # cv2.imwrite("seg.jpg", image_vis)
    
        instance_mask = torch.zeros((rgb.shape[-2], rgb.shape[-1]), dtype=torch.uint8)
        for idx, sam_mask in enumerate(masks.cpu()):
            valid_mask = torch.logical_and(mask, sam_mask)
            instance_mask[valid_mask] = idx + 1
        return instance_mask

def eval_instance(dataset, csv_file):
    instance_memory = InstanceMemory(1, 1, min_instance_height = -10.0, max_instance_height = 10.0)
    encoder = ClipEncoder()
    device = 'cuda'
    owl_processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
    owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").eval().to(device)
    if not os.path.exists('sam_vit_b_01ec64.pth'):
        wget.download('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth', out = 'sam_vit_b_01ec64.pth')
    sam = sam_model_registry['vit_b'](checkpoint='sam_vit_b_01ec64.pth')
    mask_predictor = SamPredictor(sam)
    mask_predictor.model = mask_predictor.model.eval().to(device)
    texts = [['a photo of ' + text for text in CLASS_LABELS_200]]
    for i in tqdm.tqdm(dataset):
        rgb, depth, mask, intrinsics, pose = i
        _, w, h = rgb.shape
        depth = depth.unsqueeze(0)
        mask = mask.unsqueeze(0)
        point_mask = torch.empty(mask.shape, dtype=mask.dtype).fill_(False)
        intrinsics = intrinsics.unsqueeze(0)
        pose = pose.unsqueeze(0)
        batch_xyz = get_xyz_coordinates(depth.to(pose), point_mask, pose, intrinsics).reshape(w, h, -1).to(torch.float32)
        instance_mask = run_owl_sam((rgb * 255).to(torch.uint8), ~mask[0, 0], owl_model, owl_processor, mask_predictor, texts, device)
        if instance_mask is not None:
            instance_memory.process_instances_for_env(0, instance_mask, batch_xyz, rgb * 255, encoder = encoder)
            instance_memory.associate_instances_to_memory()
    
    annotations = pd.read_csv(csv_file)
    labels = list(annotations['query'].values)
    xyzs = torch.stack([torch.from_numpy(annotations['x'].values), torch.from_numpy(annotations['y'].values), torch.from_numpy(annotations['z'].values)], dim = -1)
    afford = torch.from_numpy(annotations['affordance'].values)
    torch.save(instance_memory, 'instance.pt')
    ins_list = instance_memory.instances
    embs = []
    for i in ins_list[0]:
        embs.append(ins_list[0][i].get_image_embedding(aggregation_method="mean"))
    embs = torch.stack(embs, dim = 0)

    clip_text_tokens = encoder.model.encode_text(clip.tokenize(labels).to(device)).float()
    clip_text_tokens = F.normalize(clip_text_tokens, p=2, dim=-1).detach().cpu().float()
    features = F.normalize(embs, p=2, dim=-1).detach().cpu().float()
    point_alignments = clip_text_tokens.detach().cpu() @ features.T
    res = point_alignments.topk(k = 1, dim = -1).indices
    pred_xyzs = []
    for i in res:
        pred_xyzs.append(ins_list[0][i[0].item()].bounds.mean(dim = -1))
    pred_xyzs = torch.stack(pred_xyzs, dim = 0)
    return pred_xyzs, xyzs, labels, afford, len(torch.where(torch.linalg.norm((pred_xyzs - xyzs), dim = -1) <= afford)[0])

torch.manual_seed(1)
total_semantic_success = 0
total_queries = 0
for dataset_file, csv_file in zip(
    ['sofaroom.pkl', 'LeoBedroom.r3d', 'VenkyRoom.r3d', 'robot.pkl', 'LeoFriendKitchen.r3d', 'VenthyaKitchen.r3d'], 
    ['sofaroom.csv', 'leoroom.csv', 'venkyroom.csv', 'nyukitchen.csv', 'chuanyangkitchen.csv', 'venthyakitchen.csv']):
    print('-' * 20, 'Evaluating results of ', dataset_file, '-' * 20, '\n')
    print('-' * 10, 'Semantic Memory', '-' * 10)
    if dataset_file[-3:] == 'r3d':
        # dataset = R3DDataset('/data/peiqi/r3d/' + dataset_file, subsample_freq = 10, shape = (640, 480))
        dataset = R3DDataset('/data/peiqi/r3d/' + dataset_file, subsample_freq = 10)
    else:
        dataset = HomeRobotDataset(dataset_file)
    pred_xyzs, xyzs, labels, affords, success_num = eval_semantic(dataset, csv_file)
    total_semantic_success += success_num
    total_queries += len(labels)
    print('Success queries in this scene:', success_num)
    print('Total queries in this scene:', len(affords))
    print('Success rate of this scene is', success_num / len(affords) * 100, '%')
    print(pred_xyzs, xyzs, labels)
    # for pred_xyz, xyz, label, afford in zip(pred_xyzs, xyzs, labels, affords):
    #     print(label, 'Error: ', torch.norm(pred_xyz - xyz).item(), 'CORRECT' if torch.norm(pred_xyz - xyz).item() < afford else 'WRONG')
print('Total instance success num:', total_semantic_success)
print('Total text queries:', total_queries)
print('Total Success rate of semantic memory:', total_semantic_success * 100 / total_queries, '%')


# total_instance_success = 0
# total_queries = 0
# for dataset_file, csv_file in zip(
#     ['LeoBedroom.r3d', 'VenkyRoom.r3d', 'robot.pkl', 'VenthyaKitchen.r3d'], 
#     ['leoroom.csv', 'venkyroom.csv', 'nyukitchen.csv', 'venthyakitchen.csv']):
#     print('-' * 20, 'Evaluating results of ', dataset_file, '-' * 20, '\n')
#     print('-' * 10, 'Instance Memory', '-' * 10)
#     if dataset_file[-3:] == 'r3d':
#         # dataset = R3DDataset('/data/peiqi/r3d/' + dataset_file, subsample_freq = 10, shape = (640, 480))
#         dataset = R3DDataset('/data/peiqi/r3d/' + dataset_file, subsample_freq = 10)
#     else:
#         dataset = HomeRobotDataset(dataset_file)
#     pred_xyzs, xyzs, labels, affords, success_num = eval_instance(dataset, csv_file)
#     total_instance_success += success_num
#     total_queries += len(labels)
#     print('Success queries in this scene:', success_num)
#     print('Total queries in this scene:', len(affords))
#     print('Success rate of this scene:', success_num / len(affords) * 100, '%')
#     print(pred_xyzs, xyzs, labels)
#     # for pred_xyz, xyz, label, afford in zip(pred_xyzs, xyzs, labels, affords):
#     #     print(label, 'Error: ', torch.norm(pred_xyz - xyz).item(), 'CORRECT' if torch.norm(pred_xyz - xyz).item() < afford else 'WRONG')
# print('Total instance success num:', total_instance_success)
# print('Total text queries:', total_queries)
# print('Total Success rate of instance memory:', total_instance_success * 100 / total_queries, '%')
