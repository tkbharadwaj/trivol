'''
Tejas Bharadwaj, VMG
contains code for performing inference on TriVol model
'''


from train import TriVolModule
import os
import argparse
import numpy as np
import open3d as o3d

from pytorch_lightning.core import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import glob
from PIL import Image
from kornia import create_meshgrid
import random
import string
import torchvision
import imageio
import cv2
from datasets import *
from utils import *

from torch_scatter import scatter_mean

from models.unet3d import *
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt 

from datasets.common import get_rays, get_ray_directions_opencv


'''
Takes the path to the desired poses and the intrinsic camera matrix
and returns a tuple containing the ray origins and directions
file at pose_path should be a .txt file containing the numpy array for the pose matrix.

intrinsic_path should be the path leading to the .txt file containing the intrinsic camera matrices
'''
def sample_ray_for_inference(pose_path, intrinsic_path):
        c2w = torch.FloatTensor(np.loadtxt(pose_path))
        intrinsic_path = os.path.join(intrinsic_path, 'intrinsic_color.txt')
        intrinsic = np.loadtxt(intrinsic_path)
        fx = intrinsic[0, 0] 
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]

        # get all point xyz
        direction = get_ray_directions_opencv(args.width, args.height, fx, fy, cx, cy) # (H, W, 3)
        rays_o, rays_d = get_rays(direction, c2w) # both (H, W, 3)

        return rays_o, rays_d,

'''
loads the data from the path for inference

is_inference: describes whether or not the operation is validation (i.e. with ground truth labels)
or prediction on new data (without ground truth labels)
'''
def load_data_from_path(path, is_inference):

    test_data = None
    if is_inference != "True":
        test_data =  ScanNetDataset("val", scene_dir=path)
    else:
        batches = []
        #voxelize the input point cloud
        filename = None
        for s in os.listdir(path):
            if s.endswith(".ply"):
                filename = s
                break
        
        filename = os.path.join(path, filename)
        pcd_color = o3d.io.read_point_cloud(filename)
        points_raw = torch.FloatTensor(np.array(pcd_color.points, dtype=np.float32))        
        features = torch.FloatTensor(np.array(pcd_color.colors, dtype=np.float32))
        num_points = points_raw.shape[0]
        delta_scale = 0.1  
        aa = points_raw.min(0)[0][None]
        bb = points_raw.max(0)[0][None]  
        aa = aa - delta_scale * (bb - aa)
        bb = bb + delta_scale * (bb - aa)
        aabb = torch.cat([aa, bb], dim=0)
        C = 4
        resolution = 256
        points = (points_raw - aa) / (bb - aa + 1e-12)
        index_points = (points * (resolution - 1)).long()
        index_rgba = torch.cat([features, torch.ones_like(features[:, 0:1])], dim=1).transpose(0, 1) # [4, N]
        index = index_points[:, 2] + resolution * (index_points[:, 1] + resolution * index_points[:, 0])
        voxels = torch.zeros(C, resolution**3)
        scatter_mean(index_rgba, index, out=voxels) # B x C x reso^3
        voxels = voxels.reshape(C, resolution, resolution, resolution)

        intrinsic_path = os.path.join(path, "intrinsic")
        poses = os.listdir(os.path.join( path, "pose"))
        count = 0
        rgbs = torch.zeros(1, args.height, args.width, 3)
        for pose_path in poses:
            count = count+1
            pose_path = os.path.join(path, "pose", pose_path)
            rays_o_, rays_d_ = sample_ray_for_inference(pose_path, intrinsic_path)
            batches.append({
            "rays_o": rays_o_,
            "rays_d": rays_d_,
            "rgbs": rgbs,
            "aabb": aabb,
            "voxels": voxels,
            "paths": "",
            "filename": "inference/predictions/pred" + str(count) + ".jpg"
            })
            test_data = batches
    return DataLoader(test_data, batch_size = 1, collate_fn = trivol_collate_fn, num_workers = 2)



if __name__ == "__main__":  
    pa = argparse.ArgumentParser()
    pa.add_argument("--input_dir", type=str, help="input dir")
    pa.add_argument("--model_path", type=str, help="model path")
    pa.add_argument("--exp_name", type=str, help="exp name")
    pa.add_argument("--width", type=int, help="image width", default = 640)
    pa.add_argument("--height", type=int, help="image height", default = 512)
    pa.add_argument("--is_inference", type=str, default =  "False", help="True or False; inference or test")
    args = pa.parse_args()

    model = TriVolModule.load_from_checkpoint(args.model_path)
    model.exp_name = "inference"
    model.eval()
    data = None
    data = load_data_from_path(args.input_dir, args.is_inference)
    
    device = torch.device('cuda:0')
    tb_logger = pl_loggers.TensorBoardLogger("logs/%s" % args.exp_name)

    trainer = Trainer(max_epochs=200, 
                  devices=1,
                  accelerator="gpu", 
                      strategy="ddp", 
                      num_nodes=1,
                      num_sanity_val_steps=0,
                      logger=tb_logger,
                      )
    
    os.makedirs("logs/inference/predictions", exist_ok=True)
    if args.is_inference != "True":
        trainer.validate(model = model, dataloaders = data)
    else:
        trainer.predict(model = model, dataloaders = data
        )


