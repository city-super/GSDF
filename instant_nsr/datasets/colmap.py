import os
import math
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import instant_nsr.datasets
from instant_nsr.datasets.colmap_utils import \
    read_cameras_binary, read_images_binary, read_cameras_text, read_images_text, read_points3d_binary, read_points3d_text
from instant_nsr.models.ray_utils import get_ray_directions

from instant_nsr.utils.misc import config_to_primitive, get_rank
from instant_nsr.datasets import register



def get_center(pts):
    center = pts.mean(0)
    dis = (pts - center[None,:]).norm(p=2, dim=-1)
    mean, std = dis.mean(), dis.std()
    q25, q75 = torch.quantile(dis, 0.25), torch.quantile(dis, 0.75)
    valid = (dis > mean - 1.5 * std) & (dis < mean + 1.5 * std) & (dis > mean - (q75 - q25) * 1.5) & (dis < mean + (q75 - q25) * 1.5)
    center = pts[valid].mean(0)
    return center


def simple_normalize_poses(poses, pts, special_c2w):
    
    # first estimation scene center as the average of all camera positions
    # later we'll use the center of all points bounded by the cameras as the final scene center
    center = poses[...,3].mean(0)

    # translation and scaling
    poses_min, poses_max = poses[...,3].min(0)[0], poses[...,3].max(0)[0]
    pts_fg = pts[(poses_min[0] < pts[:,0]) & (pts[:,0] < poses_max[0]) & (poses_min[1] < pts[:,1]) & (pts[:,1] < poses_max[1])]
    center = get_center(pts_fg)
    tc = center.reshape(3, 1)
    t = -tc
    
    

    poses_homo = torch.cat([poses, torch.as_tensor([[[0.,0.,0.,1.]]]).expand(poses.shape[0], -1, -1)], dim=1)
    special_c2w_homo = torch.cat([special_c2w, torch.as_tensor([[[0.,0.,0.,1.]]]).expand(special_c2w.shape[0], -1, -1)], dim=1)

    inv_trans = torch.cat([torch.cat([torch.eye(3), t], dim=1), torch.as_tensor([[0.,0.,0.,1.]])], dim=0)
    poses_norm = (inv_trans @ poses_homo)[:,:3]
    special_c2w_homo = (inv_trans @ special_c2w_homo)[:,:3]

    

    bb=torch.cat([pts, torch.ones_like(pts[:,0:1])], dim=-1)[...,None].cuda()

    pts = (inv_trans.cuda() @ bb)[:,:3,0]

    pts=pts.cpu()

    return poses_norm, pts, special_c2w_homo

def normalize_poses_with_given(poses, pts, special_c2w,given_center, given_scale):
    
    tc = torch.tensor(given_center).reshape(3, 1)

    t = -tc

    poses_homo = torch.cat([poses, torch.as_tensor([[[0.,0.,0.,1.]]]).expand(poses.shape[0], -1, -1)], dim=1)
    special_c2w_homo = torch.cat([special_c2w, torch.as_tensor([[[0.,0.,0.,1.]]]).expand(special_c2w.shape[0], -1, -1)], dim=1)

    inv_trans = torch.cat([torch.cat([torch.eye(3), t], dim=1), torch.as_tensor([[0.,0.,0.,1.]])], dim=0)
    poses_norm = (inv_trans @ poses_homo)[:,:3]
    special_c2w_homo = (inv_trans @ special_c2w_homo)[:,:3]



    poses_norm[...,3] /= given_scale
    special_c2w_homo[...,3] /= given_scale

    bb=torch.cat([pts, torch.ones_like(pts[:,0:1])], dim=-1)[...,None].cuda()

    pts = (inv_trans.cuda() @ bb)[:,:3,0]

    pts=pts.cpu()

    
    pts = pts / given_scale


    return poses_norm, pts, special_c2w_homo




class ColmapDatasetBase():
    # the data only has to be processed once
    initialized = False
    properties = {}


    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()
        if not ColmapDatasetBase.initialized:
            load_path = os.path.join(self.config.root_dir, 'sparse/0/cameras.bin')
            if os.path.exists(load_path):
                camdata = read_cameras_binary(load_path)
            else:
                load_path = os.path.join(self.config.root_dir, 'colmap/cameras.txt')
                camdata = read_cameras_text(load_path)

            H = int(camdata[1].height)
            W = int(camdata[1].width)

            if 'img_wh' in self.config:
                w, h = self.config.img_wh
                assert round(W / w * h) == H
            elif 'img_downscale' in self.config:
                w, h = round(W / self.config.img_downscale), round(H / self.config.img_downscale)
            else:
                raise KeyError("Either img_wh or img_downscale should be specified.")

            img_wh = (w, h)
            factor = w / W
            
            if camdata[1].model == 'SIMPLE_RADIAL' or camdata[1].model == 'SIMPLE_PINHOLE':
                fx = fy = camdata[1].params[0] * factor
                cx = camdata[1].params[1] * factor
                cy = camdata[1].params[2] * factor
            elif camdata[1].model in ['PINHOLE', 'OPENCV']:
                fx = camdata[1].params[0] * factor
                fy = camdata[1].params[1] * factor
                cx = camdata[1].params[2] * factor
                cy = camdata[1].params[3] * factor
            else:
                raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
            
            directions = get_ray_directions(w, h, fx, fy, cx, cy).to(self.rank)

            im_path = os.path.join(self.config.root_dir, 'sparse/0/images.bin')
            if os.path.exists(im_path):
                imdata = read_images_binary(im_path)
            else:
                im_path = os.path.join(self.config.root_dir, 'colmap/images.txt')
                imdata = read_images_text(im_path)


            mask_dir = os.path.join(self.config.root_dir, 'masks')
            has_mask = os.path.exists(mask_dir) # TODO: support partial masks
            apply_mask = has_mask and self.config.apply_mask


            sorted_keys = sorted(imdata.keys(), reverse=False)
            
            all_c2w, all_images, all_fg_masks,all_names,special_c2w = [], [], [],[],[]
           
            if_corrected = []
           

            for i, key in enumerate(sorted_keys):
                d = imdata[key]
                R = d.qvec2rotmat()
                t = d.tvec.reshape(3, 1)
                c2w = torch.from_numpy(np.concatenate([R.T, -R.T@t], axis=1)).float()
              
                
                all_c2w.append(c2w)
                img_path = os.path.join(self.config.root_dir, 'images', d.name)
                if not os.path.exists(img_path):
                    continue
                if self.split not in ['train']:
                    
                    if i % 8 != 0: continue
                    special_c2w.append(c2w)
                    
                    img = Image.open(img_path)
                  
                    img = img.resize(img_wh, Image.BICUBIC)
                    img = TF.to_tensor(img).permute(1, 2, 0)[...,:3]
                    img = img.to(self.rank) if self.config.load_data_on_gpu else img.cpu()
                    if apply_mask:
                        mask_paths = [os.path.join(mask_dir, d.name), os.path.join(mask_dir, d.name[3:])]
                        mask_paths = list(filter(os.path.exists, mask_paths))
                        assert len(mask_paths) == 1
                        mask = Image.open(mask_paths[0]).convert('L') # (H, W, 1)
                        mask = mask.resize(img_wh, Image.BICUBIC)
                        mask = TF.to_tensor(mask)[0]
                    else:
                        mask = torch.ones_like(img[...,0], device=img.device)
                                     
                   
                    all_fg_masks.append(mask) # (h, w)
                    all_images.append(img)
                    all_names.append(d.name.split(".")[0])
                 
                   
                    if_corrected.append(1)

                else:
                    if i % 8 == 0: continue
                    
                  
                    special_c2w.append(c2w)
                    
                    img = Image.open(img_path)
                   

                    img = img.resize(img_wh, Image.BICUBIC)
                    img = TF.to_tensor(img).permute(1, 2, 0)[...,:3]
                    img = img.to(self.rank) if self.config.load_data_on_gpu else img.cpu()
                    if apply_mask:
                        mask_paths = [os.path.join(mask_dir, d.name), os.path.join(mask_dir, d.name[3:])]
                        mask_paths = list(filter(os.path.exists, mask_paths))
                      
                        assert len(mask_paths) == 1
                        mask = Image.open(mask_paths[0]).convert('L') # (H, W, 1)
                        mask = mask.resize(img_wh, Image.BICUBIC)
                        mask = TF.to_tensor(mask)[0]
                    else:
                        mask = torch.ones_like(img[...,0], device=img.device)
                    
                  
                
                    all_fg_masks.append(mask) # (h, w)
                    all_images.append(img)
                    all_names.append(d.name.split(".")[0])
                   
                    if_corrected.append(1)

          

            all_c2w = torch.stack(all_c2w, dim=0)   
            special_c2w = torch.stack(special_c2w, dim=0)   
            pts_path = os.path.join(self.config.root_dir, 'sparse/0/points3D.bin')
            if os.path.exists(pts_path):
                pts3d = read_points3d_binary(pts_path)
            else:
                pts_path = os.path.join(self.config.root_dir, 'colmap/points3D.txt')
                pts3d = read_points3d_text(pts_path)

         
            try:
                pts3d = torch.from_numpy(np.array([pts3d[k].xyz for k in pts3d])).float()
            except:
                pts3d = torch.from_numpy(pts3d[0]).float()
            #normalize the pt and camera poses accordingto the given parameters or automaticaly
            if self.config.neuralangelo_scale!=0.0 or self.config.neuralangelo_center!=[0,0,0]:
                all_c2w, pts3d, special_c2w = normalize_poses_with_given(all_c2w, pts3d, special_c2w,self.config.neuralangelo_center, self.config.neuralangelo_scale)
            else:
                
                all_c2w, pts3d, special_c2w = simple_normalize_poses(all_c2w, pts3d, special_c2w)
            ColmapDatasetBase.properties = {
                'w': w,
                'h': h,
                'img_wh': img_wh,
                'factor': factor,
                'has_mask': has_mask,
                'apply_mask': apply_mask,
                'directions': directions,
                'pts3d': pts3d,
                'all_c2w': special_c2w,
                'all_images': all_images,
                'all_fg_masks': all_fg_masks,
                'all_names': all_names,
                'if_corrected':if_corrected
            }

        
        for k, v in ColmapDatasetBase.properties.items():
            setattr(self, k, v)

        
        self.all_images, self.all_fg_masks = torch.stack(self.all_images, dim=0).float(), torch.stack(self.all_fg_masks, dim=0).float()

        """
        # for debug use
        from models.ray_utils import get_rays
        rays_o, rays_d = get_rays(self.directions.cpu(), self.all_c2w, keepdim=True)
        pts_out = []
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 1.0 0.0 0.0' for l in rays_o[:,0,0].reshape(-1, 3).tolist()]))

        t_vals = torch.linspace(0, 1, 8)
        z_vals = 0.05 * (1 - t_vals) + 0.5 * t_vals

        ray_pts = (rays_o[:,0,0][..., None, :] + z_vals[..., None] * rays_d[:,0,0][..., None, :])
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 0.0 1.0 0.0' for l in ray_pts.view(-1, 3).tolist()]))

        ray_pts = (rays_o[:,0,0][..., None, :] + z_vals[..., None] * rays_d[:,self.h-1,0][..., None, :])
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 0.0 0.0 1.0' for l in ray_pts.view(-1, 3).tolist()]))

        ray_pts = (rays_o[:,0,0][..., None, :] + z_vals[..., None] * rays_d[:,0,self.w-1][..., None, :])
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 0.0 1.0 1.0' for l in ray_pts.view(-1, 3).tolist()]))

        ray_pts = (rays_o[:,0,0][..., None, :] + z_vals[..., None] * rays_d[:,self.h-1,self.w-1][..., None, :])
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 1.0 1.0 1.0' for l in ray_pts.view(-1, 3).tolist()]))
        
        open('cameras.txt', 'w').write('\n'.join(pts_out))
        open('scene.txt', 'w').write('\n'.join([' '.join([str(p) for p in l]) + ' 0.0 0.0 0.0' for l in self.pts3d.view(-1, 3).tolist()]))

        exit(1)
        """

        self.all_c2w = self.all_c2w.float().to(self.rank)
        if self.config.load_data_on_gpu:
            self.all_images = self.all_images.to(self.rank) 
            self.all_fg_masks = self.all_fg_masks.to(self.rank)
            
         

        
   
class ColmapDataset(Dataset, ColmapDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class ColmapIterableDataset(IterableDataset, ColmapDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@register('colmap')
class ColmapDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = ColmapIterableDataset(self.config, 'train')
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = ColmapDataset(self.config, self.config.get('val_split', 'val'))
        if stage in [None, 'test']:
            self.test_dataset = ColmapDataset(self.config, self.config.get('test_split', 'test'))
        if stage in [None, 'predict']:
            self.predict_dataset = ColmapDataset(self.config, 'train')         

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       
