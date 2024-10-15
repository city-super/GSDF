#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import torch
import torch.nn.functional as F

import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

from gaussian_splatting.scene import Scene
import json
import time
from gaussian_splatting.gaussian_renderer import render, prefilter_voxel
import torchvision
from tqdm import tqdm
from gaussian_splatting.utils.general_utils import safe_state
from argparse import ArgumentParser
from gaussian_splatting.arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_splatting.gaussian_renderer import GaussianModel
from os import makedirs
import matplotlib.pyplot as plt
import cv2

# def colorize_depth_maps(depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None):
#         """
#         Colorize depth maps.
#         """
#         assert len(depth_map.shape) >= 2, "Invalid dimension"

#         if isinstance(depth_map, torch.Tensor):
#             depth = depth_map.detach().clone().squeeze().numpy()
#         elif isinstance(depth_map, np.ndarray):
#             depth = depth_map.copy().squeeze()
#         # reshape to [ (B,) H, W ]
#         if depth.ndim < 3:
#             depth = depth[np.newaxis, :, :]

#         # colorize
#         cm = matplotlib.colormaps[cmap]
#         depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
#         img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
#         img_colored_np = np.rollaxis(img_colored_np, 3, 1)

#         if valid_mask is not None:
#             if isinstance(depth_map, torch.Tensor):
#                 valid_mask = valid_mask.detach().numpy()
#             valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
#             if valid_mask.ndim < 3:
#                 valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
#             else:
#                 valid_mask = valid_mask[:, np.newaxis, :, :]
#             valid_mask = np.repeat(valid_mask, 3, axis=1)
#             img_colored_np[~valid_mask] = 0

#         if isinstance(depth_map, torch.Tensor):
#             img_colored = torch.from_numpy(img_colored_np).float()
#         elif isinstance(depth_map, np.ndarray):
#             img_colored = img_colored_np

#         return img_colored
def apply_colormap_and_save(normal_map, path, idx, colormap='Spectral'):
    """
    Applies a colormap to each channel of a 3-channel image and saves the result.
    
    Args:
    - normal_map (torch.Tensor): 3-channel normal map with shape [3, H, W].
    - path (str): Path to save the image.
    - idx (int): Image index for filename.
    - colormap (str): Colormap name.
    """
    # Ensure tensor is on CPU and detach from any computation graph
    normal_map = normal_map.cpu().detach()
    
    # Normalize the normal map to [0, 1]
    normalized_map = (normal_map - normal_map.min()) / (normal_map.max() - normal_map.min())
    
    # Initialize an empty array for the colored image
    colored_image = np.zeros((*normalized_map.shape[1:], 3), dtype=np.float32)
    
    # Apply colormap to each channel
    for i in range(normalized_map.shape[0]):
        channel = normalized_map[i].numpy()
        colored_channel = plt.get_cmap(colormap)(channel)[:, :, :3]  # Exclude alpha channel
        colored_image += colored_channel / normal_map.shape[0]  # Average the contributions
    
    # Convert numpy array back to tensor
    colored_tensor = torch.from_numpy(colored_image).permute(2, 0, 1)
    
    # Save the tensor as an image
    filename = os.path.join(path, '{:05d}.png'.format(idx))
    torchvision.utils.save_image(colored_tensor, filename)


def get_rgb_image_(img):
    img = img.cpu().numpy()
    # assert data_format in ['CHW', 'HWC']
    # if data_format == 'CHW':
    img = img.transpose(1, 2, 0)
    img = img.clip(min=-1, max=1)
    img = ((img - (-1)) / (2) * 255.).astype(np.uint8)
    imgs = [img[...,start:start+3] for start in range(0, img.shape[-1], 3)]
    imgs = [img_ if img_.shape[-1] == 3 else np.concatenate([img_, np.zeros((img_.shape[0], img_.shape[1], 3 - img_.shape[2]), dtype=img_.dtype)], axis=-1) for img_ in imgs]
    img = np.concatenate(imgs, axis=1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depths")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    lod08_path = os.path.join(model_path, name, "ours_{}".format(iteration), "lods08")
    lod12_path = os.path.join(model_path, name, "ours_{}".format(iteration), "lods12")
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normals")
    opacity_path = os.path.join(model_path, name, "ours_{}".format(iteration), "opacitys")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")


    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(lod08_path, exist_ok=True)
    makedirs(lod12_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)
    makedirs(opacity_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    name_list = []
    per_view_dict = {}
    # debug = 0
    t_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

     
        torch.cuda.synchronize(); t0 = time.time()
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask,out_depth=True,return_normal=True)
        torch.cuda.synchronize(); t1 = time.time()
        
        t_list.append(t1-t0)

        rendering = render_pkg["render"]
        depth_gs = render_pkg["depth_hand"]
        depth_gs=depth_gs/depth_gs.max()
        normal_gs = render_pkg["gs_normal"]
        normal_gs_normal=(F.normalize(normal_gs, p=2, dim=0)+1)/2
        # import pdb;pdb.set_trace()
        gt = view.original_image[0:3, :, :]
        # import pdb;pdb.set_trace()

        name_list.append('{0:05d}'.format(idx) + ".png")
        # import pdb;pdb.set_trace()
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # save_image_with_colormap(depth_gs,depth_path,idx)
        # apply_colormap_and_save(normal_gs_normal,normal_path,idx)
        # apply_colormap_and_save(depth_gs,depth_path,idx)

        # normal_gs_normal = get_rgb_image_(normal_gs)
        torchvision.utils.save_image(depth_gs, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        # cv2.imwrite(os.path.join(normal_path, '{0:05d}'.format(idx) + ".png"), normal_gs_normal)

        torchvision.utils.save_image(normal_gs_normal, os.path.join(normal_path, '{0:05d}'.format(idx) + ".png"))
        # DEFAULT_RGB_KWARGS = {'data_format': 'CHW', 'data_range': (0, 1)}
        # DEFAULT_UV_KWARGS = {'data_format': 'CHW', 'data_range': (0, 1), 'cmap': 'checkerboard'}
        # DEFAULT_GRAYSCALE_KWARGS = {'data_range': None, 'cmap': 'jet'}
        # if col['type'] == 'rgb':
        #         rgb_kwargs = DEFAULT_RGB_KWARGS.copy()
        #         rgb_kwargs.update(col['kwargs'])
        #         cols.append(self.get_rgb_image_(col['img'], **rgb_kwargs))
        #     elif col['type'] == 'uv':
        #         uv_kwargs = self.DEFAULT_UV_KWARGS.copy()
        #         uv_kwargs.update(col['kwargs'])
        #         cols.append(self.get_uv_image_(col['img'], **uv_kwargs))
        #     elif col['type'] == 'grayscale':
        #         grayscale_kwargs = self.DEFAULT_GRAYSCALE_KWARGS.copy()
        #         grayscale_kwargs.update(col['kwargs'])
        #         cols.append(self.get_grayscale_image_(col['img'], **grayscale_kwargs))

        

        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)      
     
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,given_center=[0,0,0], given_scale=0.0):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False,given_scale=given_scale,given_center=given_center)
        
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--config",required=True, help='path to config file,for the normalization parameters')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    from instant_nsr.utils.misc import load_config    

    # parse YAML config to OmegaConf
    config = load_config(args.config)
    

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, given_center=config.dataset.neuralangelo_center, given_scale=config.dataset.neuralangelo_scale)
