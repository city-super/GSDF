import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss
from plyfile import PlyData, PlyElement
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug

import time

import instant_nsr.models
from instant_nsr.models.utils import cleanup
from instant_nsr.models.ray_utils import get_rays
import instant_nsr.systems
from instant_nsr.systems.base import BaseSystem
from instant_nsr.systems.criterions import PSNR, binary_cross_entropy
from instant_nsr.utils.loss_utils import l1_loss, ssim
from gaussian_splatting import gaussian_renderer
import sys
from gaussian_splatting.scene import Scene, GaussianModel
from gaussian_splatting.utils.general_utils import safe_state
import uuid
from gaussian_splatting.utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from gaussian_splatting.arguments import ModelParams, PipelineParams, OptimizationParams
import os
import numpy as np
from random import randint
from tqdm import tqdm
from gaussian_splatting.scene.cameras import Camera
import torchvision
# from gaussian_splatting.utils.misc import config_to_primitive
import random
from pathlib import Path
from gaussian_splatting.lpipsPyTorch import lpips
import json
from os import makedirs
from PIL import Image
import torchvision.transforms.functional as tf
from instant_nsr.systems import register

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)

    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                
                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask = gaussian_renderer.prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask,out_depth=True,return_normal=True)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    depth_gs = render_pkg["depth_hand"]
                    depth_gs=depth_gs/depth_gs.max()
                    normal_gs = render_pkg["gs_normal"]
                    normal_gs_normal=(F.normalize(normal_gs, p=2, dim=0)+1)/2

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 38):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/normal".format(viewpoint.image_name), normal_gs_normal[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth_gs[None], global_step=iteration)

                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if wandb:
                                gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
          
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                
                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb is not None:
                    wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})
                
        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
            
        torch.cuda.empty_cache()

def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 

    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger


@register('neus-system')
class NeuSSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def __init__(self, config):
        super().__init__(config)
        self.current_epoch_set=0
        self.pretrain_step = 15000
        self.geometry_awared_control = False

        if self.config.model.if_gaussian:
            parser = ArgumentParser(description="Training script parameters")
            parser.source_path = config.dataset.root_dir
            print(parser.source_path)
            lp = ModelParams(parser)
            op = OptimizationParams(parser)
            pp = PipelineParams(parser)
            parser.add_argument('--ip', type=str, default="127.0.0.1")
            parser.add_argument('--port', type=int, default=6009)
            parser.add_argument('--debug_from', type=int, default=-1)
            parser.add_argument('--detect_anomaly', action='store_true', default=False)
            parser.add_argument("--test_iterations", nargs="+", type=int, default=[self.pretrain_step,self.pretrain_step+15000, self.pretrain_step+30000,self.pretrain_step+100000])
            parser.add_argument("--save_iterations", nargs="+", type=int, default=[self.pretrain_step,self.pretrain_step+15000, self.pretrain_step+30000,self.pretrain_step+100000])
            parser.add_argument("--quiet", action="store_true")
            parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
            parser.add_argument("--start_checkpoint", type=str, default = None)
            parser.add_argument('--if_merging', action='store_true', help="if using merging operator") 
            parser.add_argument('--config', required=True, help='path to config file')
            parser.add_argument('--gpu', default='0', help='GPU(s) to be used')
            # parser.add_argument('--normal_w', type=float, default=0.01, help='weight of normal loss')
            # parser.add_argument('--depth_w', type=float, default=0.01, help='weight of depth loss')
            # parser.add_argument('--growing_weight', type=float, default=0.0002, help='weight of growing operator')
            parser.add_argument('tag', default='test')
            parser.add_argument('--add', type=int, default=0)
            parser.add_argument('--exp_dir', default='./exp')
            group = parser.add_mutually_exclusive_group(required=True)
            group.add_argument('--train', action='store_true')
            out_path = "output/"+config.tag
            fake_input = ["--source_path",config.dataset.root_dir,"--model_path",out_path]
            fake_input.extend(sys.argv[1:])
            args = parser.parse_args(fake_input)
            os.makedirs(args.model_path, exist_ok=True)
            print(f'model_path: {args.model_path}')
            self.loggger = get_logger(args.model_path)
            self.loggger.info(f'args: {args}')
            self.tb_writer = self.prepare_output_and_logger(lp.extract(args))
            safe_state(args.quiet)                        
            # Start GUI server, configure and run training
            # network_gui.init(args.ip, args.port)
            torch.autograd.set_detect_anomaly(args.detect_anomaly)
            args.resolution = config.dataset.img_downscale
            self.args=args
            bg_color = [1, 1, 1] if lp.extract(args).white_background else [0, 0, 0]
            self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            
            self.op =op.extract(args)
            self.piplin=pp.extract(args)
            self.lp = lp.extract(args)
            self.saving_iterations = args.save_iterations
            self.testing_iterations = args.test_iterations
            self.gaussians = GaussianModel(self.lp.feat_dim, self.lp.n_offsets, self.lp.voxel_size, self.lp.update_depth, self.lp.update_init_factor, self.lp.update_hierachy_factor, self.lp.use_feat_bank, self.lp.use_tcnn)
            self.ema_loss_for_log = 0.0
            self.dataset_size=0
            self.last_iteration_time=0
            #Using a pretrained Scaffold-GS
            if self.config.model.using_pretrain:
                self.scene = Scene(self.lp, self.gaussians, load_iteration=15000, shuffle=False, if_pretrain=self.config.model.using_pretrain,pretrain_path=self.config.model.using_pretrain_path,given_scale=self.config.dataset.neuralangelo_scale,given_center=self.config.dataset.neuralangelo_center)
                self.gaussians.training_setup(self.op)
                self.gaussians.update_learning_rate(15000)
                self.progress_bar = tqdm(range(15000, self.op.iterations), desc="Training progress")
                self.viewpoint_stack = self.scene.getTrainCameras().copy()
                self.viewpoint_candidate = self.scene.getTrainCameras().copy()
            #Pretrain Scaffold-GS from scratch.
            else:
                self.progress_bar = tqdm(range(0, self.op.iterations), desc="Training progress")               
                self.scene = Scene(self.lp, self.gaussians, shuffle=False,given_scale=self.config.dataset.neuralangelo_scale,given_center=self.config.dataset.neuralangelo_center)
                
                self.gaussians.training_setup(self.op)
                self.viewpoint_stack = self.scene.getTrainCameras().copy()
                self.viewpoint_candidate = self.scene.getTrainCameras().copy()
                # pretrain scaffold gs
                self.pretrain_gs()
          
    def prepare(self):
        self.criterions = {
            'psnr': PSNR()
        }
        self.train_num_samples = self.config.model.train_num_rays * (self.config.model.num_samples_per_ray + self.config.model.get('num_samples_per_ray_bg', 0))
        self.train_num_rays = self.config.model.train_num_rays

    def forward(self, batch, gs_depth=None, use_depth_guide=False):
        return self.model(batch['rays'], gs_depth, use_depth_guide)
    
    def preprocess_data(self, batch, stage):
        
        if 'index' in batch: # validation / testing
            index = batch['index']
        else:
            if self.config.model.batch_image_sampling:
                index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,), device=self.dataset.all_images.device)
                
            else:
                index = torch.randint(0, len(self.dataset.all_images), size=(1,), device=self.dataset.all_images.device)
        if stage in ['train']:
            c2w = self.dataset.all_c2w[index]
            x = torch.randint(
                0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            y = torch.randint(
                0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions[y, x]
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index, y, x]
            rays_o, rays_d = get_rays(directions, c2w)

            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1).to(self.rank)

        else:
            c2w = self.dataset.all_c2w[index][0]
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index][0] 
            rays_o, rays_d = get_rays(directions, c2w)
            
            rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index].view(-1).to(self.rank)


        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        if stage in ['train']:
            if self.config.model.background_color == 'white':
                self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
            elif self.config.model.background_color == 'random':
                self.model.background_color = torch.rand((3,), dtype=torch.float32, device=self.rank)
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
        
        if self.dataset.apply_mask:
            rgb = rgb * fg_mask[...,None] + self.model.background_color * (1 - fg_mask[...,None])
        if stage in ['train']:
            batch.update({
                'rays': rays,
                'rgb': rgb,
                'fg_mask': fg_mask,
                'used_index': index,
                'used_y': y,
                'used_x': x,
            }) 
        else:
            batch.update({
                'rays': rays,
                'rgb': rgb,
                'fg_mask': fg_mask,
            })

    #Only training Scaffold-GS for the first 'pretrain_step' iterations
    def pretrain_gs(self):
        datasetname=self.args.source_path.split('/')[-1]
        for iteration in range(0, self.pretrain_step + 1): 
            iter_start = torch.cuda.Event(enable_timing = True)
            iter_end = torch.cuda.Event(enable_timing = True)
            iter_start.record()
            self.gaussians.update_learning_rate(iteration)

            if not self.viewpoint_stack:
                self.viewpoint_stack = self.scene.getTrainCameras().copy()
                self.dataset_size = len(self.viewpoint_stack)
                
            viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack)-1))
            random_background = torch.rand(3).cuda()
            # voxel_visible_mask = gaussian_renderer.prefilter_voxel(viewpoint_cam, self.gaussians, self.piplin, self.background)
            voxel_visible_mask = gaussian_renderer.prefilter_voxel(viewpoint_cam, self.gaussians, self.piplin, random_background)

            retain_grad = (iteration < self.op.update_until and iteration >= 0)

            time2=time.time()

            # render_pkg = gaussian_renderer.render(viewpoint_cam, self.gaussians, self.piplin, self.background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)
            render_pkg = gaussian_renderer.render(viewpoint_cam, self.gaussians, self.piplin, random_background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)

            image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]
            time3=time.time()
            time_2=time3-time2
            self.tb_writer.add_scalar(f'{datasetname}'+'/pure_gs_forward', time_2, iteration)
            if iteration <= self.op.iterations:
                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)
                scaling_reg = scaling.prod(dim=1).mean()
                loss_gaussian= (1.0 - self.op.lambda_dssim) * Ll1 + self.op.lambda_dssim * (1.0 - ssim(image, gt_image)) + 0.01*scaling_reg
                loss_gaussian.backward()

                time4=time.time()
                time_3=time4-time3
                self.tb_writer.add_scalar(f'{datasetname}'+'/pure_gs_backward', time_3, iteration)
                iter_end.record()

                with torch.no_grad():
                    self.ema_loss_for_log = 0.4 * loss_gaussian.item() + 0.6 * self.ema_loss_for_log
                    if iteration % 10 == 0:
                        self.progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
                        self.progress_bar.update(10)
                    if iteration == self.op.iterations:
                        self.progress_bar.close()
                    training_report(self.tb_writer, self.args.source_path.split('/')[-1], iteration, Ll1, loss_gaussian, l1_loss, iter_start.elapsed_time(iter_end), self.args.test_iterations, self.scene, gaussian_renderer.render, (self.piplin, self.background), None, self.loggger)
                    if (iteration in self.saving_iterations):
                        self.loggger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                        self.scene.save(iteration)

                    if iteration < self.op.update_until and iteration > self.op.start_stat:
                        self.gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                        
                        # densification
                        if iteration > self.op.update_from and iteration % 100 ==0: # opt.update_intern_interval == 0:
                            self.gaussians.adjust_anchor(check_interval=self.op.update_interval, extent=self.scene.cameras_extent, success_threshold=self.op.success_threshold, grad_threshold=self.op.densify_grad_threshold, min_opacity=self.op.min_opacity)

                    # Optimizer step
                    if iteration < self.op.iterations:
                        self.gaussians.optimizer.step()
                        self.gaussians.optimizer.zero_grad(set_to_none = True)

                    if (iteration in self.args.checkpoint_iterations):
                        # if 'debug' not in scene.model_path:
                        self.loggger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                        torch.save((self.gaussians.capture(), iteration), self.scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    
    # vector similarity
    def cos_similarity_loss(self, a, b):
        return 1.0-((a*b).sum(dim=-1) / (a.norm(dim=-1)*b.norm(dim=-1)+1e-8)).abs().mean()

    # Training step for both Scaffold-GS and Instant-nsr
    def training_step(self, batch, batch_idx):
        random_background = torch.rand(3).cuda()
        datasetname=self.args.source_path.split('/')[-1]
        time1=time.time()

        if self.last_iteration_time!=0:
            time_5=time1-self.last_iteration_time
            self.tb_writer.add_scalar(f'{datasetname}'+'/time_5', time_5, self.current_epoch_set)

        self.current_epoch_set=self.current_epoch_set+1

        #inite loss of gs
        loss_gaussian=0

        # Reducing the normal and depth loss weight in the later iterations
        if self.current_epoch_set > 15000:
        # if self.current_epoch_set > (self.op.iterations-15000)/2:
            self.config.system.loss.normal_w = self.config.system.loss.normal_w/10
            self.config.system.loss.depth_w = self.config.system.loss.depth_w/10

        # Training for Scaffold-GS
        if self.config.model.if_gaussian:
            current_epoch_gs = self.current_epoch_set + self.pretrain_step
            iter_start = torch.cuda.Event(enable_timing = True)
            iter_end = torch.cuda.Event(enable_timing = True)
            iter_start.record()
            self.gaussians.update_learning_rate(current_epoch_gs)

            # Get the same image index as Instant-nsr
            viewpoint_cam = self.scene.getTrainCameras()[batch['used_index']]

            # Get the same pixel indexes as Instant-nsr
            yy = batch['used_y']
            xx = batch['used_x']

            ## Forward of Scaffold-GS
            # filter 3D Gaussians out of frumstum.
            # voxel_visible_mask = gaussian_renderer.prefilter_voxel(viewpoint_cam, self.gaussians, self.piplin, self.background)
            voxel_visible_mask = gaussian_renderer.prefilter_voxel(viewpoint_cam, self.gaussians, self.piplin, random_background)

            retain_grad = (current_epoch_gs < self.op.update_until and current_epoch_gs >= 0)
            render_pkg = gaussian_renderer.render(viewpoint_cam, self.gaussians, self.piplin, random_background, visible_mask=voxel_visible_mask, retain_grad=retain_grad, out_depth=True, return_normal=True, radius=self.config.model.radius)

            # render_pkg = gaussian_renderer.render(viewpoint_cam, self.gaussians, self.piplin, self.background, visible_mask=voxel_visible_mask, retain_grad=retain_grad, out_depth=True, return_normal=True, radius=self.config.model.radius)
            image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity_gs, gs_depth_hand,gs_normal = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"], render_pkg["depth_hand"], render_pkg["gs_normal"]
            gs_depth = gs_depth_hand.mean(dim=0,keepdim=True).permute(1, 2, 0)
            picked_gs_depth = gs_depth[yy,xx]
            gs_normal = gs_normal.permute(1, 2, 0)
            picked_gs_normal = gs_normal[yy,xx]
            time2=time.time()
            time_1=time2-time1
            
            self.tb_writer.add_scalar(f'{datasetname}'+'/time_1', time_1, self.current_epoch_set)
        # Using predicted depth of Scaffold-GS to guide the ray sampling of Instant-nsr after warm-up of Instant-nsr.
        picked_gs_depth_dt = picked_gs_depth.detach()
        # if self.current_epoch_set > self.config.model.geometry.xyz_encoding_config.start_step and self.current_epoch_set%500>100:
        if self.current_epoch_set > self.config.model.geometry.xyz_encoding_config.start_step:
            out = self(batch, picked_gs_depth_dt, use_depth_guide=True) 
        else:
            out = self(batch, picked_gs_depth_dt, use_depth_guide=False)
        time3=time.time()
        time_2=time3-time2
        self.tb_writer.add_scalar(f'{datasetname}'+'/time_2', time_2, self.current_epoch_set)

        # If all the sampled pixels are belong to background, skiping this training iteration. 
        if out['zero_samples']==True:
            return None
        
        # loss of Instant-NSR
        loss = 0.

        # predicted normal and depth of Scaffold-GS, taken as GT of the Instant-NSR side
        fixed_picked_gs_normal = picked_gs_normal[out['rays_valid'][...,0]].detach()
        fixed_picked_gs_depth = picked_gs_depth[out['rays_valid'][...,0]].detach()
        # The depth loss for the Instant-nsr.
        diff_neus = torch.abs(out['depth'][out['rays_valid'][...,0]] - fixed_picked_gs_depth)

        # Filter out the huge depth differents, which could be the impact of background.
        if self.current_epoch_set > self.config.model.geometry.xyz_encoding_config.start_step:
            depth_ratio = 10.0
        else:
            depth_ratio = 2.0
        diff_neus[diff_neus > self.config.model.radius/depth_ratio] = 0
        diff_neus_count = (diff_neus>0).sum()
        loss_depth_L1 = diff_neus.sum() / (diff_neus_count+1e-8)
        # normalzied the depth loss by the frontground size.
        loss += loss_depth_L1 * self.C(self.config.system.loss.depth_w)/self.config.model.radius
        self.log('train/loss_depth_L1_neus', float(loss_depth_L1/self.config.model.radius))

        # The normal loss for the Instant-nsr is only taken into account after the warmup period.
        if self.current_epoch_set > self.config.model.geometry.xyz_encoding_config.start_step:
            normal_diff = self.cos_similarity_loss(fixed_picked_gs_normal,out['comp_normal'][out['rays_valid'][...,0]])
            loss +=  normal_diff * self.config.system.loss.normal_w
        else:
            normal_diff = self.cos_similarity_loss(fixed_picked_gs_normal,out['comp_normal'][out['rays_valid'][...,0]])
            loss +=  normal_diff * 0.0
        self.log('train/normal_loss_neus', normal_diff)

        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples_full'].sum().item()))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)
        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)

        # RGB L1 loss
        loss_rgb_l1 = F.l1_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
        self.log('train/loss_rgb', loss_rgb_l1)
        loss += loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)        

        # Eikonal loss
        loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()
        self.log('train/loss_eikonal', loss_eikonal)
        loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)

        # Curvature loss, Note that the curvature loss weight is adaptived to the training iteration.
        if self.C(self.config.system.loss.lambda_smoothing)>0:
            loss_smoothing = out['smoothing'].abs().mean()
            self.log('train/loss_smoothing', loss_smoothing)
            
            loss+=  loss_smoothing * self.C(self.config.system.loss.lambda_smoothing)
          
        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_
        
        for name, value in self.config.system.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))

        # log
        self.log('train/inv_s', out['inv_s'], prog_bar=True)
        
        time4=time.time()
        time_3=time4-time3
        self.tb_writer.add_scalar(f'{datasetname}'+'/time_3', time_3, self.current_epoch_set)

        # Calculate the loss of Scaffold-GS
        if self.config.model.if_gaussian:
            if current_epoch_gs <= self.op.iterations:
                gt_image = viewpoint_cam.original_image.cuda()

                # RGB L1 loss
                Ll1 = l1_loss(image, gt_image)
                self.log('train/GS_render_loss', Ll1)

                # Scalling loss 
                scaling_reg = scaling.prod(dim=1).mean()
                self.log('train/GS_scaling_reg', scaling_reg)
                
                # Predicted depth and normal of Instant-NSR, taken as GT of the GS side.
                fixed_neus_picked_depth = out['depth'][out['rays_valid'][...,0]].detach()
                fixed_neus_picked_normal = out['comp_normal'][out['rays_valid'][...,0]].detach()
               
                #SSIM loss
                ssim_loss = 1.0 - ssim(image, gt_image)
                self.log('train/GS_ssim_loss', ssim_loss)

                # Normal loss of the GS side.
                # Ignore normal loss in the warmup period of Instant-NSR 
                if self.current_epoch_set < self.config.model.geometry.xyz_encoding_config.start_step:                
                    normal_loss_gs = 0.0
                else:
                    normal_loss_gs = self.cos_similarity_loss(picked_gs_normal[out['rays_valid'][...,0]],fixed_neus_picked_normal)* self.config.system.loss.normal_w

                self.log('train/GS_normal_loss_gs', normal_loss_gs)
                # depth loss of GS side
                diff = torch.abs(fixed_neus_picked_depth - picked_gs_depth[out['rays_valid'][...,0]])
                
                # Filter out the huge depth differents, which could be the impact of background.
                depth_ratio = 10.0
                
                diff[diff > self.config.model.radius/depth_ratio] = 0
                
                diff_count = (diff>0.0).sum()

                loss_depth_L1_gs = diff.sum() / (diff_count+1e-8)
                self.log('train/GS_loss_depth', loss_depth_L1_gs)

                # normalzied the depth loss by the frontground size.
                depth_loss_gs = loss_depth_L1_gs * self.C(self.config.system.loss.depth_w)/self.config.model.radius

                # GS loss and backward
                loss_gaussian= (1.0 - self.op.lambda_dssim) * Ll1 + \
                    self.op.lambda_dssim * ssim_loss + 0.01*scaling_reg + \
                        depth_loss_gs  + \
                            normal_loss_gs
                
                self.log('train/loss_gaussian', float(loss_gaussian))
                time41=time.time()
                time_41=time41-time4
                self.tb_writer.add_scalar(f'{datasetname}'+'/time_41', time_41, self.current_epoch_set)

                loss_gaussian.backward()
                iter_end.record()

                time42=time.time()
                time_42=time42-time41
                self.tb_writer.add_scalar(f'{datasetname}'+'/time_42', time_42, self.current_epoch_set)

                # GS densification
                with torch.no_grad():
                    self.ema_loss_for_log = 0.4 * loss_gaussian.item() + 0.6 * self.ema_loss_for_log
                    
                    if current_epoch_gs % 10 == 0:
                        self.progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
                        self.progress_bar.update(10)
                    
                    if current_epoch_gs == self.op.iterations:
                        self.progress_bar.close()
                    training_report(self.tb_writer, self.args.source_path.split('/')[-1], current_epoch_gs, Ll1, loss_gaussian, l1_loss, iter_start.elapsed_time(iter_end), self.args.test_iterations, self.scene, gaussian_renderer.render, (self.piplin, self.background), None, self.loggger)
                    if (current_epoch_gs in self.saving_iterations):
                        self.loggger.info("\n[ITER {}] Saving Gaussians".format(current_epoch_gs))
                        self.scene.save(current_epoch_gs)
                    time43=time.time()
                    time_43=time43-time42
                    self.tb_writer.add_scalar(f'{datasetname}'+'/time_43', time_43, self.current_epoch_set)
                    if current_epoch_gs < self.op.update_until and current_epoch_gs > self.op.start_stat: 
                        
                        self.gaussians.training_statis(viewspace_point_tensor, opacity_gs, visibility_filter, offset_selection_mask, voxel_visible_mask, grad_threshold=self.op.densify_grad_threshold)
                        
                        if current_epoch_gs > self.op.update_from and current_epoch_gs % 100 == 0: # opt.update_intern_interval == 0:
                            if self.geometry_awared_control:
                                # Original density control
                                self.gaussians.adjust_anchor(check_interval=self.op.update_interval, extent=self.scene.cameras_extent, success_threshold=self.op.success_threshold, grad_threshold=self.op.densify_grad_threshold, min_opacity=self.op.min_opacity, growing_weight=self.config.system.growing_weight)
                                
                            else:
                                #  Density control guided by predicted sdf
                                if self.current_epoch_set > self.config.model.geometry.xyz_encoding_config.start_step:
                                    # guide density control after warmup of Instant-nsr
                                    # Identify the 3D Gaussians in the frontground
                                    scaling = self.gaussians.get_scaling[:,:3]
                                    scaling_repeat = scaling.unsqueeze(dim=1).repeat([1, self.gaussians.n_offsets, 1]).view([-1, 3]) 
                                    gs_positions = self.gaussians.get_anchor.unsqueeze(dim=1).repeat([1, self.gaussians.n_offsets, 1]).view([-1, 3]) + self.gaussians._offset.view([-1, 3])*scaling_repeat

                                    min_point = torch.tensor([-self.config.model.radius, -self.config.model.radius, -self.config.model.radius],device=gs_positions.device)
                                    max_point = torch.tensor([self.config.model.radius, self.config.model.radius, self.config.model.radius],device=gs_positions.device)
                                    inside_box = (gs_positions > min_point) & (gs_positions < max_point)
                                    inside_box = inside_box.all(dim=1)

                                    inside_positions = gs_positions[inside_box]
                                    # set the sdf of 3D gaussians in the background to 100000.
                                    xyz_sdf = torch.ones(gs_positions.shape[0]).to(gs_positions.device)*100000
                                    # calculate the sdf of 3D Gaussians in the frontground.
                                    inside_xyz_sdf = self.model.geometry(inside_positions, with_grad=False, with_feature=False)

                                    xyz_sdf[inside_box] = inside_xyz_sdf
                                    # calculate the sdf of anchor points in the frontground
                                    anchor_positions = self.gaussians.get_anchor
                                    anchor_inside_box = (anchor_positions > min_point) & (anchor_positions < max_point)
                                    anchor_inside_box = anchor_inside_box.all(dim=1)
                                    anchor_sdf = self.model.geometry(anchor_positions, with_grad=False, with_feature=False)
                                      
                                else:
                                    # using the original density control in the warmup of Instant-nsr
                                    xyz_sdf=None
                                    anchor_sdf=None
                                    inside_box=None
                                    anchor_inside_box=None
                                self.gaussians.adjust_anchor(check_interval=self.op.update_interval, extent=self.scene.cameras_extent, success_threshold=self.op.success_threshold, grad_threshold=self.op.densify_grad_threshold, min_opacity=self.op.min_opacity, xyz_sdf=xyz_sdf, anchor_sdf=anchor_sdf, inside_box=inside_box, anchor_inside_box=anchor_inside_box, growing_weight=self.config.system.growing_weight)

                    elif current_epoch_gs == self.op.update_until:
                        del self.gaussians.opacity_accum
                        del self.gaussians.offset_gradient_accum
                        del self.gaussians.offset_denom
                        torch.cuda.empty_cache()
                    time44=time.time()
                    time_44=time44-time43
                    self.tb_writer.add_scalar(f'{datasetname}'+'/time_44', time_44, self.current_epoch_set)
                    
                    # Optimizer step
                    if current_epoch_gs < self.op.iterations:
                        self.gaussians.optimizer.step()
                        self.gaussians.optimizer.zero_grad(set_to_none = True)

                    if (current_epoch_gs in self.args.checkpoint_iterations):
                        self.loggger.info("\n[ITER {}] Saving Checkpoint".format(current_epoch_gs))
                        torch.save((self.gaussians.capture(), current_epoch_gs), self.scene.model_path + "/chkpnt" + str(current_epoch_gs) + ".pth")
                    time45=time.time()
                    time_45=time45-time44
                    self.tb_writer.add_scalar(f'{datasetname}'+'/time_45', time_45, self.current_epoch_set)
        
        self.last_iteration_time=time.time()

        return {
            'loss': loss
        }
    
    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """
    
    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ] + ([
            {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ])
        return {
            'psnr': psnr,
            'index': batch['index']
        }
          
    
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """
    
    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)
            self.export()         

    def test_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ] + ([
            {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ])
        return {
            'psnr': psnr,
            'index': batch['index']
        }      
    
    def test_epoch_end(self, out):
        """
        Synchronize devices.
        Generate image sequence using test outputs.
        """
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True)    

            self.export()
    
    def export(self):
        mesh = self.model.export(self.config.export)
        # if self.config.model.if_gaussian:
        #     tc = torch.tensor(self.scene.center).reshape(3)
        #     pts = mesh['v_pos']
        #     pts = pts * self.scene.scale
        #     pts += tc
        #     mesh['v_pos'] = pts
      
        self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.ply",
            **mesh
        )        

    def prepare_output_and_logger(self,args):   

        if not args.model_path:
            if os.getenv('OAR_JOB_ID'):
                unique_str=os.getenv('OAR_JOB_ID')
            else:
                unique_str = str(uuid.uuid4())
            args.model_path = os.path.join("./output/", unique_str[0:10])
                
        # Set up output folder
        print("Output folder: {}".format(args.model_path))
        os.makedirs(args.model_path, exist_ok = True)
        with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(args))))

        # Create Tensorboard writer
        tb_writer = None
        if TENSORBOARD_FOUND:
            tb_writer = SummaryWriter(args.model_path)
        else:
            print("Tensorboard not available: not logging progress")
        return tb_writer

