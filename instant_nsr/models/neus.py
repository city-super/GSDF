import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import instant_nsr.models as models
from instant_nsr.models.base import BaseModel
from instant_nsr.models.utils import chunk_batch, ray_upsample_hierarchical
from instant_nsr.systems.utils import update_module_step
from nerfacc import ContractionType, OccupancyGrid, ray_marching, render_weight_from_density, render_weight_from_alpha, accumulate_along_rays
from nerfacc.intersection import ray_aabb_intersect
import random
from instant_nsr.models import register



class VarianceNetwork(nn.Module):
    def __init__(self, config):
        super(VarianceNetwork, self).__init__()
        self.config = config
        self.init_val = self.config.init_val
        self.register_parameter('variance', nn.Parameter(torch.tensor(self.config.init_val)))
        self.modulate = self.config.get('modulate', False)
        if self.modulate:
            self.mod_start_steps = self.config.mod_start_steps
            self.reach_max_steps = self.config.reach_max_steps
            self.max_inv_s = self.config.max_inv_s
    
    @property
    # Global kernel size 
    def inv_s(self):
        val = torch.exp(self.variance * 10.0)
        if self.modulate and self.do_mod:
            val = val.clamp_max(self.mod_val)
        return val

    def forward(self, x):
        return torch.ones([len(x), 1], device=self.variance.device) * self.inv_s
    
    def update_step(self, epoch, global_step):
        if self.modulate:
            self.do_mod = global_step > self.mod_start_steps
            if not self.do_mod:
                self.prev_inv_s = self.inv_s.item()
            else:
                self.mod_val = min((global_step / self.reach_max_steps) * (self.max_inv_s - self.prev_inv_s) + self.prev_inv_s, self.max_inv_s)


@register('neus')
class NeuSModel(BaseModel):
    def setup(self):
        self.geometry = models.make(self.config.geometry.name, self.config.geometry)
        self.texture = models.make(self.config.texture.name, self.config.texture)
        self.geometry.contraction_type = ContractionType.AABB

        if self.config.learned_background:
            self.geometry_bg = models.make(self.config.geometry_bg.name, self.config.geometry_bg)
            self.texture_bg = models.make(self.config.texture_bg.name, self.config.texture_bg)
            self.geometry_bg.contraction_type = ContractionType.UN_BOUNDED_SPHERE
            self.near_plane_bg, self.far_plane_bg = 0.1, 1e3
            self.cone_angle_bg = 10**(math.log10(self.far_plane_bg) / self.config.num_samples_per_ray_bg) - 1.
            self.render_step_size_bg = 0.01            

        self.variance = VarianceNetwork(self.config.variance)
        self.register_buffer('scene_aabb', torch.as_tensor([-self.config.radius, -self.config.radius, -self.config.radius, self.config.radius, self.config.radius, self.config.radius], dtype=torch.float32))
        if self.config.grid_prune:
            if not self.config.gs_sampling:
                self.occupancy_grid = OccupancyGrid(
                    roi_aabb=self.scene_aabb,
                    resolution=256,
                    contraction_type=ContractionType.AABB
                )
            if self.config.learned_background:
                self.occupancy_grid_bg = OccupancyGrid(
                    roi_aabb=self.scene_aabb,
                    resolution=128,
                    contraction_type=ContractionType.UN_BOUNDED_SPHERE
                )
        self.randomized = self.config.randomized
        self.background_color = None
        self.render_step_size = 1.732 * 2 * self.config.radius / self.config.num_samples_per_ray
    
    def update_step(self, epoch, global_step):
        update_module_step(self.geometry, epoch, global_step)
        update_module_step(self.texture, epoch, global_step)
        if self.config.learned_background:
            update_module_step(self.geometry_bg, epoch, global_step)
            update_module_step(self.texture_bg, epoch, global_step)
        update_module_step(self.variance, epoch, global_step)

        cos_anneal_end = self.config.get('cos_anneal_end', 0)
        self.cos_anneal_ratio = 1.0 if cos_anneal_end == 0 else min(1.0, global_step / cos_anneal_end)

        def occ_eval_fn(x):

            sdf = self.geometry(x, with_grad=False, with_feature=False)
            inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
            inv_s = inv_s.expand(sdf.shape[0], 1)
            estimated_next_sdf = sdf[...,None] - self.render_step_size * 0.5
            estimated_prev_sdf = sdf[...,None] + self.render_step_size * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            p = prev_cdf - next_cdf
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)
            return alpha

        def occ_eval_fn_bg(x):
            density, _ = self.geometry_bg(x)
            return density[...,None] * self.render_step_size_bg
        
        if self.training and self.config.grid_prune:
            # Sampling guided by predicted depth of rendering branch instead of maintaining an occupancy grid for the frontground
            if not self.config.gs_sampling:
                self.occupancy_grid.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn, occ_thre=self.config.get('grid_prune_occ_thre', 0.01))
            # Maintain an occupancy grid for the background
            if self.config.learned_background:
                self.occupancy_grid_bg.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn_bg, occ_thre=self.config.get('grid_prune_occ_thre_bg', 0.01))
    # surface extraction
    def isosurface(self):
        mesh = self.geometry.isosurface()
        return mesh

    def get_alpha(self, sdf, normal, dirs, dists):
        inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(sdf.shape[0], 1)

        true_cos = (dirs * normal).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio) +
                     F.relu(-true_cos) * self.cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf[...,None] + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf[...,None] - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0.0, 1.0)
        return alpha


    # ray sampling guided by the predicted depth of rendering branch
    def ray_upsampe_hier(self, rays_o, rays_d, gs_depth=None, stratified=True, use_depth_guide=False):

        t_min, t_max = ray_aabb_intersect(rays_o, rays_d, self.scene_aabb)
        
        if gs_depth is not None:

            # Obtain the depth points according to the depthmap
            gs_depth_positions = rays_o[:, None, :] + rays_d[:, None, :] * gs_depth[..., None]
            gs_depth_positions = gs_depth_positions.reshape(-1, 3)

            # Calculate the sdf values in the depth points
            gs_depth_sdf = self.geometry(gs_depth_positions, with_grad=False, with_feature=False)
            gs_depth_probe = gs_depth

            # Coarse and fine intervals 'a' (positive related to the |sdf value|)
            interval = gs_depth_sdf.abs()*self.config.radius*3 
            # interval = gs_depth_sdf.abs()*self.config.radius*10

            interval_fine = gs_depth_sdf.abs()*self.config.radius

            # Set lower bound for sampling intervals.
            interval = interval.clamp_min(self.geometry._finite_difference_eps_list[0]*16)
            interval_fine = interval_fine.clamp_min(self.geometry._finite_difference_eps_list[0]*8)

            low_bound = gs_depth_probe.squeeze(dim=-1) - interval
            upper_bound = gs_depth_probe.squeeze(dim=-1) + interval

            low_bound_fine = gs_depth_probe.squeeze(dim=-1) - interval_fine
            upper_bound_fine = gs_depth_probe.squeeze(dim=-1) + interval_fine
            # Filter out the ray out of scene aabb
            intersected_ray_indices = ((t_max > 0) & (t_max < 1e9) & (gs_depth_probe.squeeze(dim=-1) < t_max) & (gs_depth_probe.squeeze(dim=-1) > t_min)).nonzero(as_tuple=False).view(-1)
        
        if use_depth_guide:
            t_min_fine = torch.stack([low_bound_fine.squeeze(dim=-1), t_min],dim=1).max(dim=1).values
            t_max_fine = torch.stack([upper_bound_fine.squeeze(dim=-1), t_max],dim=1).min(dim=1).values

            t_min = torch.stack([low_bound.squeeze(dim=-1), t_min],dim=1).max(dim=1).values
            t_max = torch.stack([upper_bound.squeeze(dim=-1), t_max],dim=1).min(dim=1).values
            n_equispaced = self.config.num_samples_equispaced
            # n_equispaced = self.config.num_samples_equispaced*10

            n_equispaced_fine = self.config.num_samples_equispaced

        else:
            intersected_ray_indices = ((t_max > 0) & (t_max < 1e9)).nonzero(as_tuple=False).view(-1)
            n_equispaced = self.config.num_samples_full
            n_equispaced_fine=0


        t_min = t_min[intersected_ray_indices][:, None]
        t_max = t_max[intersected_ray_indices][:, None]
        
        rays_o_ = rays_o[intersected_ray_indices]
        rays_d_ = rays_d[intersected_ray_indices]

        
        if stratified:
            rands = torch.rand(n_equispaced, dtype=rays_o.dtype, device=rays_o.device)
        else:
            rands = torch.ones(n_equispaced, dtype=rays_o.dtype, device=rays_o.device) * 0.5
        rands += torch.arange(n_equispaced, dtype=rays_o.dtype, device=rays_o.device)

        dists = rands[None, :] / n_equispaced * (t_max - t_min) + t_min    # [N_rays, N_samples]
        
        
        if use_depth_guide:
            t_min_fine = t_min_fine[intersected_ray_indices][:, None]
            t_max_fine = t_max_fine[intersected_ray_indices][:, None]
            if stratified:
                rands_fine = torch.rand(n_equispaced_fine, dtype=rays_o.dtype, device=rays_o.device)
            else:
                rands_fine = torch.ones(n_equispaced_fine, dtype=rays_o.dtype, device=rays_o.device) * 0.5
            rands_fine += torch.arange(n_equispaced_fine, dtype=rays_o.dtype, device=rays_o.device)
            dists_fine = rands_fine[None, :] / n_equispaced_fine * (t_max_fine - t_min_fine) + t_min_fine    # [N_rays, N_samples]
            dists = torch.cat([dists, dists_fine], dim=-1)
            dists, _ = torch.sort(dists, dim=-1)
            
        # format data
        ray_indices = torch.arange(rays_o.shape[0], dtype=torch.int64, device=rays_o.device)[intersected_ray_indices][:, None]
        ray_indices = ray_indices.expand(-1, n_equispaced+n_equispaced_fine).reshape(-1)
        
        midpoints = dists.reshape(-1, 1)

        positions = rays_o_[:, None, :] + rays_d_[:, None, :] * dists[..., None]
        positions = positions.reshape(-1, 3)

        last_i_dists = 1.732 * 2 * self.config.radius / (n_equispaced+n_equispaced_fine)
        interval_dists = dists[..., 1:] - dists[..., :-1]
        interval_dists = torch.cat([interval_dists,
                                    torch.empty_like(interval_dists[..., :1], dtype=dists.dtype, device=dists.device).fill_(last_i_dists)],
                                   dim=-1).reshape(-1, 1)
        return ray_indices, midpoints, positions, interval_dists, intersected_ray_indices


    def forward_bg_(self, rays):
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)

        def sigma_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            density, _ = self.geometry_bg(positions)
            return density[...,None]            

        _, t_max = ray_aabb_intersect(rays_o, rays_d, self.scene_aabb)
        # if the ray intersects with the bounding box, start from the farther intersection point
        # otherwise start from self.far_plane_bg
        # note that in nerfacc t_max is set to 1e10 if there is no intersection
        near_plane = torch.where(t_max > 1e9, self.near_plane_bg, t_max)
        with torch.no_grad():
            ray_indices, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=None,
                grid=self.occupancy_grid_bg if self.config.grid_prune else None,
                sigma_fn=sigma_fn,
                near_plane=near_plane, far_plane=self.far_plane_bg,
                render_step_size=self.render_step_size_bg,
                stratified=self.randomized,
                cone_angle=self.cone_angle_bg,
                alpha_thre=0.0
            )       
        
        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * midpoints  
        intervals = t_ends - t_starts

        density, feature = self.geometry_bg(positions) 
        rgb = self.texture_bg(feature, t_dirs)

        weights = render_weight_from_density(t_starts, t_ends, density[...,None], ray_indices=ray_indices, n_rays=n_rays)
        opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        comp_rgb = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays)
        comp_rgb = comp_rgb + self.background_color * (1.0 - opacity)       

        out = {
            'comp_rgb': comp_rgb,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device)
        }

        if self.training:
            out.update({
                'weights': weights.view(-1),
                'points': midpoints.view(-1),
                'intervals': intervals.view(-1),
                'ray_indices': ray_indices.view(-1)
            })

        return out
    
    def forward_(self, rays, gs_depth=None, use_depth_guide=False):
        
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
        if self.config.gs_sampling:
            # neus importance sampling
            with torch.no_grad():
                ray_indices, midpoints, positions, dists, intersected_ray_indices = self.ray_upsampe_hier(rays_o=rays_o, rays_d=rays_d, gs_depth=gs_depth, use_depth_guide=use_depth_guide)
           
            t_dirs = rays_d[ray_indices]
        else:
            # nerfacc occ grid sampling
          
            with torch.no_grad():
                ray_indices, t_starts, t_ends = ray_marching(
                    rays_o, rays_d,
                    scene_aabb=self.scene_aabb,
                    grid=self.occupancy_grid if self.config.grid_prune else None,
                    alpha_fn=None,
                    near_plane=None, far_plane=None,
                    render_step_size=self.render_step_size,
                    stratified=self.randomized,
                    cone_angle=0.0,
                    alpha_thre=0.0
                )
        
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            midpoints = (t_starts + t_ends) / 2.
            positions = t_origins + t_dirs * midpoints
            dists = t_ends - t_starts
          


        if self.training:
            if positions.shape[0]==0:
                out={'zero_samples': True}
                return {
                **out
                }

        
        # Calculating sdf, normal, feature ...
        if self.config.geometry.grad_type == 'finite_difference':
            sdf, sdf_grad, feature, sdf_laplace = self.geometry(positions, with_grad=True, with_feature=True, with_laplace=True)
        else:
            sdf, sdf_grad, feature = self.geometry(positions, with_grad=True, with_feature=True,with_laplace=False)
        # normal
        normal = F.normalize(sdf_grad, p=2, dim=-1)
        alpha = self.get_alpha(sdf, normal, t_dirs, dists)[...,None]
        # RGB field
        rgb = self.texture(feature, t_dirs, normal)

        weights = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
        opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        comp_rgb = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays)

        # Predicted normal of reconstruct branch. 
        comp_normal = accumulate_along_rays(weights, ray_indices, values=normal, n_rays=n_rays)
        comp_normal = F.normalize(comp_normal, p=2, dim=-1)
       
        if self.training:
            # curvature of sdf field
            curvature = self.geometry.get_sdf_and_curvature_1d_precomputed_gradient_normal_based(positions,normal)

        out = {
            'comp_rgb': comp_rgb,
            'comp_normal': comp_normal,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(midpoints)], dtype=torch.int32, device=rays.device),
        }

        if self.training:
            out.update({
                'sdf_samples': sdf,
                'sdf_grad_samples': sdf_grad,
                'weights': weights.view(-1),
                'points': midpoints.view(-1),
                'intervals': dists.view(-1),
                'ray_indices': ray_indices.view(-1) ,
                'positions':positions,
                'sample_rgb':rgb,
                'sdf_features':feature,
                'zero_samples': False,
                'intersected_ray_indices': intersected_ray_indices
                           
            })
            

            if not self.config.geometry.grad_type == 'finite_difference':
                out.update({
                    'smoothing': curvature
                })

            if self.config.geometry.grad_type == 'finite_difference':
                out.update({
                    'sdf_laplace_samples': sdf_laplace
                })
           
        # Background
        if self.config.learned_background:
            out_bg = self.forward_bg_(rays)
        else:
            out_bg = {
                'comp_rgb': self.background_color[None,:].expand(*comp_rgb.shape),
                'num_samples': torch.zeros_like(out['num_samples']),
                'rays_valid': torch.zeros_like(out['rays_valid'])
            }

        out_full = {
            'comp_rgb': out['comp_rgb'] + out_bg['comp_rgb'] * (1.0 - out['opacity']),
            'num_samples': out['num_samples'] + out_bg['num_samples'],
            'rays_valid': out['rays_valid'] | out_bg['rays_valid']
        }

        

        return {
            **out,
            **{k + '_bg': v for k, v in out_bg.items()},
            **{k + '_full': v for k, v in out_full.items()}
            
        }

    def forward(self, rays, gs_depth=None, use_depth_guide=False):
        
        if self.training:
            out = self.forward_(rays, gs_depth, use_depth_guide)
        else:
            out = chunk_batch(self.forward_, self.config.ray_chunk, True, rays)
        return {
            **out,
            'inv_s': self.variance.inv_s
        }

    def train(self, mode=True):
        self.randomized = mode and self.config.randomized
        return super().train(mode=mode)
    
    def eval(self):
        self.randomized = False
        return super().eval()
    
    def regularizations(self, out):
        losses = {}
        losses.update(self.geometry.regularizations(out))
        losses.update(self.texture.regularizations(out))
        return losses

    @torch.no_grad()
    def export(self, export_config):
        mesh = self.isosurface()
        if export_config.export_vertex_color:
            _, sdf_grad, feature = chunk_batch(self.geometry, export_config.chunk_size, False, mesh['v_pos'].to(self.rank), with_grad=True, with_feature=True)
            normal = F.normalize(sdf_grad, p=2, dim=-1)
            rgb = self.texture(feature, -normal, normal) # set the viewing directions to the normal to get "albedo"
            mesh['v_rgb'] = rgb.cpu()
        return mesh
