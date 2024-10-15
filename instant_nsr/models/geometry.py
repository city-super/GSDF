import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.utilities.rank_zero import rank_zero_info

import instant_nsr.models
from instant_nsr.models.base import BaseModel
from instant_nsr.models.utils import scale_anything, get_activation, cleanup, chunk_batch
from instant_nsr.models.network_utils import get_encoding, get_mlp, get_encoding_with_network
from instant_nsr.utils.misc import get_rank
from instant_nsr.systems.utils import update_module_step
from nerfacc import ContractionType

import numpy as np
import math
import random
from instant_nsr.models import register

def contract_to_unisphere(x, radius, contraction_type):
    if contraction_type == ContractionType.AABB:
        x = scale_anything(x, (-radius, radius), (0, 1))
    elif contraction_type == ContractionType.UN_BOUNDED_SPHERE:
        x = scale_anything(x, (-radius, radius), (0, 1))
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = x.norm(dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
    else:
        raise NotImplementedError
    return x

def properties_to_colors(properties):
    # Convert properties in the range [0, 1] to grayscale RGB values.
    # You can adjust this function as needed.
    colors = (255 * properties).astype(np.uint8)
    return np.stack([colors, colors, colors], axis=-1)
class MarchingCubeHelper(nn.Module):
    def __init__(self, resolution, use_torch=True):
        super().__init__()
        self.resolution = resolution
        self.use_torch = use_torch
        self.points_range = (0, 1)
        if self.use_torch:
            import torchmcubes
            self.mc_func = torchmcubes.marching_cubes
        else:
            import mcubes
            self.mc_func = mcubes.marching_cubes
        self.verts = None

    def grid_vertices(self):
        if self.verts is None:
            x, y, z = torch.linspace(*self.points_range, self.resolution), torch.linspace(*self.points_range, self.resolution), torch.linspace(*self.points_range, self.resolution)
            x, y, z = torch.meshgrid(x, y, z, indexing='ij')
            verts = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1).reshape(-1, 3)
            self.verts = verts
        return self.verts

    def forward(self, level, threshold=0.):
        level = level.float().view(self.resolution, self.resolution, self.resolution)
        if self.use_torch:
            verts, faces = self.mc_func(level.to(get_rank()), threshold)
            verts, faces = verts.cpu(), faces.cpu().long()
        else:
            verts, faces = self.mc_func(-level.numpy(), threshold) # transform to numpy
            verts, faces = torch.from_numpy(verts.astype(np.float32)), torch.from_numpy(faces.astype(np.int64)) # transform back to pytorch
        verts = verts / (self.resolution - 1.)
        return {
            'v_pos': verts,
            't_pos_idx': faces
        }


class BaseImplicitGeometry(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        if self.config.isosurface is not None:
            assert self.config.isosurface.method in ['mc', 'mc-torch']
            if self.config.isosurface.method == 'mc-torch':
                raise NotImplementedError("Please do not use mc-torch. It currently has some scaling issues I haven't fixed yet.")
            self.helper = MarchingCubeHelper(self.config.isosurface.resolution, use_torch=self.config.isosurface.method=='mc-torch')
        self.radius = self.config.radius
        self.contraction_type = None # assigned in system

    def forward_level(self, points):
        raise NotImplementedError

    def isosurface_(self, vmin, vmax):
        def batch_func(x):
            x = torch.stack([
                scale_anything(x[...,0], (0, 1), (vmin[0], vmax[0])),
                scale_anything(x[...,1], (0, 1), (vmin[1], vmax[1])),
                scale_anything(x[...,2], (0, 1), (vmin[2], vmax[2])),
            ], dim=-1).to(self.rank)
            rv = self.forward_level(x).cpu()
            cleanup()
            return rv
    
        level = chunk_batch(batch_func, self.config.isosurface.chunk, True, self.helper.grid_vertices())
        mesh = self.helper(level, threshold=self.config.isosurface.threshold)
        mesh['v_pos'] = torch.stack([
            scale_anything(mesh['v_pos'][...,0], (0, 1), (vmin[0], vmax[0])),
            scale_anything(mesh['v_pos'][...,1], (0, 1), (vmin[1], vmax[1])),
            scale_anything(mesh['v_pos'][...,2], (0, 1), (vmin[2], vmax[2]))
        ], dim=-1)
        return mesh

    @torch.no_grad()

    def isosurface(self):
        if self.config.isosurface is None:
            raise NotImplementedError
        mesh_coarse = self.isosurface_((-self.radius, -self.radius, -self.radius), (self.radius, self.radius, self.radius))
        if mesh_coarse['v_pos'].shape[0]>0:
            vmin, vmax = mesh_coarse['v_pos'].amin(dim=0), mesh_coarse['v_pos'].amax(dim=0)
            vmin_ = (vmin - (vmax - vmin) * 0.1).clamp(-self.radius, self.radius)
            vmax_ = (vmax + (vmax - vmin) * 0.1).clamp(-self.radius, self.radius)
            mesh_fine = self.isosurface_(vmin_, vmax_)
        else:
            mesh_fine=mesh_coarse
        return mesh_fine 


#SDF Field for the frontground
@register('volume-sdf-sg')
class VolumeSDF_gaussian(BaseImplicitGeometry):
    def setup(self):
        self._finite_difference_eps_list=[]
        self.n_output_dims = self.config.feature_dim
        encoding = get_encoding(3, self.config.xyz_encoding_config)
       
        network = get_mlp(encoding.n_output_dims, self.n_output_dims, self.config.mlp_network_config)
        self.encoding, self.network = encoding, network
        self.grad_type = self.config.grad_type
        self.finite_difference_eps = self.config.get('finite_difference_eps', 1e-3)

        self._finite_difference_eps = None
        if self.grad_type == 'finite_difference':
            rank_zero_info(f"Using finite difference to compute gradients with eps={self.finite_difference_eps}")


    def forward(self, points, with_grad=True, with_feature=True, with_laplace=False):
        
        with torch.inference_mode(torch.is_inference_mode_enabled() and not (with_grad and self.grad_type == 'analytic')):
            with torch.set_grad_enabled(self.training or (with_grad and self.grad_type == 'analytic')):
                if with_grad and self.grad_type == 'analytic':
                    if not self.training:
                        points = points.clone() # points may be in inference mode, get a copy to enable grad
                        
                    points.requires_grad_(True)
                    
                
                points_ = points # points in the original scale
                points = contract_to_unisphere(points, self.radius, self.contraction_type) # points normalized to (0, 1)

                out = self.network(self.encoding(points.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims).float()
 
                sdf, feature = out[...,0], out

                if 'sdf_activation' in self.config:
                    sdf = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
                if 'feature_activation' in self.config:
                    feature = get_activation(self.config.feature_activation)(feature)
                if with_grad:
                    if self.grad_type == 'analytic':
                        grad = torch.autograd.grad(
                            sdf, points_, grad_outputs=torch.ones_like(sdf),
                            create_graph=True, retain_graph=True, only_inputs=True
                        )[0]
                    elif self.grad_type == 'finite_difference':
                        eps = self._finite_difference_eps
                        # eps = random.choice(self._finite_difference_eps_list)* random.uniform(0.1, 3)

                        offsets = torch.as_tensor(
                            [
                                [eps, 0.0, 0.0],
                                [-eps, 0.0, 0.0],
                                [0.0, eps, 0.0],
                                [0.0, -eps, 0.0],
                                [0.0, 0.0, eps],
                                [0.0, 0.0, -eps],
                            ]
                        ).to(points_)
                        points_d_ = (points_[...,None,:] + offsets).clamp(-self.radius, self.radius)
                        
                        points_d = scale_anything(points_d_, (-self.radius, self.radius), (0, 1))
                        
                        points_d_sdf = self.network(self.encoding(points_d.view(-1, 3)))[...,0].view(*points.shape[:-1], 6).float()
                        grad = 0.5 * (points_d_sdf[..., 0::2] - points_d_sdf[..., 1::2]) / eps  

                        if with_laplace:
                            laplace = (points_d_sdf[..., 0::2] + points_d_sdf[..., 1::2] - 2 * sdf[..., None]).sum(-1) / (eps ** 2)
                

        rv = [sdf]
        if with_grad:
            rv.append(grad)
        if with_feature:
            rv.append(feature)
        if with_laplace:
            assert self.config.grad_type == 'finite_difference', "Laplace computation is only supported with grad_type='finite_difference'"
            rv.append(laplace)
      

        rv = [v if self.training else v.detach() for v in rv]
        return rv[0] if len(rv) == 1 else rv


    def get_sdf_and_gradient(self, points):
        #do it with autograd
        with torch.set_grad_enabled(True):
            points.requires_grad_(True)
            infor = self.forward(points, with_grad=True, with_feature=False)
        return infor[1]

    # Calculate the curvature, inspired by the PermutoSDF.
    def get_sdf_and_curvature_1d_precomputed_gradient_normal_based(self, points,normals):
        #get the curvature along a certain random direction for each point
        #does it by computing the normal at a shifted point on the tangent plant and then computing a dot produt
        
        # The perturbation epsilon is smaller and smaller and we introduced randomness.
        # epsilon = random.choice(self._finite_difference_eps_list)* random.uniform(0.1, 3)
        epsilon = self._finite_difference_eps

        nr_points_original=points.shape[0]

        rand_directions=torch.randn(nr_points_original, 3).to(points.device)
        rand_directions=F.normalize(rand_directions,dim=-1)

        
        tangent=torch.cross(normals, rand_directions, dim=1)

        rand_directions=tangent 

        points_shifted=points.clone()+rand_directions*epsilon
        points_shifted = points_shifted.view(-1,3)
        
        #get the gradient at the shifted point
        sdf_gradients_shift=self.get_sdf_and_gradient(points_shifted) 

        normals_shifted=F.normalize(sdf_gradients_shift,p=2, dim=-1)
        dot=(normals*normals_shifted).sum(dim=-1, keepdim=True)
        
        #the dot would assign low weight importance to normals that are almost the same, and increasing error the more they deviate. So it's something like and L2 loss. But we want a L1 loss so we get the angle, and then we map it to range [0,1]
        angle=torch.acos(torch.clamp(dot, -1.0+1e-6, 1.0-1e-6)) #goes to range 0 when the angle is the same and 2pi when is opposite

        
        curvature=angle/(math.pi) #map to [0,1 range]

        curvature = torch.mean(curvature, dim=1)
        return curvature


    def forward_level(self, points):

        points = contract_to_unisphere(points, self.radius, self.contraction_type) # points normalized to (0, 1)

        sdf = self.network(self.encoding(points.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims)[...,0]
        
        if 'sdf_activation' in self.config:
            sdf = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
        return sdf

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)    
        update_module_step(self.network, epoch, global_step)  
        if self.grad_type == 'finite_difference' or self.config.custom_smoothing:
            if isinstance(self.finite_difference_eps, float):
                self._finite_difference_eps = self.finite_difference_eps
            elif self.finite_difference_eps == 'progressive':
                hg_conf = self.config.xyz_encoding_config
                assert hg_conf.otype == "ProgressiveBandHashGrid", "finite_difference_eps='progressive' only works with ProgressiveBandHashGrid"
                # if global_step>hg_conf.start_step:
                current_level = min(
                    hg_conf.start_level + max(global_step - hg_conf.start_step, 0) // hg_conf.update_steps,
                    hg_conf.n_levels
                )
                grid_res = hg_conf.base_resolution * hg_conf.per_level_scale**(current_level - 1)
                grid_size = 2 * self.config.radius / grid_res
                if grid_size != self._finite_difference_eps:
                    rank_zero_info(f"Update finite_difference_eps to {grid_size}!!!!!")
                    self._finite_difference_eps_list.append(grid_size)
                self._finite_difference_eps = grid_size
            else:
                raise ValueError(f"Unknown finite_difference_eps={self.finite_difference_eps}")

# Density Field for the background
@register('volume-density')
class VolumeDensity(BaseImplicitGeometry):
    def setup(self):
        self.n_input_dims = self.config.get('n_input_dims', 3)
        self.n_output_dims = self.config.feature_dim
        self.encoding_with_network = get_encoding_with_network(self.n_input_dims, self.n_output_dims, self.config.xyz_encoding_config, self.config.mlp_network_config)

    def forward(self, points):
        points = contract_to_unisphere(points, self.radius, self.contraction_type)
        out = self.encoding_with_network(points.view(-1, self.n_input_dims)).view(*points.shape[:-1], self.n_output_dims).float()
        density, feature = out[...,0], out
        if 'density_activation' in self.config:
            density = get_activation(self.config.density_activation)(density + float(self.config.density_bias))
        if 'feature_activation' in self.config:
            feature = get_activation(self.config.feature_activation)(feature)
        return density, feature

    def forward_level(self, points):
        points = contract_to_unisphere(points, self.radius, self.contraction_type)
        density = self.encoding_with_network(points.reshape(-1, self.n_input_dims)).reshape(*points.shape[:-1], self.n_output_dims)[...,0]
        if 'density_activation' in self.config:
            density = get_activation(self.config.density_activation)(density + float(self.config.density_bias))
        return -density      

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding_with_network, epoch, global_step)

        
