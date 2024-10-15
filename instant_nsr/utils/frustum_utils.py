import torch

def world_to_camera(points, extrinsics):
    points_world_homogeneous = torch.cat([points, torch.ones(points.shape[0], 1).to(points.device)], dim=1)
    points_camera_homogeneous = points_world_homogeneous @ extrinsics
    points_camera = points_camera_homogeneous[:, :3] / points_camera_homogeneous[:, 3:]
    
    return points_camera

def camera_to_world(points_camera, extrinsics):
    extrinsics_inv = torch.linalg.inv(extrinsics)
    points_camera_homogeneous = torch.cat([points_camera, torch.ones(points_camera.shape[0], 1).to(points_camera.device)], dim=1)
    points_world_homogeneous = points_camera_homogeneous @ extrinsics_inv
    points_world = points_world_homogeneous[:, :3] / points_world_homogeneous[:, 3:]

    return points_world

def cartesian_to_spherical(xyz):
    '''
        use local position
    '''
    rho = xyz.norm(dim=1, keepdim=True)
    theta = torch.atan2(xyz[:,1:2], xyz[:,:1])
    phi = torch.acos(xyz[:,2:] / rho)
    # check nan
    idx = (rho==0)
    theta[idx] = 0.0
    phi[idx] = 0.0

    return torch.cat([rho, theta, phi], dim=1)

def spherical_to_cartesian(polar, ):
    '''
        use local position
    '''
    x = polar[:,:1] * torch.sin(polar[:,2:]) * torch.cos(polar[:,1:2])
    y = polar[:,:1] * torch.sin(polar[:,2:]) * torch.sin(polar[:,1:2])
    z = polar[:,:1] * torch.cos(polar[:,2:])

    return torch.cat([x, y, z], dim=1)

def voxel_to_frustum(polar, feats, frustum_size):
    '''
        1. xyz 2 polar
        2. quantization in polar
        3. polar 2 xyz
    '''
    polar_quantization = torch.round(polar/frustum_size).int()
    # print(f'polar_quantization: {polar_quantization.shape}')
    polar_quantization = torch.cat([polar_quantization, torch.zeros_like(polar_quantization[:,:1]).to(polar.device).int()], dim=1)
    polar_coords = torch.unique(polar_quantization, dim=0) # [m, 4]
    # print(f'polar_coords: {polar_coords.shape}')
    full_hash = F.sphash(polar_quantization)
    polar_hash = F.sphash(polar_coords)
    idx_query = F.sphashquery(full_hash, polar_hash)
    counts = F.spcount(idx_query.int(), len(polar_coords))
    inpterpolated_feat = F.spvoxelize(feats, idx_query, counts)
    # print(f'counts: {counts.shape}, {counts.max()}, {counts.min()}, {counts.float().mean()}')
    # until now, feature has been inserted into each grid of the 

    return polar_coords, inpterpolated_feat, counts