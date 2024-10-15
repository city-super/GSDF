import torch
import torch.nn as nn
import torchsparse.nn as spnn
import torchsparse.nn.functional as F
from torchsparse import PointTensor, SparseTensor
from torchsparse.nn.utils import get_kernel_offsets

class BasicConvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            # spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        return out
    
class BasicDeconvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        stride=stride,
                        transposed=True),
            # spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)
    
class ResidualBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            # spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation,
                        stride=1),
            # spnn.BatchNorm(outc),
        )

        if inc == outc and stride == 1:
            self.downsample = nn.Sequential()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1,
                            stride=stride),
                # spnn.BatchNorm(outc),
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out
    

# z: PointTensor
# return: SparseTensor
def initial_voxelize(z, init_res, after_res):
    new_float_coord = torch.cat(
        [(z.C[:, :3] * init_res) / after_res, torch.zeros_like(z.C[:, -1]).to(z.C.device).view(-1, 1)], 1)

    pc_hash = F.sphash(torch.floor(new_float_coord).int())
    sparse_hash = torch.unique(pc_hash)
    idx_query = F.sphashquery(pc_hash, sparse_hash)
    counts = F.spcount(idx_query.int(), len(sparse_hash))

    inserted_coords = F.spvoxelize(torch.floor(new_float_coord), idx_query,
                                   counts)
    inserted_coords = torch.round(inserted_coords).int()
    inserted_feat = F.spvoxelize(z.F, idx_query, counts)

    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)
    new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)
    z.additional_features['idx_query'][1] = idx_query
    z.additional_features['counts'][1] = counts
    z.C = new_float_coord

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: SparseTensor
def point_to_voxel(x, z):
    if z.additional_features is None or z.additional_features.get(
            'idx_query') is None or z.additional_features['idx_query'].get(
                x.s) is None:
        pc_hash = F.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1))
        sparse_hash = F.sphash(x.C)
        idx_query = F.sphashquery(pc_hash, sparse_hash)
        counts = F.spcount(idx_query.int(), x.C.shape[0])
        z.additional_features['idx_query'][x.s] = idx_query
        z.additional_features['counts'][x.s] = counts
    else:
        idx_query = z.additional_features['idx_query'][x.s]
        counts = z.additional_features['counts'][x.s]

    inserted_feat = F.spvoxelize(z.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_feat, x.C, x.s)
    new_tensor.cmaps = x.cmaps
    new_tensor.kmaps = x.kmaps

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: PointTensor
def voxel_to_point(x, z, nearest=False):
    if z.idx_query is None or z.weights is None or z.idx_query.get(
            x.s) is None or z.weights.get(x.s) is None:
        off = get_kernel_offsets(2, x.s, 1, device=z.F.device)
        
        old_hash = F.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1), off)
        pc_hash = F.sphash(x.C.to(z.F.device))
        # import pdb;pdb.set_trace()
        idx_query = F.sphashquery(old_hash, pc_hash)
        if (idx_query.sum(dim=0) == -8).sum() > 0:
            print((idx_query.sum(dim=0) == -8).sum())
            import pdb;pdb.set_trace()
        
        weights = F.calc_ti_weights(z.C, idx_query,
                                    scale=x.s[0]).transpose(0, 1).contiguous()
        
        idx_query = idx_query.transpose(0, 1).contiguous()
        if nearest:
            weights[:, 1:] = 0.
            idx_query[:, 1:] = -1
        new_feat = F.spdevoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features
        new_tensor.idx_query[x.s] = idx_query
        new_tensor.weights[x.s] = weights
        z.idx_query[x.s] = idx_query
        z.weights[x.s] = weights

    else:
        new_feat = F.spdevoxelize(x.F, z.idx_query.get(x.s), z.weights.get(x.s))
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features

    return new_tensor
import numpy as np
from plyfile import PlyData, PlyElement
def save_ply(points,path):
  
    points = points.cpu().numpy()  # Convert tensor to numpy array

    # Ensure that your points array is in the shape (N, 3) for 3D points
    # points = points.reshape(-1, 3)  # Uncomment and modify this line if needed

    # Create a structured array
    structured_array = np.array([(point[0], point[1], point[2]) for point in points],
                                dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    # Create PlyElement object
    ply_element = PlyElement.describe(structured_array, 'vertex')

    # Create PlyData object
    ply_data = PlyData([ply_element], text=True)

    # Write to a PLY file
    ply_data.write(path)
def voxel_to_point_pure(anchor_poisitions,anchor_features, smaple_points,voxel_size, nearest=False):
    # import pdb;pdb.set_trace()
    off = get_kernel_offsets(2, 1, 1, device=smaple_points.device)
    old_hash = F.sphash(
        torch.cat([
            torch.floor(smaple_points[:, :3] / voxel_size).int(),
            torch.zeros_like(smaple_points[:, -1]).to(smaple_points.device).view(-1, 1).int()
        ], 1), off)
    # torch.cat([torch.floor(smaple_points[:, :3] / voxel_size).int(),smaple_points[:, -1].int().view(-1, 1)], 1)

    # save_ply(torch.floor(smaple_points[:, :3] / voxel_size).int(),'./scaled_sampled_points.ply')
    # import pdb;pdb.set_trace()
    try:
        pc_hash = F.sphash(anchor_poisitions.to(smaple_points.device))
    except:
        print("anchor_poisitions shape:", anchor_poisitions.shape)
        print("smaple_points shape:", smaple_points.shape)
        print("anchor_poisitions device:", anchor_poisitions.device)
        print("smaple_points device:", smaple_points.device)
    # import pdb;pdb.set_trace()
    idx_query = F.sphashquery(old_hash, pc_hash)
    # print("neighbor ratio:", (idx_query.sum(dim=0) == -8).sum()/idx_query.shape[1])
    weights = F.calc_ti_weights(smaple_points, idx_query,scale=voxel_size).transpose(0, 1).contiguous()
    # import pdb;pdb.set_trace()
    idx_query = idx_query.transpose(0, 1).contiguous()
    if nearest:
        weights[:, 1:] = 0.
        idx_query[:, 1:] = -1
    new_feat = F.spdevoxelize(anchor_features, idx_query, weights)
    # import pdb;pdb.set_trace()



    return new_feat

def voxel_to_point_simple(anchor_poisitions, smaple_points,voxel_size, nearest=False):
    off = get_kernel_offsets(2, 1, 1, device=smaple_points.device)
    old_hash = F.sphash(
        torch.cat([
            torch.floor(smaple_points[:, :3] / voxel_size).int(),
            torch.zeros_like(smaple_points[:, -1]).to(smaple_points.device).view(-1, 1).int()
        ], 1), off)
    try:
        pc_hash = F.sphash(anchor_poisitions.to(smaple_points.device))
    except:
        print("anchor_poisitions shape:", anchor_poisitions.shape)
        print("smaple_points shape:", smaple_points.shape)
        print("anchor_poisitions device:", anchor_poisitions.device)
        print("smaple_points device:", smaple_points.device)

    idx_query = F.sphashquery(old_hash, pc_hash)

    idx_query = idx_query.transpose(0, 1).contiguous()


    is_empty = (idx_query!=-1).any(dim=-1)






    return is_empty.float()