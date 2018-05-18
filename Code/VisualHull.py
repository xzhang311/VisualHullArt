from copy import deepcopy
import os
from skimage import measure
import numpy as np
import cv2
import scipy
import tensorflow as tf

# zx

def CarveDirX(voxels_voted, imgx):
    imgx = (imgx>250).astype(np.uint8)
    r, c = np.where(imgx==0)
    grids = np.zeros([200, 200, 200])
    grids[r, c, :] = 1
    grids = grids.flatten()
    voxels_voted[:, 3] +=grids
    return voxels_voted

def CarveDirY(voxels_voted, imgy):
    imgy = (imgy>250).astype(np.uint8)
    r, c = np.where(imgy==0)
    grids = np.zeros([200, 200, 200])
    grids[r, :, c] = 1
    grids = grids.flatten()
    voxels_voted[:, 3] += grids
    return voxels_voted

def CarveDirZ(voxels_voted, imgz):
    imgz = (imgz>250).astype(np.uint8)
    r, c = np.where(imgz==0)
    grids = np.zeros([200, 200, 200])
    grids[:, r, c] = 1
    grids = grids.flatten()
    voxels_voted[:, 3] += grids
    return voxels_voted


def voxels_to_mesh(voxels_voted, use_classic=False):
    ''' Convert voxels to surface using marching cubes.

    Args:
        voxels_voted (numpy.array): nx5 array with voxel data in the format [x,y,z,votes,vote2]
        error_pct (float): percentage of maximum votes to be considered inside the
            object. This will be used as iso-value for marching cubes. The higher the
            value, the bigger the resulting mesh.

    Returns:
        verts (numpy.array): n_vx3 array with 3d vertex coordinates [x,y,z]
        faces (numpy.array): n_fx3 array with indices from verts that make
            each triangular face
    '''
    xs, xinds = np.unique(voxels_voted[:, 0], return_inverse=True)
    ys, yinds = np.unique(voxels_voted[:, 1], return_inverse=True)
    zs, zinds = np.unique(voxels_voted[:, 2], return_inverse=True)
    [dx, dy, dz] = [xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]]
    [nx, ny, nz] = [len(xs), len(ys), len(zs)]
    nv = voxels_voted.shape[0]
    votes = np.zeros((nx, ny, nz))
    for i in range(nv):
        votes[xinds[i], yinds[i], zinds[i]] = voxels_voted[i, 3]

    votes = scipy.ndimage.filters.gaussian_filter(votes, 1)


    maxv = np.max(votes)
    iso_value = 3-0.1
    if use_classic:
        verts, faces = measure.marching_cubes_classic(votes, level=iso_value)
        faces = measure.correct_mesh_orientation(votes, verts, faces,
                                                 gradient_direction='ascent')  # face orientations not consistent, so need to be recalculated for classic version
    else:
        verts, faces, _, _ = measure.marching_cubes(votes, level=iso_value)
        faces = np.fliplr(faces)  # need to invert face orientation to point outwards
    v_int = verts.astype(int)
    v_frac = verts - v_int
    verts[:, 0] = xs[v_int[:, 0]] + dx * v_frac[:, 0]
    verts[:, 1] = ys[v_int[:, 1]] + dy * v_frac[:, 1]
    verts[:, 2] = zs[v_int[:, 2]] + dz * v_frac[:, 2]
    return verts, faces


def write_ply(verts, faces, fname):
    ''' Write vertex and face data to a ply mesh file

    Args:
        verts (numpy.array): n_vx3 array with 3d vertex coordinates [x,y,z]
        faces (numpy.array): n_fx3 array with indices from verts that make
            each triangular face
        fname (str): To write to
    '''
    outstr = 'ply \n'
    outstr += 'format ascii 1.0 \n'
    outstr += 'comment a9_3d \n'
    outstr += 'element vertex {}\n'.format(len(verts))
    outstr += 'property float x\n'
    outstr += 'property float y\n'
    outstr += 'property float z\n'
    outstr += 'element face {}\n'.format(len(faces))
    outstr += 'property list uchar int vertex_indices\n'
    outstr += 'end_header\n'
    for vert in verts:
        for v in vert:
            outstr += '{:.10} '.format(v)
        outstr += '\n'
    for face in faces:
        outstr += '{} '.format(len(face))
        for f in face:
            outstr += '{} '.format(f)
        outstr += '\n'
    with open(fname, 'w') as f:
        f.write(outstr)

    return

def main():
    imgx = cv2.imread('/Users/xizhn/Projects/Visualhull/Imgs/l.jpg', 0)
    imgx = np.flip(imgx, axis = 1)
    imgy = cv2.imread('/Users/xizhn/Projects/Visualhull/Imgs/j.jpg', 0)
    imgz = cv2.imread('/Users/xizhn/Projects/Visualhull/Imgs/s.jpg', 0)

    h, w, d = 200, 200, 200
    voxel_counts = h*w*d
    num_voxels = int(np.prod(voxel_counts))

    xs = np.arange(0, h)
    ys = np.arange(0, w)
    zs = np.arange(0, d)

    voxels_x, voxels_y, voxels_z = np.meshgrid(xs, ys, zs)
    voxels_voted = np.zeros((num_voxels, 5))
    voxels_voted[:, 0] = np.reshape(voxels_x, (num_voxels,))
    voxels_voted[:, 1] = np.reshape(voxels_y, (num_voxels,))
    voxels_voted[:, 2] = np.reshape(voxels_z, (num_voxels,))

    voxels_voted = CarveDirX(voxels_voted, imgx)
    voxels_voted = CarveDirY(voxels_voted, imgy)
    voxels_voted = CarveDirZ(voxels_voted, imgz)

    verts, faces = voxels_to_mesh(voxels_voted, use_classic=False)

    meshPath = '/Users/xizhn/Desktop/name_mesh.ply'
    write_ply(verts, faces, meshPath)
    zx = 0

if __name__ == "__main__":
    main()