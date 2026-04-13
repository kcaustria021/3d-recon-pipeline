# general imports
import numpy as np
import glob
import argparse
import os

# class imports

# function imports
from vol_sfs import *
import dual_space_sfs

def get_args():
    parser = argparse.ArgumentParser(
            description="An argument parser"
        )

    parser.add_argument(
            "imgs_path",
            type=str,
            help="The path of the images of the object"
        )
    parser.add_argument(
            "masks_path",
            type=str,
            help="The path of the binary masks for the object"
        )
    parser.add_argument(
            "output_file",
            type=str,
            help="The file to write the point clouds to"
        )
    parser.add_argument(
            "projs_file",
            type=str,
            help="The .npz file that contains the projection matrices"
        )
    parser.add_argument(
            "-s",
            action="store_true",
            help="Whether or not a single view is used"
        )
    parser.add_argument(
            "method",
            type=str,
            help="The method for SfS"
        )

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    imgs_path = args.imgs_path
    masks_path = args.masks_path
    output_file = args.output_file
    projs_file = args.projs_file
    single_view = args.s
    sfs_method = args.method

    if imgs_path == "media/bunny_data/images":
        # bunny data has different file extensions
        imgs = sorted(glob.glob(os.path.join(imgs_path, "*.ppm")))
        masks = sorted(glob.glob(os.path.join(masks_path, "*.pgm")))
    else:
        # synthetic data is normal
        imgs = sorted(glob.glob(os.path.join(imgs_path, "*.png")))
        masks = sorted(glob.glob(os.path.join(masks_path, "*.png")))

    assert len(imgs) == len(masks), "Number of images must match number of masks"
    n_views = len(imgs)

    # initialize views
    projs = np.load(projs_file, allow_pickle=True)

    views = []
    for i in range(n_views):
        print(f"Initialize view {i}")

        if single_view:
            best_view = 3 # selected at random
            proj_dict = projs[f"view{best_view:05d}"].item()
            view = dual_space_sfs.View2(imgs[i], masks[i], proj_dict)
        else:
            proj_dict = projs[f"view{i:05d}"].item()
            view = dual_space_sfs.View2(imgs[i], masks[i], proj_dict)

        views.append(view)

    min_bound, max_bound = compute_bounds(views)
    cube_min, cube_max = cubify(min_bound, max_bound)
    x_min, y_min, z_min = cube_min
    x_max, y_max, z_max = cube_max

    if sfs_method == "volumetric":
        # vanilla sfs
        octree_root = dual_space_sfs.Node(bounds=(x_min, x_max, y_min, y_max, z_min, z_max), max_depth=6)
        print("Carving voxels...")
        carve_voxels(octree_root, views)

        octree_to_ply(octree_root, output_file)
    elif sfs_method == "halfspace":
        # dual space
        Fs = get_fundamental_matrices(views)
        print("Recovering surface points...")
        surface_pts = dual_space_sfs.reconstruct2(views, Fs)
        
        pts_to_ply(surface_pts, output_file)
        print(f"Saved {surface_pts.shape[0]} points to {output_file}")

if __name__ == "__main__":
    main()
