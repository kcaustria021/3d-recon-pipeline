import open3d as o3d
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument("input_path", type=str, help="Path to npy file containing surface points")

    args = parser.parse_args()
    input_path = args.input_path

    print(f"Reading point cloud data from {input_path}...")

    if input_path.endswith(".npy"):
        pts = np.load(input_path)
        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(pts)

        o3d.visualization.draw_geometries([pcd])
    elif input_path.endswith(".ply"):
        pcd = o3d.io.read_point_cloud(input_path)
        o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()
