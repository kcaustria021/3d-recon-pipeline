import open3d as o3d
import argparse

def main():
    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument("view", type=int, help="Where to view the point cloud from")

    args = parser.parse_args()
    input_path = args.input_path

    print(f"Reading point cloud data from {input_path}...")
    pcd = o3d.io.read_point_cloud(input_path)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()