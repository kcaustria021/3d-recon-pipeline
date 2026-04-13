import open3d as o3d
import argparse

def main():
    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument("input_path", type=str, help="The path to the ply file to visualize")

    args = parser.parse_args()
    input_path = args.input_path

    if not input_path.endswith(".ply"):
        print("Please enter a valid ply file")
    else:
        print(f"Reading point cloud data from {input_path}...")
        pcd = o3d.io.read_point_cloud(input_path)
        o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()