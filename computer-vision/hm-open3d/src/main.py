import numpy as np
import open3d as o3d


def main() -> None:
    ply_point_cloud = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
    print(pcd)
    print(np.asarray(pcd.points))

    demo_crop_data = o3d.data.DemoCropPointCloud()
    vol = o3d.visualization.read_selection_polygon_volume(
        demo_crop_data.cropped_json_path,
    )
    chair = vol.crop_point_cloud(pcd)

    dists = pcd.compute_point_cloud_distance(chair)
    dists = np.asarray(dists)
    idx = np.where(dists > 0.01)[0]
    pcd_without_chair = pcd.select_by_index(idx)

    axis_aligned_bounding_box = chair.get_axis_aligned_bounding_box()
    axis_aligned_bounding_box.color = (1, 0, 0)

    oriented_bounding_box = chair.get_oriented_bounding_box()
    oriented_bounding_box.color = (0, 1, 0)

    o3d.visualization.draw_geometries(
        [pcd_without_chair, chair, axis_aligned_bounding_box, oriented_bounding_box],
        zoom=0.3412,
        front=[0.4, -0.2, -0.9],
        lookat=[2.6, 2.0, 1.5],
        up=[-0.10, -1.0, 0.2],
    )


if __name__ == "__main__":
    main()
