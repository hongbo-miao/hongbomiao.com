pub mod camera {
    pub mod types {
        pub mod camera_detection;
    }
    pub mod services {
        pub mod detect_objects_in_camera;
        pub mod log_camera_to_rerun;
        pub mod visualize_camera_only;
    }
}
pub mod radar {
    pub mod types {
        pub mod radar_detection;
    }
    pub mod services {
        pub mod create_radar_detection;
        pub mod load_radar_data;
        pub mod log_radar_to_rerun;
    }
    pub mod utils {
        pub mod project_radar_to_camera;
    }
}
pub mod lidar {
    pub mod types {
        pub mod lidar_detection;
    }
    pub mod services {
        pub mod create_lidar_detection;
        pub mod load_lidar_data;
        pub mod log_lidar_to_rerun;
    }
    pub mod utils {
        pub mod project_lidar_to_camera;
        pub mod transform_lidar_to_vehicle;
    }
}
pub mod fusion {
    pub mod constants {
        pub mod colors;
    }
    pub mod types {
        pub mod fused_track;
    }
    pub mod utils {
        pub mod calculate_distance_color;
        pub mod calculate_distance_lidar_to_bounding_box;
        pub mod calculate_distance_to_bounding_box;
        pub mod calculate_fused_distance;
        pub mod is_lidar_inside_bounding_box;
        pub mod is_radar_inside_bounding_box;
    }
    pub mod services {
        pub mod associate_camera_lidar_detections;
        pub mod associate_camera_radar_detections;
        pub mod create_fused_track;
        pub mod fuse_camera_lidar;
        pub mod fuse_camera_radar;
        pub mod fuse_camera_radar_lidar;
        pub mod visualize_camera_lidar_fusion;
        pub mod visualize_camera_radar_fusion;
        pub mod visualize_camera_radar_lidar_fusion;
    }
}
pub mod map {
    pub mod types {
        pub mod ego_pose;
    }
    pub mod services {
        pub mod load_ego_pose;
        pub mod log_ego_position_to_rerun;
        pub mod log_ego_trajectory_to_rerun;
        pub mod log_ego_vehicle_to_rerun;
    }
    pub mod utils {
        pub mod derive_latitude_longitude;
        pub mod get_reference_coordinate;
    }
}
pub mod nuscenes {
    pub mod types {
        pub mod nuscenes_calibrated_sensor;
        pub mod nuscenes_log;
        pub mod nuscenes_sample;
        pub mod nuscenes_sample_data;
        pub mod nuscenes_scene;
        pub mod nuscenes_sensor;
    }
}
pub mod occupancy {
    pub mod types {
        pub mod occupancy_grid;
    }
    pub mod services {
        pub mod build_occupancy_grid_from_lidar;
        pub mod clear_distant_voxels;
        pub mod decay_occupancy_probabilities;
        pub mod log_occupancy_grid_to_rerun;
    }
}
pub mod rerun {
    pub mod constants {
        pub mod entity_paths;
    }
    pub mod services {
        pub mod log_rerun_image;
    }
}
