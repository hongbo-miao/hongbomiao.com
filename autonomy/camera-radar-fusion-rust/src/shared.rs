pub mod camera {
    pub mod types {
        pub mod camera_detection;
    }
    pub mod services {
        pub mod detect_objects_in_camera;
    }
}
pub mod radar {
    pub mod types {
        pub mod radar_detection;
    }
    pub mod services {
        pub mod create_radar_detection;
        pub mod load_radar_data;
    }
}
pub mod fusion {
    pub mod types {
        pub mod fused_track;
    }
    pub mod utils {
        pub mod calculate_distance_to_bounding_box;
        pub mod is_radar_inside_bounding_box;
        pub mod project_radar_to_camera;
    }
    pub mod services {
        pub mod associate_camera_radar_detections;
        pub mod create_fused_track;
        pub mod fuse_camera_radar;
        pub mod visualize_camera_radar_fusion;
    }
}
pub mod constants {
    pub mod colors;
}
