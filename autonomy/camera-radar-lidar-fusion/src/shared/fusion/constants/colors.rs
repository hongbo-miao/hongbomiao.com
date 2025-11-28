use opencv::core::Scalar;

pub const COLOR_BLACK_SCALAR: Scalar = Scalar::new(0.0, 0.0, 0.0, 0.0);
pub const COLOR_BLUE_SCALAR: Scalar = Scalar::new(255.0, 0.0, 0.0, 0.0);
pub const COLOR_RED_SCALAR: Scalar = Scalar::new(0.0, 0.0, 255.0, 0.0); // for camera-only
pub const COLOR_YELLOW_SCALAR: Scalar = Scalar::new(0.0, 255.0, 255.0, 0.0); // for camera+radar and camera+lidar fusion
pub const COLOR_GREEN_SCALAR: Scalar = Scalar::new(0.0, 255.0, 0.0, 0.0); // for camera+radar+lidar fusion
