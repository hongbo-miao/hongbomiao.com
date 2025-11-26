use opencv::core::Scalar;

pub const COLOR_BLACK_SCALAR: Scalar = Scalar::new(0.0, 0.0, 0.0, 0.0);
pub const COLOR_BLUE_SCALAR: Scalar = Scalar::new(255.0, 100.0, 0.0, 0.0);
pub const COLOR_CORAL_RED_SCALAR: Scalar = Scalar::new(101.0, 101.0, 255.0, 0.0); // #ff6565 for camera-only
pub const COLOR_GOLDEN_YELLOW_SCALAR: Scalar = Scalar::new(0.0, 223.0, 255.0, 0.0); // #ffdf00 for camera+radar and camera+lidar fusion
pub const COLOR_MINT_GREEN_SCALAR: Scalar = Scalar::new(158.0, 233.0, 134.0, 0.0); // #86e99e for camera+radar+lidar fusion
