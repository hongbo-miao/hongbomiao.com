use nalgebra::Vector4;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraDetection {
    pub bounding_box: Vector4<f64>,
    pub confidence: f64,
    pub class_id: i32,
    pub class_name: String,
}

impl CameraDetection {
    pub fn new(
        bounding_box: Vector4<f64>,
        confidence: f64,
        class_id: i32,
        class_name: String,
    ) -> Self {
        Self {
            bounding_box,
            confidence,
            class_id,
            class_name,
        }
    }

    pub fn center_x(&self) -> f64 {
        (self.bounding_box[0] + self.bounding_box[2]) / 2.0
    }

    pub fn center_y(&self) -> f64 {
        (self.bounding_box[1] + self.bounding_box[3]) / 2.0
    }
}

impl std::fmt::Display for CameraDetection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CameraDetection(class={}, confidence={:.2}, center=({:.0}, {:.0}))",
            self.class_name,
            self.confidence,
            self.center_x(),
            self.center_y()
        )
    }
}
