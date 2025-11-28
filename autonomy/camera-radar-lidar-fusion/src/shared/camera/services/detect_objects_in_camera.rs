use crate::shared::camera::types::camera_detection::CameraDetection;
use anyhow::{Context, Result};
use nalgebra::Vector4;
use opencv::core::{CV_32F, Mat, Rect, Scalar, Size};
use opencv::dnn::blob_from_image;
use opencv::imgproc::{INTER_LINEAR, resize};
use opencv::prelude::{MatTraitConst, MatTraitConstManual};
use ort::session::{Session, SessionOutputs};
use ort::value::Value;
use std::path::Path;
use tracing::debug;

pub struct YoloModel {
    session: Session,
    input_width: i32,
    input_height: i32,
    class_names: Vec<String>,
}

impl YoloModel {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let session = Session::builder()
            .context("Failed to create ONNX session builder")?
            .commit_from_file(model_path.as_ref())
            .context("Failed to load ONNX model")?;

        let class_names = Self::get_coco_class_names();

        Ok(Self {
            session,
            input_width: 640,
            input_height: 640,
            class_names,
        })
    }

    fn get_coco_class_names() -> Vec<String> {
        vec![
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }
}

pub fn detect_objects_in_camera(
    image: &Mat,
    model: &mut YoloModel,
    confidence_threshold: f32,
) -> Result<Vec<CameraDetection>> {
    let original_height = image.rows();
    let original_width = image.cols();

    let input_size = Size::new(model.input_width, model.input_height);
    let scale_w = model.input_width as f32 / original_width as f32;
    let scale_h = model.input_height as f32 / original_height as f32;
    let scale = scale_w.min(scale_h);
    let new_w = ((original_width as f32) * scale).round() as i32;
    let new_h = ((original_height as f32) * scale).round() as i32;

    let mut resized = Mat::default();
    resize(
        image,
        &mut resized,
        Size::new(new_w, new_h),
        0.0,
        0.0,
        INTER_LINEAR,
    )
    .context("Failed to resize image")?;

    let pad_x = (model.input_width - new_w) / 2;
    let pad_y = (model.input_height - new_h) / 2;
    let mut padded = Mat::new_size_with_default(
        input_size,
        image.typ(),
        Scalar::new(114.0, 114.0, 114.0, 0.0),
    )
    .context("Failed to create padded image")?;
    let roi = Rect::new(pad_x, pad_y, new_w, new_h);
    let mut roi_mat = Mat::roi_mut(&mut padded, roi)?;
    resized.copy_to(&mut roi_mat)?;

    let blob = blob_from_image(
        &padded,
        1.0 / 255.0,
        input_size,
        Scalar::default(),
        true,
        false,
        CV_32F,
    )
    .context("Failed to create blob from image")?;

    let blob_data = blob
        .data_typed::<f32>()
        .context("Failed to get blob data")?;

    let shape = vec![
        1,
        3,
        model.input_height as usize,
        model.input_width as usize,
    ];
    let data = blob_data.to_vec();

    let input_value =
        Value::from_array((shape.as_slice(), data)).context("Failed to create input value")?;

    let outputs: SessionOutputs = model
        .session
        .run(ort::inputs![input_value])
        .context("Failed to run ONNX inference")?;

    let output_tensor_data = outputs[0]
        .try_extract_tensor::<f32>()
        .context("Failed to extract output tensor")?;

    let (shape, data) = output_tensor_data;
    let shape_vec: Vec<usize> = shape.iter().map(|&x| x as usize).collect();

    debug!("YOLO output shape: {:?}", shape_vec);
    debug!("YOLO output data length: {}", data.len());

    let mut detections = Vec::new();

    // YOLOv8/v12 output format is typically [batch, num_predictions, num_features]
    // where num_features = 4 (bbox) + num_classes
    // OR [batch, num_features, num_predictions] (transposed)

    if shape_vec.len() < 2 {
        return Ok(detections);
    }

    // Check if output is [1, 84, N] or [1, N, 84] format
    let (num_boxes, num_features) = if shape_vec.len() == 3 {
        if shape_vec[1] > shape_vec[2] {
            // Format: [1, N, features]
            (shape_vec[1], shape_vec[2])
        } else {
            // Format: [1, features, N] - transposed
            (shape_vec[2], shape_vec[1])
        }
    } else {
        return Ok(detections);
    };

    let num_classes = if num_features > 4 {
        num_features - 4
    } else {
        80
    };

    debug!(
        "Detected format: {} boxes, {} features, {} classes",
        num_boxes, num_features, num_classes
    );

    // YOLOv8+ format: [cx, cy, w, h, class1_prob, class2_prob, ...]
    // No objectness score, just class probabilities
    let is_transposed = shape_vec[1] < shape_vec[2];

    for box_idx in 0..num_boxes {
        let mut box_data = Vec::with_capacity(num_features);

        // Extract data for this box
        for feat_idx in 0..num_features {
            let data_idx = if is_transposed {
                feat_idx * num_boxes + box_idx
            } else {
                box_idx * num_features + feat_idx
            };

            if data_idx < data.len() {
                box_data.push(data[data_idx]);
            } else {
                continue;
            }
        }

        if box_data.len() < 4 + num_classes {
            continue;
        }

        // Find max class probability
        // YOLOv8/v12 outputs logits, need to apply sigmoid
        let mut max_class_score = 0.0f32;
        let mut class_id = 0;
        for class_idx in 0..num_classes.min(model.class_names.len()) {
            let logit = box_data[4 + class_idx];
            // Apply sigmoid: 1 / (1 + e^(-x))
            let score = 1.0 / (1.0 + (-logit).exp());
            if score > max_class_score {
                max_class_score = score;
                class_id = class_idx;
            }
        }

        // Filter by confidence
        if max_class_score < confidence_threshold {
            continue;
        }

        let center_x = box_data[0];
        let center_y = box_data[1];
        let width = box_data[2];
        let height = box_data[3];

        let mut x1 = center_x - width / 2.0;
        let mut y1 = center_y - height / 2.0;
        let mut x2 = center_x + width / 2.0;
        let mut y2 = center_y + height / 2.0;

        let pad_x_f = pad_x as f32;
        let pad_y_f = pad_y as f32;
        let inv_scale = 1.0 / scale;

        x1 = ((x1 - pad_x_f) * inv_scale).max(0.0);
        y1 = ((y1 - pad_y_f) * inv_scale).max(0.0);
        x2 = ((x2 - pad_x_f) * inv_scale).min(original_width as f32);
        y2 = ((y2 - pad_y_f) * inv_scale).min(original_height as f32);

        // Basic size/area sanity filters to reduce obvious false positives
        let box_w = (x2 - x1).max(0.0);
        let box_h = (y2 - y1).max(0.0);
        if box_w < 5.0 || box_h < 5.0 {
            continue;
        }
        let area = box_w * box_h;
        let image_area = (original_width as f32) * (original_height as f32);
        if area / image_area > 0.85 {
            continue;
        }

        let class_name = model.class_names[class_id].clone();

        let detection = CameraDetection::new(
            Vector4::new(x1 as f64, y1 as f64, x2 as f64, y2 as f64),
            max_class_score as f64,
            class_id as i32,
            class_name,
        );

        detections.push(detection);
    }

    debug!("Detected {} objects before NMS", detections.len());

    // Apply class-wise NMS (non-maximum suppression)
    let nms_threshold = 0.5;
    let mut detections = apply_classwise_nms(detections, nms_threshold);

    // Cap maximum number of detections for stability
    let max_detections: usize = 100;
    if detections.len() > max_detections {
        detections.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .expect("Failed to compare confidence values")
        });
        detections.truncate(max_detections);
    }

    debug!("Detected {} objects after NMS", detections.len());
    Ok(detections)
}

fn apply_classwise_nms(
    detections: Vec<CameraDetection>,
    iou_threshold: f64,
) -> Vec<CameraDetection> {
    if detections.is_empty() {
        return detections;
    }

    // Group detections by class id
    let mut by_class: std::collections::HashMap<i32, Vec<CameraDetection>> =
        std::collections::HashMap::new();
    for det in detections.into_iter() {
        by_class.entry(det.class_id).or_default().push(det);
    }

    let mut final_keep: Vec<CameraDetection> = Vec::new();

    for (_class_id, mut class_detections) in by_class.into_iter() {
        // Sort by confidence desc per class
        class_detections.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .expect("Failed to compare confidence values")
        });

        let mut suppressed = vec![false; class_detections.len()];
        for i in 0..class_detections.len() {
            if suppressed[i] {
                continue;
            }

            final_keep.push(class_detections[i].clone());
            let bounding_box_a = &class_detections[i].bounding_box;

            for j in (i + 1)..class_detections.len() {
                if suppressed[j] {
                    continue;
                }
                let bounding_box_b = &class_detections[j].bounding_box;

                let x1 = bounding_box_a[0].max(bounding_box_b[0]);
                let y1 = bounding_box_a[1].max(bounding_box_b[1]);
                let x2 = bounding_box_a[2].min(bounding_box_b[2]);
                let y2 = bounding_box_a[3].min(bounding_box_b[3]);

                let intersection_width = (x2 - x1).max(0.0);
                let intersection_height = (y2 - y1).max(0.0);
                let intersection_area = intersection_width * intersection_height;

                let area_a = (bounding_box_a[2] - bounding_box_a[0]).max(0.0)
                    * (bounding_box_a[3] - bounding_box_a[1]).max(0.0);
                let area_b = (bounding_box_b[2] - bounding_box_b[0]).max(0.0)
                    * (bounding_box_b[3] - bounding_box_b[1]).max(0.0);
                let union_area = area_a + area_b - intersection_area;

                let intersection_over_union = if union_area > 0.0 {
                    intersection_area / union_area
                } else {
                    0.0
                };
                if intersection_over_union > iou_threshold {
                    suppressed[j] = true;
                }
            }
        }
    }

    final_keep
}
