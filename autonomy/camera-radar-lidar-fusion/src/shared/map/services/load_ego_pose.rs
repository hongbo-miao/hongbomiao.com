use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::shared::map::types::ego_pose::EgoPose;

pub fn load_ego_poses<P: AsRef<Path>>(json_root: P) -> Result<HashMap<String, EgoPose>> {
    let ego_pose_file = json_root.as_ref().join("ego_pose.json");

    if !ego_pose_file.exists() {
        anyhow::bail!("ego_pose.json not found at {}", ego_pose_file.display());
    }

    let bytes = fs::read(&ego_pose_file)
        .with_context(|| format!("Failed to read {}", ego_pose_file.display()))?;

    let poses: Vec<EgoPose> = serde_json::from_slice(&bytes)
        .with_context(|| format!("Failed to parse {}", ego_pose_file.display()))?;

    let pose_map = poses
        .into_iter()
        .map(|pose| (pose.token.clone(), pose))
        .collect();

    Ok(pose_map)
}
