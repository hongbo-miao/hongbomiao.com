use std::collections::HashMap;

use rig::agent::{AgentHook, CompletionCallAction, CompletionCallEvent, HookContext, RequestPatch};
use rig::completion::Document;

use crate::fluss::types::shared_live_sensor_context::SharedLiveSensorContext;
use crate::fluss::utils::render_live_sensor_context_summary::render_live_sensor_context_summary;

/// Injects the latest rolling-window snapshot as extra context on every
/// completion call, including tool continuations, so answers always reflect
/// readings that streamed in after the run started.
pub struct SensorContextHook {
    shared_live_sensor_context: SharedLiveSensorContext,
}

impl SensorContextHook {
    pub fn new(shared_live_sensor_context: SharedLiveSensorContext) -> Self {
        Self {
            shared_live_sensor_context,
        }
    }
}

impl AgentHook for SensorContextHook {
    async fn on_completion_call(
        &self,
        _context: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        let live_sensor_context = self.shared_live_sensor_context.load();
        let summary = render_live_sensor_context_summary(&live_sensor_context);

        CompletionCallAction::patch(RequestPatch::new().context(Document {
            id: "live-sensor-context".to_string(),
            text: summary,
            additional_props: HashMap::new(),
        }))
    }
}
