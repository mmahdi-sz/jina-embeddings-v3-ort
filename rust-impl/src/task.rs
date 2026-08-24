use std::fmt;
use std::str::FromStr;
use serde::{Deserialize, Serialize};

/// The five task-specific LoRA adapters supported by jina-embeddings-v3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JinaTask {
    /// Asymmetric query search (adapter index 0).
    #[serde(rename = "retrieval.query")]
    RetrievalQuery,
    /// Asymmetric passage / document indexing (adapter index 1).
    #[serde(rename = "retrieval.passage")]
    RetrievalPassage,
    /// Clustering and semantic separation (adapter index 2).
    #[serde(rename = "separation")]
    Separation,
    /// Downstream classification (adapter index 3).
    #[serde(rename = "classification")]
    Classification,
    /// Symmetric textual similarity and STS (adapter index 4).
    #[serde(rename = "text-matching")]
    TextMatching,
}

impl JinaTask {
    /// All 5 supported LoRA tasks in adapter index order.
    pub const ALL: [JinaTask; 5] = [
        JinaTask::RetrievalQuery,
        JinaTask::RetrievalPassage,
        JinaTask::Separation,
        JinaTask::Classification,
        JinaTask::TextMatching,
    ];

    /// Returns the integer `task_id` expected by the ONNX model input tensor.
    #[inline]
    pub fn task_id(&self) -> i64 {
        match self {
            JinaTask::RetrievalQuery => 0,
            JinaTask::RetrievalPassage => 1,
            JinaTask::Separation => 2,
            JinaTask::Classification => 3,
            JinaTask::TextMatching => 4,
        }
    }

    /// Returns the canonical task name string.
    #[inline]
    pub fn as_str(&self) -> &'static str {
        match self {
            JinaTask::RetrievalQuery => "retrieval.query",
            JinaTask::RetrievalPassage => "retrieval.passage",
            JinaTask::Separation => "separation",
            JinaTask::Classification => "classification",
            JinaTask::TextMatching => "text-matching",
        }
    }

    /// Parses a task from its canonical string representation.
    pub fn from_str_name(s: &str) -> Option<Self> {
        match s {
            "retrieval.query" => Some(JinaTask::RetrievalQuery),
            "retrieval.passage" => Some(JinaTask::RetrievalPassage),
            "separation" => Some(JinaTask::Separation),
            "classification" => Some(JinaTask::Classification),
            "text-matching" => Some(JinaTask::TextMatching),
            _ => None,
        }
    }
}

impl fmt::Display for JinaTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for JinaTask {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_str_name(s).ok_or_else(|| {
            format!(
                "Invalid task '{}'. Expected one of: retrieval.query, retrieval.passage, separation, classification, text-matching",
                s
            )
        })
    }
}
