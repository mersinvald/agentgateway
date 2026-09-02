use agent_core::strng;
use agent_core::strng::Strng;

use crate::apply;

pub const DEFAULT_HOST_STR: &str = "chatgpt.com";
pub const DEFAULT_HOST: Strng = strng::literal!(DEFAULT_HOST_STR);
pub const DEFAULT_BASE_PATH: &str = "/backend-api/codex";
pub const MODELS_PATH: &str = "/backend-api/codex/models";

#[apply(schema!)]
pub struct Provider {
	#[serde(default = "default_refresh_interval")]
	pub refresh_interval: std::time::Duration,
	#[serde(default = "default_stale_while_revalidate")]
	pub stale_while_revalidate: std::time::Duration,
	#[serde(default = "default_allow_models")]
	pub allow_models: Vec<Strng>,
	#[serde(default)]
	pub deny_models: Vec<Strng>,
}

impl super::Provider for Provider {
	const NAME: Strng = strng::literal!("codexSubscription");
}

fn default_refresh_interval() -> std::time::Duration {
	std::time::Duration::from_secs(5 * 60)
}

fn default_stale_while_revalidate() -> std::time::Duration {
	std::time::Duration::from_secs(60 * 60)
}

fn default_allow_models() -> Vec<Strng> {
	vec![strng::literal!("*")]
}
