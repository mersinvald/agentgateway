use agent_core::strng;
use agent_core::strng::Strng;

use crate::apply;

pub const DEFAULT_HOST_STR: &str = "chatgpt.com";
pub const DEFAULT_HOST: Strng = strng::literal!(DEFAULT_HOST_STR);
pub const DEFAULT_BASE_PATH: &str = "/backend-api/codex";
pub const MODELS_PATH: &str = "/backend-api/codex/models";
pub const MODEL_PREFIX: &str = "openai/";
pub const MODEL_PATTERN: &str = "openai/*";

pub fn public_model(slug: &str) -> String {
	format!("{MODEL_PREFIX}{slug}")
}

pub fn upstream_model(model: &str) -> Option<&str> {
	model
		.strip_prefix(MODEL_PREFIX)
		.filter(|slug| !slug.is_empty())
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn public_namespace_is_removed_exactly_once() {
		assert_eq!(public_model("gpt-test"), "openai/gpt-test");
		assert_eq!(upstream_model("openai/gpt-test"), Some("gpt-test"));
		assert_eq!(
			upstream_model("openai/openai/gpt-test"),
			Some("openai/gpt-test")
		);
		assert_eq!(upstream_model("openai/"), None);
		assert_eq!(upstream_model("gpt-test"), None);
		assert_eq!(upstream_model("other/gpt-test"), None);
	}
}

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
