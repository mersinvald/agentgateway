use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use agent_core::prelude::Strng;
use agent_core::strng;
use parking_lot::RwLock;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::http::apikey::model_pattern_matches;

#[derive(Debug, Clone, Deserialize)]
pub struct UpstreamCatalog {
	pub models: Vec<UpstreamModel>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UpstreamModel {
	pub slug: String,
	pub visibility: String,
	pub supported_in_api: bool,
	#[serde(default)]
	pub priority: i64,
	#[serde(flatten)]
	pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone)]
pub struct Catalog {
	models: Vec<UpstreamModel>,
	by_slug: BTreeMap<Strng, usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admission {
	Allowed,
	Unknown,
	Denied,
}

#[derive(Debug, Clone)]
pub struct Snapshot {
	pub catalog: Arc<Catalog>,
	pub etag: Option<String>,
	pub validated_at: Instant,
	pub fresh_until: Instant,
	pub stale_until: Instant,
}

impl Snapshot {
	fn state(&self, now: Instant) -> CacheState {
		if now <= self.fresh_until {
			CacheState::Fresh
		} else if now <= self.stale_until {
			CacheState::Stale
		} else {
			CacheState::Expired
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheState {
	Fresh,
	Stale,
	Expired,
	Missing,
}

/// In-memory catalog snapshots keyed by an opaque caller-provided credential
/// partition. Callers derive the key from sensitive credential context and
/// never expose it in telemetry or responses.
const DEFAULT_MAX_PARTITIONS: usize = 128;
const MAX_CATALOG_BYTES: usize = 1024 * 1024;
const MAX_MODELS: usize = 1_000;
const MAX_SLUG_BYTES: usize = 256;
const MAX_METADATA_BYTES: usize = 128 * 1024;

#[derive(Debug)]
struct Partition {
	snapshot: Option<Snapshot>,
	refreshing: Arc<AtomicBool>,
}

/// A per-partition single-flight permit. It is deliberately not `Clone`: only
/// its holder may publish a refresh result, and dropping it releases waiters.
#[derive(Debug)]
pub struct RefreshPermit {
	partition: Strng,
	refreshing: Arc<AtomicBool>,
}

impl Drop for RefreshPermit {
	fn drop(&mut self) {
		self.refreshing.store(false, Ordering::Release);
	}
}

/// In-memory catalog snapshots keyed by an opaque caller-provided credential
/// partition. The caller must include credential generation, account/residency,
/// and target in this key so replacement credentials get a distinct partition.
#[derive(Debug)]
pub struct Cache {
	partitions: RwLock<HashMap<Strng, Partition>>,
	max_partitions: usize,
}

impl Default for Cache {
	fn default() -> Self {
		Self::with_max_partitions(DEFAULT_MAX_PARTITIONS)
	}
}

impl Cache {
	pub fn with_max_partitions(max_partitions: usize) -> Self {
		assert!(
			max_partitions > 0,
			"catalog cache needs at least one partition"
		);
		Self {
			partitions: RwLock::new(HashMap::new()),
			max_partitions,
		}
	}

	pub fn get(&self, partition: &str, now: Instant) -> (Option<Snapshot>, CacheState) {
		let snapshot = self
			.partitions
			.read()
			.get(partition)
			.and_then(|entry| entry.snapshot.clone());
		let state = snapshot
			.as_ref()
			.map_or(CacheState::Missing, |entry| entry.state(now));
		(snapshot.filter(|_| state != CacheState::Expired), state)
	}

	pub fn update(
		&self,
		partition: Strng,
		catalog: Catalog,
		etag: Option<String>,
		now: Instant,
		refresh: Duration,
		stale: Duration,
	) {
		let mut partitions = self.partitions.write();
		let entry = self.entry_for_update(&mut partitions, partition);
		let validated_at = entry
			.snapshot
			.as_ref()
			.map_or(now, |snapshot| snapshot.validated_at.max(now));
		entry.snapshot = Some(Snapshot {
			catalog: Arc::new(catalog),
			etag,
			validated_at,
			fresh_until: validated_at + refresh,
			stale_until: validated_at + refresh + stale,
		});
	}

	pub fn revalidate(
		&self,
		partition: &str,
		now: Instant,
		refresh: Duration,
		stale: Duration,
	) -> bool {
		let mut partitions = self.partitions.write();
		let Some(snapshot) = partitions
			.get_mut(partition)
			.and_then(|entry| entry.snapshot.as_mut())
		else {
			return false;
		};
		if now < snapshot.validated_at {
			return false;
		}
		snapshot.validated_at = now;
		snapshot.fresh_until = now + refresh;
		snapshot.stale_until = now + refresh + stale;
		true
	}

	/// Attempts to become the sole refresher for `partition`. Cold callers can
	/// wait on their fetch orchestration; stale callers can return their snapshot
	/// immediately when this returns `None`.
	pub fn try_begin_refresh(&self, partition: Strng) -> Option<RefreshPermit> {
		let mut partitions = self.partitions.write();
		let entry = self.entry_for_update(&mut partitions, partition.clone());
		if entry
			.refreshing
			.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
			.is_err()
		{
			return None;
		}
		Some(RefreshPermit {
			partition,
			refreshing: entry.refreshing.clone(),
		})
	}

	/// Publishes only if `permit` still owns this exact partition generation.
	/// This prevents an old asynchronous refresh from replacing a recreated
	/// credential partition.
	pub fn publish(
		&self,
		permit: &RefreshPermit,
		catalog: Catalog,
		etag: Option<String>,
		now: Instant,
		refresh: Duration,
		stale: Duration,
	) -> bool {
		let mut partitions = self.partitions.write();
		let Some(entry) = partitions.get_mut(&permit.partition) else {
			return false;
		};
		if !Arc::ptr_eq(&entry.refreshing, &permit.refreshing)
			|| !permit.refreshing.load(Ordering::Acquire)
			|| entry
				.snapshot
				.as_ref()
				.is_some_and(|snapshot| snapshot.validated_at > now)
		{
			return false;
		}
		entry.snapshot = Some(Snapshot {
			catalog: Arc::new(catalog),
			etag,
			validated_at: now,
			fresh_until: now + refresh,
			stale_until: now + refresh + stale,
		});
		true
	}

	fn entry_for_update<'a>(
		&self,
		partitions: &'a mut HashMap<Strng, Partition>,
		partition: Strng,
	) -> &'a mut Partition {
		if !partitions.contains_key(&partition) && partitions.len() == self.max_partitions {
			// Evict the deterministically oldest stale deadline. A displaced refresh
			// loses its publication race through the permit identity check below.
			if let Some(key) = partitions
				.iter()
				.min_by(|(left_key, left), (right_key, right)| {
					left
						.snapshot
						.as_ref()
						.map(|snapshot| snapshot.stale_until)
						.cmp(&right.snapshot.as_ref().map(|snapshot| snapshot.stale_until))
						.then_with(|| left_key.cmp(right_key))
				})
				.map(|(key, _)| key.clone())
			{
				partitions.remove(&key);
			}
		}
		partitions.entry(partition).or_insert_with(|| Partition {
			snapshot: None,
			refreshing: Arc::new(AtomicBool::new(false)),
		})
	}
}

impl Catalog {
	pub fn parse(bytes: &[u8]) -> Result<Self, serde_json::Error> {
		if bytes.len() > MAX_CATALOG_BYTES {
			return Err(invalid_catalog("response exceeds catalog size limit"));
		}
		let upstream: UpstreamCatalog = serde_json::from_slice(bytes)?;
		if upstream.models.len() > MAX_MODELS {
			return Err(invalid_catalog("response exceeds model count limit"));
		}
		let mut seen = HashSet::new();
		for model in &upstream.models {
			if !valid_slug(&model.slug) {
				return Err(invalid_catalog("model slug is empty or invalid"));
			}
			if model.visibility.is_empty() || model.visibility.len() > MAX_SLUG_BYTES {
				return Err(invalid_catalog("model visibility is empty or invalid"));
			}
			if !seen.insert(model.slug.clone()) {
				return Err(invalid_catalog("duplicate model slug"));
			}
			if serde_json::to_vec(&model.extra)
				.map_err(|_| invalid_catalog("model metadata is invalid"))?
				.len()
				> MAX_METADATA_BYTES
			{
				return Err(invalid_catalog("model metadata exceeds size limit"));
			}
		}
		let mut models: Vec<_> = upstream
			.models
			.into_iter()
			.filter(|model| model.visibility == "list" && model.supported_in_api)
			.collect();
		models.sort_by(|a, b| {
			a.priority
				.cmp(&b.priority)
				.then_with(|| a.slug.cmp(&b.slug))
		});
		let by_slug = models
			.iter()
			.enumerate()
			.map(|(index, model)| (strng::new(&model.slug), index))
			.collect();
		Ok(Self { models, by_slug })
	}

	pub fn admit(&self, model: &str, allow: &[Strng], deny: &[Strng]) -> Admission {
		if !self.by_slug.contains_key(model) {
			return Admission::Unknown;
		}
		if deny
			.iter()
			.any(|pattern| model_pattern_matches(pattern, model))
		{
			return Admission::Denied;
		}
		if allow
			.iter()
			.any(|pattern| model_pattern_matches(pattern, model))
		{
			Admission::Allowed
		} else {
			Admission::Denied
		}
	}

	pub fn openai_response(&self, allow: &[Strng], deny: &[Strng]) -> Vec<u8> {
		self.openai_response_for(allow, deny, None)
	}

	pub fn openai_response_for(
		&self,
		allow: &[Strng],
		deny: &[Strng],
		policy: Option<&crate::http::apikey::ModelAccessPolicy>,
	) -> Vec<u8> {
		let data: Vec<Value> = self
			.models
			.iter()
			.filter(|model| self.admit(&model.slug, allow, deny) == Admission::Allowed)
			.filter(|model| {
				policy.is_none_or(|policy| {
					policy.allows(&agent_llm::codex_subscription::public_model(&model.slug))
				})
			})
			.map(|model| {
				let mut record = Map::new();
				record.insert(
					"id".into(),
					Value::String(agent_llm::codex_subscription::public_model(&model.slug)),
				);
				record.insert("object".into(), Value::String("model".into()));
				record.insert("created".into(), Value::from(0));
				record.insert("owned_by".into(), Value::String("openai".into()));
				record.insert("visibility".into(), Value::String(model.visibility.clone()));
				record.insert(
					"supported_in_api".into(),
					Value::Bool(model.supported_in_api),
				);
				record.insert("priority".into(), Value::from(model.priority));
				for (key, value) in &model.extra {
					if !matches!(
						key.as_str(),
						"id"
							| "object"
							| "created"
							| "owned_by"
							| "slug"
							| "visibility"
							| "supported_in_api"
							| "priority"
					) {
						record.insert(key.clone(), value.clone());
					}
				}
				Value::Object(record)
			})
			.collect();
		serde_json::to_vec(&json!({ "object": "list", "data": data }))
			.expect("OpenAI model response is serializable")
	}
}

fn valid_slug(slug: &str) -> bool {
	!slug.is_empty()
		&& slug.len() <= MAX_SLUG_BYTES
		&& slug
			.bytes()
			.all(|byte| !byte.is_ascii_control() && !byte.is_ascii_whitespace())
}

fn invalid_catalog(message: &'static str) -> serde_json::Error {
	serde_json::Error::io(std::io::Error::new(
		std::io::ErrorKind::InvalidData,
		message,
	))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn parses_filters_orders_and_renders_upstream_models() {
		let catalog = Catalog::parse(
			br#"{"models":[
				{"slug":"hidden","visibility":"hidden","supported_in_api":true},
				{"slug":"unsupported","visibility":"list","supported_in_api":false},
				{"slug":"gpt-low","visibility":"list","supported_in_api":true,"priority":2,"context_window":100,"capabilities":{"reasoning":true}},
				{"slug":"gpt-high","visibility":"list","supported_in_api":true,"priority":1}
			]}"#,
		)
		.unwrap();
		let allow = vec![strng::new("gpt-*")];
		assert_eq!(catalog.admit("hidden", &allow, &[]), Admission::Unknown);
		assert_eq!(
			catalog.admit("unsupported", &allow, &[]),
			Admission::Unknown
		);
		assert_eq!(catalog.admit("gpt-high", &allow, &[]), Admission::Allowed);
		let response: Value = serde_json::from_slice(&catalog.openai_response(&allow, &[])).unwrap();
		assert_eq!(response["data"][0]["id"], "openai/gpt-high");
		assert_eq!(response["data"][1]["context_window"], 100);
		assert_eq!(response["data"][1]["visibility"], "list");
		assert_eq!(response["data"][1]["supported_in_api"], true);
		assert_eq!(response["data"][1]["capabilities"]["reasoning"], true);
	}

	#[test]
	fn rejects_invalid_records_and_duplicate_slugs_atomically() {
		for payload in [
			br#"{"models":[{"slug":"valid","visibility":"list","supported_in_api":true},{"slug":"","visibility":"list","supported_in_api":true}]}"# as &[u8],
			br#"{"models":[{"slug":"valid","visibility":"list","supported_in_api":true},{"slug":"valid","visibility":"hidden","supported_in_api":false}]}"#,
			br#"{"models":[{"slug":"valid","visibility":"list"}]}"#,
			br#"{"models":[{"slug":"has space","visibility":"list","supported_in_api":true}]}"#,
		] {
			assert!(Catalog::parse(payload).is_err(), "{payload:?}");
		}
	}

	#[test]
	fn rendering_keeps_extensions_but_reserves_openai_identity() {
		let catalog = Catalog::parse(
			br#"{"models":[{"slug":"gpt","visibility":"list","supported_in_api":true,"id":"bad","object":"bad","created":99,"owned_by":"bad","context_window":128}]}"#,
		)
		.unwrap();
		let response: Value =
			serde_json::from_slice(&catalog.openai_response(&[strng::new("*")], &[])).unwrap();
		let model = &response["data"][0];
		assert_eq!(model["id"], "openai/gpt");
		assert_eq!(model["object"], "model");
		assert_eq!(model["created"], 0);
		assert_eq!(model["owned_by"], "openai");
		assert_eq!(model["context_window"], 128);
	}

	#[test]
	fn empty_allow_is_not_an_implicit_allow_all() {
		let catalog =
			Catalog::parse(br#"{"models":[{"slug":"gpt","visibility":"list","supported_in_api":true}]}"#)
				.unwrap();
		assert_eq!(catalog.admit("gpt", &[], &[]), Admission::Denied);
		assert_eq!(
			catalog.admit("gpt", &[strng::new("*")], &[]),
			Admission::Allowed
		);
		assert_eq!(
			catalog.admit("gpt", &[strng::new("*")], &[strng::new("gpt")]),
			Admission::Denied
		);
	}

	#[test]
	fn cache_has_bounded_fresh_and_stale_states() {
		let now = Instant::now();
		let cache = Cache::default();
		let catalog = Catalog::parse(br#"{"models":[]}"#).unwrap();
		cache.update(
			strng::literal!("opaque-partition"),
			catalog,
			Some("etag".into()),
			now,
			Duration::from_secs(5),
			Duration::from_secs(10),
		);
		assert_eq!(cache.get("opaque-partition", now).1, CacheState::Fresh);
		assert_eq!(
			cache
				.get("opaque-partition", now + Duration::from_secs(6))
				.1,
			CacheState::Stale
		);
		assert_eq!(
			cache
				.get("opaque-partition", now + Duration::from_secs(16))
				.1,
			CacheState::Expired
		);
		assert!(
			cache
				.get("opaque-partition", now + Duration::from_secs(16))
				.0
				.is_none()
		);
	}

	#[test]
	fn revalidation_renews_the_existing_snapshot_and_etag() {
		let now = Instant::now();
		let cache = Cache::default();
		cache.update(
			strng::literal!("partition"),
			Catalog::parse(br#"{"models":[]}"#).unwrap(),
			Some("etag-a".into()),
			now,
			Duration::from_secs(5),
			Duration::from_secs(10),
		);
		assert!(cache.revalidate(
			"partition",
			now + Duration::from_secs(6),
			Duration::from_secs(5),
			Duration::from_secs(10),
		));
		let (snapshot, state) = cache.get("partition", now + Duration::from_secs(6));
		assert_eq!(state, CacheState::Fresh);
		assert_eq!(snapshot.unwrap().etag.as_deref(), Some("etag-a"));
	}

	#[test]
	fn older_revalidation_cannot_move_snapshot_deadlines_backwards() {
		let now = Instant::now();
		let cache = Cache::default();
		cache.update(
			strng::literal!("partition"),
			Catalog::parse(br#"{"models":[]}"#).unwrap(),
			None,
			now,
			Duration::from_secs(5),
			Duration::from_secs(10),
		);
		assert!(!cache.revalidate(
			"partition",
			now - Duration::from_secs(1),
			Duration::from_secs(5),
			Duration::from_secs(10),
		));
		assert_eq!(cache.get("partition", now).1, CacheState::Fresh);
	}

	#[test]
	fn refresh_permits_are_single_flight_and_cannot_publish_after_eviction() {
		let now = Instant::now();
		let cache = Cache::with_max_partitions(1);
		let permit = cache
			.try_begin_refresh(strng::literal!("old-credential"))
			.expect("first refresher wins");
		assert!(
			cache
				.try_begin_refresh(strng::literal!("old-credential"))
				.is_none()
		);
		let replacement = cache
			.try_begin_refresh(strng::literal!("new-credential"))
			.expect("new credential has an isolated partition");
		assert!(!cache.publish(
			&permit,
			Catalog::parse(br#"{"models":[]}"#).unwrap(),
			Some("old".into()),
			now,
			Duration::from_secs(5),
			Duration::from_secs(10),
		));
		assert!(cache.publish(
			&replacement,
			Catalog::parse(br#"{"models":[]}"#).unwrap(),
			Some("new".into()),
			now,
			Duration::from_secs(5),
			Duration::from_secs(10),
		));
		assert_eq!(cache.get("old-credential", now).1, CacheState::Missing);
		assert_eq!(
			cache.get("new-credential", now).0.unwrap().etag.as_deref(),
			Some("new")
		);
	}
}
