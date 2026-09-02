//! Runtime-only OAuth state for a Codex subscription credential.
//!
//! Token transport and persistence are deliberately traits: this module owns
//! refresh coordination and never formats, logs, or serializes token values.

use std::fmt;
use std::net::{IpAddr, Ipv4Addr};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use base64::Engine;
use secrecy::{ExposeSecret, SecretString};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use url::Url;

use crate::control::GrpcChannel;
use crate::crypto::digest::sha256;
use crate::database::DatabasePool;
use crate::http::sessionpersistence::Encoder;

pub const DEFAULT_TOKEN_LIFETIME: Duration = Duration::from_secs(60 * 60);
const CODEX_OAUTH_ISSUER: &str = "https://auth.openai.com";
const CODEX_OAUTH_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const DEFAULT_DEVICE_POLL_INTERVAL: Duration = Duration::from_secs(5);
const DEFAULT_DEVICE_AUTHORIZATION_LIFETIME: Duration = Duration::from_secs(15 * 60);

#[derive(Clone)]
pub struct Credential {
	pub access_token: SecretString,
	pub refresh_token: SecretString,
	pub expires_at: SystemTime,
	pub account_id: Option<String>,
	pub residency: Option<String>,
}

impl fmt::Debug for Credential {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("Credential")
			.field("access_token", &"<redacted>")
			.field("refresh_token", &"<redacted>")
			.field("expires_at", &self.expires_at)
			.field("account_id", &self.account_id)
			.field("residency", &self.residency)
			.finish()
	}
}

impl Credential {
	pub fn is_valid_at(&self, now: SystemTime) -> bool {
		self.expires_at > now
	}
}

#[derive(Clone, Debug)]
pub struct TokenResponse {
	pub access_token: SecretString,
	pub refresh_token: Option<SecretString>,
	pub expires_in: Option<Duration>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceAuthorizationRequest {
	pub code_challenge: String,
}

#[derive(Clone, Debug)]
pub struct DeviceAuthorizationResponse {
	pub verification_uri: String,
	pub verification_uri_complete: Option<String>,
	pub user_code: String,
	pub device_code: SecretString,
	pub expires_in: Duration,
	pub interval: Duration,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AuthorizationState {
	Pending {
		verification_uri: String,
		verification_uri_complete: Option<String>,
		user_code: String,
		expires_at: SystemTime,
		poll_after: Duration,
	},
	Authorized,
	Expired,
	Denied,
	Failed,
}

#[derive(Clone, Debug)]
pub enum DevicePoll {
	Pending {
		retry_after: Duration,
	},
	Authorized {
		authorization_code: SecretString,
		pkce_verifier: SecretString,
	},
	Expired,
	Denied,
}

/// Public error categories only. Endpoint response bodies and token values must
/// be reduced to one of these before crossing this module boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum OAuthError {
	#[error("OAuth endpoint is unavailable")]
	Unavailable,
	#[error("OAuth authorization was denied")]
	Denied,
	#[error("OAuth authorization expired")]
	Expired,
	#[error("OAuth credential is unavailable")]
	CredentialUnavailable,
	#[error("OAuth response was invalid")]
	InvalidResponse,
	#[error("OAuth credential storage failed")]
	Storage,
}

#[async_trait]
pub trait TokenEndpoint: Send + Sync {
	async fn begin_device_authorization(
		&self,
		request: DeviceAuthorizationRequest,
	) -> Result<DeviceAuthorizationResponse, OAuthError>;

	async fn poll_device_authorization(
		&self,
		device_code: &SecretString,
		user_code: &str,
	) -> Result<DevicePoll, OAuthError>;

	async fn exchange_authorization_code(
		&self,
		authorization_code: &SecretString,
		pkce_verifier: &SecretString,
	) -> Result<TokenResponse, OAuthError>;

	async fn refresh(&self, refresh_token: &SecretString) -> Result<TokenResponse, OAuthError>;
}

#[async_trait]
pub trait CredentialStore: Send + Sync {
	async fn load(&self) -> Result<Option<Credential>, OAuthError>;
	async fn replace(&self, credential: Credential) -> Result<(), OAuthError>;
}

#[derive(Default)]
pub struct MemoryCredentialStore(Mutex<Option<Credential>>);

#[async_trait]
impl CredentialStore for MemoryCredentialStore {
	async fn load(&self) -> Result<Option<Credential>, OAuthError> {
		Ok(self.0.lock().await.clone())
	}

	async fn replace(&self, credential: Credential) -> Result<(), OAuthError> {
		*self.0.lock().await = Some(credential);
		Ok(())
	}
}

/// Stores the sole Codex OAuth credential as one authenticated encrypted value.
///
/// The database never receives plaintext tokens. The session persistence encoder
/// is accepted only when configured for AES, never its base64-only mode.
pub struct DatabaseCredentialStore {
	pool: DatabasePool,
	encoder: Encoder,
}

/// Persists the shared Kubernetes credential through the authenticated controller
/// API. The controller owns encryption and namespace selection; this client never
/// receives Kubernetes API credentials.
pub struct ControlPlaneCredentialStore {
	client: Mutex<protos::agentgateway::dev::credential::codex_credential_service_client::CodexCredentialServiceClient<GrpcChannel>>,
	generation: Mutex<String>,
}

impl ControlPlaneCredentialStore {
	pub fn new(channel: GrpcChannel) -> Self {
		Self {
			client: Mutex::new(protos::agentgateway::dev::credential::codex_credential_service_client::CodexCredentialServiceClient::new(channel)),
			generation: Mutex::new(String::new()),
		}
	}
}

#[async_trait]
impl CredentialStore for ControlPlaneCredentialStore {
	async fn load(&self) -> Result<Option<Credential>, OAuthError> {
		let response = match self
			.client
			.lock()
			.await
			.load(tonic::Request::new(
				protos::agentgateway::dev::credential::CodexCredentialLoadRequest {},
			))
			.await
		{
			Ok(response) => response.into_inner(),
			Err(status) if status.code() == tonic::Code::NotFound => return Ok(None),
			Err(_) => return Err(OAuthError::Storage),
		};
		*self.generation.lock().await = response.generation;
		if response.credential.is_empty() {
			return Ok(None);
		}
		let persisted: PersistedCredential =
			serde_json::from_slice(&response.credential).map_err(|_| OAuthError::Storage)?;
		let expires_at = SystemTime::UNIX_EPOCH
			.checked_add(Duration::from_millis(
				u64::try_from(persisted.expires_at_unix_millis).map_err(|_| OAuthError::Storage)?,
			))
			.ok_or(OAuthError::Storage)?;
		Ok(Some(Credential {
			access_token: SecretString::from(persisted.access_token),
			refresh_token: SecretString::from(persisted.refresh_token),
			expires_at,
			account_id: persisted.account_id,
			residency: persisted.residency,
		}))
	}

	async fn replace(&self, credential: Credential) -> Result<(), OAuthError> {
		let expires_at_unix_millis = credential
			.expires_at
			.duration_since(SystemTime::UNIX_EPOCH)
			.map_err(|_| OAuthError::Storage)?
			.as_millis()
			.try_into()
			.map_err(|_| OAuthError::Storage)?;
		let record = serde_json::to_vec(&PersistedCredential {
			access_token: credential.access_token.expose_secret().to_string(),
			refresh_token: credential.refresh_token.expose_secret().to_string(),
			expires_at_unix_millis,
			account_id: credential.account_id,
			residency: credential.residency,
		})
		.map_err(|_| OAuthError::Storage)?;
		let expected_generation = self.generation.lock().await.clone();
		let response = self
			.client
			.lock()
			.await
			.replace(tonic::Request::new(
				protos::agentgateway::dev::credential::CodexCredentialReplaceRequest {
					expected_generation,
					credential: record,
				},
			))
			.await
			.map_err(|_| OAuthError::Storage)?
			.into_inner();
		*self.generation.lock().await = response.generation;
		Ok(())
	}
}

#[derive(Serialize, Deserialize)]
struct PersistedCredential {
	access_token: String,
	refresh_token: String,
	expires_at_unix_millis: i64,
	account_id: Option<String>,
	residency: Option<String>,
}

impl DatabaseCredentialStore {
	pub async fn new(pool: DatabasePool, encoder: Encoder) -> Result<Self, OAuthError> {
		if !matches!(encoder, Encoder::Aes(_)) {
			return Err(OAuthError::Storage);
		}
		match &pool {
			DatabasePool::Sqlite(pool) => {
				sqlx::raw_sql(SQLITE_SCHEMA)
					.execute(pool)
					.await
					.map_err(|_| OAuthError::Storage)?;
			},
			DatabasePool::Postgres(pool) => {
				sqlx::raw_sql(POSTGRES_SCHEMA)
					.execute(pool)
					.await
					.map_err(|_| OAuthError::Storage)?;
			},
		}
		Ok(Self { pool, encoder })
	}
}

#[async_trait]
impl CredentialStore for DatabaseCredentialStore {
	async fn load(&self) -> Result<Option<Credential>, OAuthError> {
		let encrypted: Option<String> = match &self.pool {
			DatabasePool::Sqlite(pool) => {
				sqlx::query_scalar::<_, String>(
					"SELECT encrypted_credential FROM agw_codex_oauth_credential WHERE id = 1",
				)
				.fetch_optional(pool)
				.await
			},
			DatabasePool::Postgres(pool) => {
				sqlx::query_scalar::<_, String>(
					"SELECT encrypted_credential FROM agw_codex_oauth_credential WHERE id = 1",
				)
				.fetch_optional(pool)
				.await
			},
		}
		.map_err(|_| OAuthError::Storage)?;
		let Some(encrypted) = encrypted else {
			return Ok(None);
		};
		let plaintext = self
			.encoder
			.decrypt(&encrypted)
			.map_err(|_| OAuthError::Storage)?;
		let persisted: PersistedCredential =
			serde_json::from_slice(&plaintext).map_err(|_| OAuthError::Storage)?;
		let expires_at = SystemTime::UNIX_EPOCH
			.checked_add(Duration::from_millis(
				u64::try_from(persisted.expires_at_unix_millis).map_err(|_| OAuthError::Storage)?,
			))
			.ok_or(OAuthError::Storage)?;
		Ok(Some(Credential {
			access_token: SecretString::from(persisted.access_token),
			refresh_token: SecretString::from(persisted.refresh_token),
			expires_at,
			account_id: persisted.account_id,
			residency: persisted.residency,
		}))
	}

	async fn replace(&self, credential: Credential) -> Result<(), OAuthError> {
		let expires_at_unix_millis = credential
			.expires_at
			.duration_since(SystemTime::UNIX_EPOCH)
			.map_err(|_| OAuthError::Storage)?
			.as_millis()
			.try_into()
			.map_err(|_| OAuthError::Storage)?;
		let plaintext = serde_json::to_string(&PersistedCredential {
			access_token: credential.access_token.expose_secret().to_string(),
			refresh_token: credential.refresh_token.expose_secret().to_string(),
			expires_at_unix_millis,
			account_id: credential.account_id,
			residency: credential.residency,
		})
		.map_err(|_| OAuthError::Storage)?;
		let encrypted = self
			.encoder
			.encrypt(&plaintext)
			.map_err(|_| OAuthError::Storage)?;
		match &self.pool {
			DatabasePool::Sqlite(pool) => {
				sqlx::query(SQLITE_UPSERT)
					.bind(encrypted)
					.execute(pool)
					.await
					.map_err(|_| OAuthError::Storage)?;
			},
			DatabasePool::Postgres(pool) => {
				sqlx::query(POSTGRES_UPSERT)
					.bind(encrypted)
					.execute(pool)
					.await
					.map_err(|_| OAuthError::Storage)?;
			},
		}
		Ok(())
	}
}

const SQLITE_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS agw_codex_oauth_credential (
	id INTEGER PRIMARY KEY CHECK (id = 1),
	encrypted_credential TEXT NOT NULL
);
"#;

const POSTGRES_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS agw_codex_oauth_credential (
	id SMALLINT PRIMARY KEY CHECK (id = 1),
	encrypted_credential TEXT NOT NULL
);
"#;

const SQLITE_UPSERT: &str = r#"
INSERT INTO agw_codex_oauth_credential (id, encrypted_credential)
VALUES (1, ?)
ON CONFLICT(id) DO UPDATE SET encrypted_credential = excluded.encrypted_credential
"#;

const POSTGRES_UPSERT: &str = r#"
INSERT INTO agw_codex_oauth_credential (id, encrypted_credential)
VALUES (1, $1)
ON CONFLICT(id) DO UPDATE SET encrypted_credential = EXCLUDED.encrypted_credential
"#;

pub struct UnavailableTokenEndpoint;

#[async_trait]
impl TokenEndpoint for UnavailableTokenEndpoint {
	async fn begin_device_authorization(
		&self,
		_request: DeviceAuthorizationRequest,
	) -> Result<DeviceAuthorizationResponse, OAuthError> {
		Err(OAuthError::Unavailable)
	}

	async fn poll_device_authorization(
		&self,
		_device_code: &SecretString,
		_user_code: &str,
	) -> Result<DevicePoll, OAuthError> {
		Err(OAuthError::Unavailable)
	}

	async fn exchange_authorization_code(
		&self,
		_authorization_code: &SecretString,
		_pkce_verifier: &SecretString,
	) -> Result<TokenResponse, OAuthError> {
		Err(OAuthError::Unavailable)
	}

	async fn refresh(&self, _refresh_token: &SecretString) -> Result<TokenResponse, OAuthError> {
		Err(OAuthError::Unavailable)
	}
}

/// HTTP transport for the Codex subscription OAuth device flow.
///
/// The issuer is intentionally fixed in production. Redirects are disabled so
/// a compromised or misconfigured endpoint cannot redirect credentials off the
/// OpenAI authorization host.
pub struct HttpTokenEndpoint {
	client: reqwest::Client,
	issuer: Url,
}

impl HttpTokenEndpoint {
	pub fn new() -> Self {
		Self {
			client: reqwest::Client::builder()
				.redirect(reqwest::redirect::Policy::none())
				.local_address(Some(IpAddr::V4(Ipv4Addr::UNSPECIFIED)))
				.build()
				.expect("default reqwest client configuration is valid"),
			issuer: Url::parse(CODEX_OAUTH_ISSUER).expect("constant OAuth issuer is valid"),
		}
	}

	#[cfg(test)]
	fn for_test(issuer: Url) -> Self {
		Self {
			client: reqwest::Client::builder()
				.redirect(reqwest::redirect::Policy::none())
				.build()
				.expect("default reqwest client configuration is valid"),
			issuer,
		}
	}

	fn endpoint(&self, path: &str) -> Result<Url, OAuthError> {
		if self.issuer.scheme() != "https"
			|| self.issuer.host_str() != Some("auth.openai.com")
			|| self.issuer.port().is_some()
		{
			#[cfg(not(test))]
			return Err(OAuthError::Unavailable);
		}

		self.issuer.join(path).map_err(|_| OAuthError::Unavailable)
	}

	async fn post_json<T: DeserializeOwned>(
		&self,
		path: &str,
		body: serde_json::Value,
	) -> Result<(reqwest::StatusCode, Option<T>), OAuthError> {
		let response = self
			.client
			.post(self.endpoint(path)?)
			// OpenAI's device authorization edge currently accepts the same client
			// identifier used by the OpenCode reference implementation.
			.header(reqwest::header::USER_AGENT, "opencode/1.0.0")
			.json(&body)
			.send()
			.await
			.map_err(|_| OAuthError::Unavailable)?;
		let status = response.status();
		if !status.is_success() {
			return Ok((status, None));
		}
		let body = response
			.bytes()
			.await
			.map_err(|_| OAuthError::Unavailable)?;
		if body.is_empty() {
			return Ok((status, None));
		}
		let parsed = serde_json::from_slice(&body).map_err(|_| OAuthError::InvalidResponse)?;
		Ok((status, Some(parsed)))
	}

	async fn post_form(&self, body: &[(&str, &str)]) -> Result<TokenResponse, OAuthError> {
		let response = self
			.client
			.post(self.endpoint("/oauth/token")?)
			.form(body)
			.send()
			.await
			.map_err(|_| OAuthError::Unavailable)?;
		if !response.status().is_success() {
			return Err(OAuthError::Unavailable);
		}
		let response: RawTokenResponse = response
			.json()
			.await
			.map_err(|_| OAuthError::InvalidResponse)?;
		response.into_token()
	}
}

impl Default for HttpTokenEndpoint {
	fn default() -> Self {
		Self::new()
	}
}

#[derive(Deserialize)]
struct RawDeviceAuthorizationResponse {
	device_auth_id: SecretString,
	user_code: String,
	#[serde(default)]
	interval: Option<NumberOrString>,
	#[serde(default)]
	expires_in: Option<NumberOrString>,
}

#[derive(Deserialize)]
struct RawDevicePollResponse {
	authorization_code: SecretString,
	code_verifier: SecretString,
}

#[derive(Deserialize)]
struct RawTokenResponse {
	access_token: SecretString,
	#[serde(default)]
	refresh_token: Option<SecretString>,
	#[serde(default)]
	expires_in: Option<NumberOrString>,
}

impl RawTokenResponse {
	fn into_token(self) -> Result<TokenResponse, OAuthError> {
		if self.access_token.expose_secret().is_empty() {
			return Err(OAuthError::InvalidResponse);
		}
		Ok(TokenResponse {
			access_token: self.access_token,
			refresh_token: self.refresh_token,
			expires_in: self.expires_in.map(NumberOrString::duration).transpose()?,
		})
	}
}

#[derive(Deserialize)]
#[serde(untagged)]
enum NumberOrString {
	Number(u64),
	String(String),
}

impl NumberOrString {
	fn duration(self) -> Result<Duration, OAuthError> {
		let seconds = match self {
			Self::Number(seconds) => seconds,
			Self::String(seconds) => seconds.parse().map_err(|_| OAuthError::InvalidResponse)?,
		};
		Ok(Duration::from_secs(seconds))
	}
}

#[async_trait]
impl TokenEndpoint for HttpTokenEndpoint {
	async fn begin_device_authorization(
		&self,
		_request: DeviceAuthorizationRequest,
	) -> Result<DeviceAuthorizationResponse, OAuthError> {
		let (status, response): (_, Option<RawDeviceAuthorizationResponse>) = self
			.post_json(
				"/api/accounts/deviceauth/usercode",
				serde_json::json!({ "client_id": CODEX_OAUTH_CLIENT_ID }),
			)
			.await?;
		if !status.is_success() {
			return Err(OAuthError::Unavailable);
		}
		let response = response.ok_or(OAuthError::InvalidResponse)?;
		if response.device_auth_id.expose_secret().is_empty() || response.user_code.is_empty() {
			return Err(OAuthError::InvalidResponse);
		}
		Ok(DeviceAuthorizationResponse {
			verification_uri: format!("{CODEX_OAUTH_ISSUER}/codex/device"),
			verification_uri_complete: None,
			user_code: response.user_code,
			device_code: response.device_auth_id,
			expires_in: response
				.expires_in
				.map(NumberOrString::duration)
				.transpose()?
				.unwrap_or(DEFAULT_DEVICE_AUTHORIZATION_LIFETIME),
			interval: response
				.interval
				.map(NumberOrString::duration)
				.transpose()?
				.filter(|interval| !interval.is_zero())
				.unwrap_or(DEFAULT_DEVICE_POLL_INTERVAL),
		})
	}

	async fn poll_device_authorization(
		&self,
		device_code: &SecretString,
		user_code: &str,
	) -> Result<DevicePoll, OAuthError> {
		let (status, response): (_, Option<RawDevicePollResponse>) = self
			.post_json(
				"/api/accounts/deviceauth/token",
				serde_json::json!({
					"device_auth_id": device_code.expose_secret(),
					"user_code": user_code,
				}),
			)
			.await?;
		if status.is_success() {
			let response = response.ok_or(OAuthError::InvalidResponse)?;
			if response.authorization_code.expose_secret().is_empty()
				|| response.code_verifier.expose_secret().is_empty()
			{
				return Err(OAuthError::InvalidResponse);
			}
			return Ok(DevicePoll::Authorized {
				authorization_code: response.authorization_code,
				pkce_verifier: response.code_verifier,
			});
		}
		if matches!(
			status,
			reqwest::StatusCode::FORBIDDEN | reqwest::StatusCode::NOT_FOUND
		) {
			return Ok(DevicePoll::Pending {
				retry_after: DEFAULT_DEVICE_POLL_INTERVAL,
			});
		}
		Err(OAuthError::Unavailable)
	}

	async fn exchange_authorization_code(
		&self,
		authorization_code: &SecretString,
		pkce_verifier: &SecretString,
	) -> Result<TokenResponse, OAuthError> {
		self
			.post_form(&[
				("grant_type", "authorization_code"),
				("code", authorization_code.expose_secret()),
				(
					"redirect_uri",
					"https://auth.openai.com/deviceauth/callback",
				),
				("client_id", CODEX_OAUTH_CLIENT_ID),
				("code_verifier", pkce_verifier.expose_secret()),
			])
			.await
	}

	async fn refresh(&self, refresh_token: &SecretString) -> Result<TokenResponse, OAuthError> {
		self
			.post_form(&[
				("grant_type", "refresh_token"),
				("refresh_token", refresh_token.expose_secret()),
				("client_id", CODEX_OAUTH_CLIENT_ID),
			])
			.await
	}
}

struct PendingDeviceAuthorization {
	device_code: SecretString,
	verification_uri: String,
	verification_uri_complete: Option<String>,
	user_code: String,
	expires_at: SystemTime,
	interval: Duration,
}

impl PendingDeviceAuthorization {
	fn state(&self) -> AuthorizationState {
		AuthorizationState::Pending {
			verification_uri: self.verification_uri.clone(),
			verification_uri_complete: self.verification_uri_complete.clone(),
			user_code: self.user_code.clone(),
			expires_at: self.expires_at,
			poll_after: self.interval,
		}
	}
}

#[derive(Default)]
struct RuntimeState {
	credential: Option<Credential>,
	loaded: bool,
	pending: Option<PendingDeviceAuthorization>,
}

/// A process-local coordinator for one durable Codex OAuth credential.
///
/// Holding `state` across endpoint calls is intentional: it is the single-flight
/// barrier for refreshes and device polls. Those operations are rare, while a
/// valid credential returns without network I/O.
pub struct Manager {
	endpoint: Arc<dyn TokenEndpoint>,
	store: Arc<dyn CredentialStore>,
	state: Mutex<RuntimeState>,
}

impl fmt::Debug for Manager {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("CodexOAuthManager").finish_non_exhaustive()
	}
}

impl Manager {
	pub fn new(endpoint: Arc<dyn TokenEndpoint>, store: Arc<dyn CredentialStore>) -> Self {
		Self {
			endpoint,
			store,
			state: Mutex::new(RuntimeState::default()),
		}
	}

	pub fn unavailable() -> Self {
		Self::new(
			Arc::new(HttpTokenEndpoint::new()),
			Arc::new(MemoryCredentialStore::default()),
		)
	}

	/// Returns a valid credential, refreshing it at most once for concurrent callers.
	pub async fn credential(&self) -> Result<Credential, OAuthError> {
		let mut state = self.state.lock().await;
		self.load_if_needed(&mut state).await?;
		let now = SystemTime::now();
		if let Some(credential) = state
			.credential
			.as_ref()
			.filter(|credential| credential.is_valid_at(now))
		{
			return Ok(credential.clone());
		}
		let previous = state
			.credential
			.as_ref()
			.ok_or(OAuthError::CredentialUnavailable)?;
		let token = self.endpoint.refresh(&previous.refresh_token).await?;
		let credential = credential_from_token(token, Some(&previous.refresh_token), now)?;
		self.store.replace(credential.clone()).await?;
		state.credential = Some(credential.clone());
		Ok(credential)
	}

	/// Starts a device session, or advances the outstanding session by one poll.
	/// This is safe to call repeatedly from an OpenAI-compatible control endpoint.
	pub async fn start_or_poll(&self) -> Result<AuthorizationState, OAuthError> {
		let mut state = self.state.lock().await;
		self.load_if_needed(&mut state).await?;
		if state
			.credential
			.as_ref()
			.is_some_and(|credential| credential.is_valid_at(SystemTime::now()))
		{
			return Ok(AuthorizationState::Authorized);
		}

		if state
			.pending
			.as_ref()
			.is_some_and(|pending| pending.expires_at <= SystemTime::now())
		{
			state.pending = None;
			return Ok(AuthorizationState::Expired);
		}

		if state.pending.is_none() {
			let pkce_verifier = generate_pkce_verifier();
			let response = self
				.endpoint
				.begin_device_authorization(DeviceAuthorizationRequest {
					code_challenge: pkce_challenge(&pkce_verifier),
				})
				.await?;
			let expires_at = SystemTime::now()
				.checked_add(response.expires_in)
				.ok_or(OAuthError::InvalidResponse)?;
			let pending = PendingDeviceAuthorization {
				device_code: response.device_code,
				verification_uri: response.verification_uri,
				verification_uri_complete: response.verification_uri_complete,
				user_code: response.user_code,
				expires_at,
				interval: response.interval,
			};
			let result = pending.state();
			state.pending = Some(pending);
			return Ok(result);
		}

		let pending = state.pending.as_ref().expect("pending was checked");
		match self
			.endpoint
			.poll_device_authorization(&pending.device_code, &pending.user_code)
			.await?
		{
			DevicePoll::Pending { retry_after } => {
				let pending = state.pending.as_mut().expect("pending was checked");
				pending.interval = retry_after;
				Ok(pending.state())
			},
			DevicePoll::Authorized {
				authorization_code,
				pkce_verifier,
			} => {
				let token = self
					.endpoint
					.exchange_authorization_code(&authorization_code, &pkce_verifier)
					.await?;
				let fallback_refresh = state
					.credential
					.as_ref()
					.map(|credential| &credential.refresh_token);
				let credential = credential_from_token(token, fallback_refresh, SystemTime::now())?;
				self.store.replace(credential.clone()).await?;
				state.credential = Some(credential);
				state.pending = None;
				Ok(AuthorizationState::Authorized)
			},
			DevicePoll::Expired => {
				state.pending = None;
				Ok(AuthorizationState::Expired)
			},
			DevicePoll::Denied => {
				state.pending = None;
				Ok(AuthorizationState::Denied)
			},
		}
	}

	async fn load_if_needed(&self, state: &mut RuntimeState) -> Result<(), OAuthError> {
		if !state.loaded {
			state.credential = match self.store.load().await {
				Ok(credential) => credential,
				// A first-time authorization may race controller credential-store
				// availability. Permit device authorization to begin; successful
				// completion still requires the durable Replace operation.
				Err(OAuthError::Storage) => None,
				Err(err) => return Err(err),
			};
			state.loaded = true;
		}
		Ok(())
	}
}

pub fn generate_pkce_verifier() -> SecretString {
	let mut bytes = [0u8; 32];
	rand::fill(&mut bytes);
	SecretString::from(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes))
}

pub fn pkce_challenge(verifier: &SecretString) -> String {
	base64::engine::general_purpose::URL_SAFE_NO_PAD
		.encode(sha256(verifier.expose_secret().as_bytes()))
}

fn credential_from_token(
	token: TokenResponse,
	fallback_refresh_token: Option<&SecretString>,
	now: SystemTime,
) -> Result<Credential, OAuthError> {
	let expires_at = now
		.checked_add(token.expires_in.unwrap_or(DEFAULT_TOKEN_LIFETIME))
		.ok_or(OAuthError::InvalidResponse)?;
	let claims = extract_jwt_claims(&token.access_token)?;
	Ok(Credential {
		access_token: token.access_token,
		refresh_token: token
			.refresh_token
			.or_else(|| fallback_refresh_token.cloned())
			.ok_or(OAuthError::InvalidResponse)?,
		expires_at,
		account_id: claims.account_id,
		residency: claims.residency,
	})
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct JwtClaims {
	pub account_id: Option<String>,
	pub residency: Option<String>,
}

#[derive(Deserialize)]
struct RawClaims {
	#[serde(default)]
	account_id: Option<String>,
	#[serde(default)]
	chatgpt_account_id: Option<String>,
	#[serde(default)]
	residency: Option<String>,
	#[serde(rename = "x-openai-residency", default)]
	x_openai_residency: Option<String>,
	#[serde(rename = "https://api.openai.com/auth", default)]
	openai_auth: Option<Box<RawClaims>>,
}

/// Extracts routing claims without validating the JWT signature. Signature
/// validation belongs to the OAuth issuer/token endpoint; this only reads the
/// access token returned over that authenticated channel.
pub fn extract_jwt_claims(token: &SecretString) -> Result<JwtClaims, OAuthError> {
	let payload = token
		.expose_secret()
		.split('.')
		.nth(1)
		.ok_or(OAuthError::InvalidResponse)?;
	let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD
		.decode(payload)
		.map_err(|_| OAuthError::InvalidResponse)?;
	let claims: RawClaims =
		serde_json::from_slice(&payload).map_err(|_| OAuthError::InvalidResponse)?;
	let nested = claims.openai_auth.as_deref();
	Ok(JwtClaims {
		account_id: claims.chatgpt_account_id.or(claims.account_id).or_else(|| {
			nested.and_then(|claims| {
				claims
					.chatgpt_account_id
					.clone()
					.or(claims.account_id.clone())
			})
		}),
		residency: claims.x_openai_residency.or(claims.residency).or_else(|| {
			nested.and_then(|claims| {
				claims
					.x_openai_residency
					.clone()
					.or(claims.residency.clone())
			})
		}),
	})
}

#[cfg(test)]
mod tests {
	use std::sync::atomic::{AtomicUsize, Ordering};

	use wiremock::matchers::{body_json, method, path};
	use wiremock::{Mock, MockServer, ResponseTemplate};

	use super::*;

	#[derive(Default)]
	struct MockEndpoint {
		refreshes: AtomicUsize,
		begins: AtomicUsize,
		polls: AtomicUsize,
	}

	fn jwt() -> SecretString {
		let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(
			br#"{"https://api.openai.com/auth":{"chatgpt_account_id":"account","x-openai-residency":"eu"}}"#,
		);
		SecretString::from(format!("header.{payload}.signature"))
	}

	async fn sqlite_store() -> (DatabasePool, Encoder, DatabaseCredentialStore) {
		let pool = DatabasePool::connect_with_max_connections("sqlite::memory:", Some(1))
			.await
			.expect("sqlite pool");
		let encoder = Encoder::aes("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f")
			.expect("AES encoder");
		let store = DatabaseCredentialStore::new(pool.clone(), encoder.clone())
			.await
			.expect("credential store");
		(pool, encoder, store)
	}

	fn persisted_credential() -> Credential {
		Credential {
			access_token: SecretString::from("access-token-secret"),
			refresh_token: SecretString::from("refresh-token-secret"),
			expires_at: SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000),
			account_id: Some("account".into()),
			residency: Some("eu".into()),
		}
	}

	#[tokio::test]
	async fn database_store_persists_only_encrypted_credentials() {
		let (pool, encoder, store) = sqlite_store().await;
		store.replace(persisted_credential()).await.unwrap();

		let DatabasePool::Sqlite(sqlite) = &pool else {
			unreachable!("sqlite_store returns sqlite")
		};
		let encrypted: String = sqlx::query_scalar(
			"SELECT encrypted_credential FROM agw_codex_oauth_credential WHERE id = 1",
		)
		.fetch_one(sqlite)
		.await
		.unwrap();
		assert!(!encrypted.contains("access-token-secret"));
		assert!(!encrypted.contains("refresh-token-secret"));

		// Reopening the store also verifies schema initialization is idempotent.
		let reopened = DatabaseCredentialStore::new(pool, encoder).await.unwrap();
		let credential = reopened.load().await.unwrap().unwrap();
		assert_eq!(
			credential.access_token.expose_secret(),
			"access-token-secret"
		);
		assert_eq!(
			credential.refresh_token.expose_secret(),
			"refresh-token-secret"
		);
		assert_eq!(credential.account_id.as_deref(), Some("account"));
		assert_eq!(credential.residency.as_deref(), Some("eu"));
	}

	#[tokio::test]
	async fn database_store_rejects_tampered_ciphertext() {
		let (pool, _encoder, store) = sqlite_store().await;
		store.replace(persisted_credential()).await.unwrap();
		let DatabasePool::Sqlite(sqlite) = &pool else {
			unreachable!("sqlite_store returns sqlite")
		};
		sqlx::query("UPDATE agw_codex_oauth_credential SET encrypted_credential = 'tampered'")
			.execute(sqlite)
			.await
			.unwrap();
		assert_eq!(store.load().await.unwrap_err(), OAuthError::Storage);
	}

	#[tokio::test]
	async fn database_store_requires_aes_encoder() {
		let pool = DatabasePool::connect_with_max_connections("sqlite::memory:", Some(1))
			.await
			.unwrap();
		assert!(matches!(
			DatabaseCredentialStore::new(pool, Encoder::base64()).await,
			Err(OAuthError::Storage)
		));
	}

	#[async_trait]
	impl TokenEndpoint for MockEndpoint {
		async fn begin_device_authorization(
			&self,
			request: DeviceAuthorizationRequest,
		) -> Result<DeviceAuthorizationResponse, OAuthError> {
			assert!(!request.code_challenge.is_empty());
			self.begins.fetch_add(1, Ordering::Relaxed);
			Ok(DeviceAuthorizationResponse {
				verification_uri: "https://example.test/device".into(),
				verification_uri_complete: None,
				user_code: "USER-CODE".into(),
				device_code: SecretString::from("device"),
				expires_in: Duration::from_secs(60),
				interval: Duration::from_secs(1),
			})
		}

		async fn poll_device_authorization(
			&self,
			_device_code: &SecretString,
			_user_code: &str,
		) -> Result<DevicePoll, OAuthError> {
			self.polls.fetch_add(1, Ordering::Relaxed);
			Ok(DevicePoll::Authorized {
				authorization_code: SecretString::from("code"),
				pkce_verifier: SecretString::from("server-verifier"),
			})
		}

		async fn exchange_authorization_code(
			&self,
			_authorization_code: &SecretString,
			_pkce_verifier: &SecretString,
		) -> Result<TokenResponse, OAuthError> {
			Ok(TokenResponse {
				access_token: jwt(),
				refresh_token: Some(SecretString::from("refresh")),
				expires_in: None,
			})
		}

		async fn refresh(&self, _refresh_token: &SecretString) -> Result<TokenResponse, OAuthError> {
			self.refreshes.fetch_add(1, Ordering::Relaxed);
			Ok(TokenResponse {
				access_token: jwt(),
				refresh_token: None,
				expires_in: Some(Duration::from_secs(120)),
			})
		}
	}

	#[test]
	fn pkce_is_url_safe_and_deterministic_from_verifier() {
		let verifier = SecretString::from("0123456789012345678901234567890123456789012");
		assert_eq!(
			pkce_challenge(&verifier),
			"_RpfHqw8pAZIomzVUE7sjRmHSM543WVdC4o-Kc4_3C0"
		);
		assert!(generate_pkce_verifier().expose_secret().len() >= 43);
	}

	#[test]
	fn extracts_nested_openai_claims() {
		assert_eq!(
			extract_jwt_claims(&jwt()).unwrap(),
			JwtClaims {
				account_id: Some("account".into()),
				residency: Some("eu".into()),
			}
		);
	}

	#[tokio::test]
	async fn device_flow_replaces_credentials_with_default_expiry() {
		let endpoint = Arc::new(MockEndpoint::default());
		let store = Arc::new(MemoryCredentialStore::default());
		let manager = Manager::new(endpoint.clone(), store.clone());
		assert!(matches!(
			manager.start_or_poll().await.unwrap(),
			AuthorizationState::Pending { .. }
		));
		assert_eq!(
			manager.start_or_poll().await.unwrap(),
			AuthorizationState::Authorized
		);
		let credential = manager.credential().await.unwrap();
		assert_eq!(credential.account_id.as_deref(), Some("account"));
		assert!(
			credential
				.expires_at
				.duration_since(SystemTime::now())
				.unwrap()
				<= DEFAULT_TOKEN_LIFETIME
		);
		assert_eq!(endpoint.begins.load(Ordering::Relaxed), 1);
		assert_eq!(endpoint.polls.load(Ordering::Relaxed), 1);
		assert!(store.load().await.unwrap().is_some());
	}

	#[tokio::test]
	async fn concurrent_expired_credentials_share_one_refresh() {
		let endpoint = Arc::new(MockEndpoint::default());
		let store = Arc::new(MemoryCredentialStore::default());
		store
			.replace(Credential {
				access_token: jwt(),
				refresh_token: SecretString::from("refresh"),
				expires_at: SystemTime::UNIX_EPOCH,
				account_id: None,
				residency: None,
			})
			.await
			.unwrap();
		let manager = Arc::new(Manager::new(endpoint.clone(), store));
		let (left, right) = tokio::join!(manager.credential(), manager.credential());
		assert!(left.is_ok());
		assert!(right.is_ok());
		assert_eq!(endpoint.refreshes.load(Ordering::Relaxed), 1);
	}

	#[tokio::test]
	async fn http_endpoint_uses_codex_device_and_token_protocol() {
		let server = MockServer::start().await;
		Mock::given(method("POST"))
			.and(path("/api/accounts/deviceauth/usercode"))
			.and(body_json(
				serde_json::json!({ "client_id": CODEX_OAUTH_CLIENT_ID }),
			))
			.respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
				"device_auth_id": "device-secret",
				"user_code": "USER-CODE",
				"interval": "3",
				"expires_in": 120
			})))
			.mount(&server)
			.await;
		Mock::given(method("POST"))
			.and(path("/api/accounts/deviceauth/token"))
			.and(body_json(serde_json::json!({
				"device_auth_id": "device-secret",
				"user_code": "USER-CODE",
			})))
			.respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
				"authorization_code": "authorization-secret",
				"code_verifier": "server-verifier"
			})))
			.mount(&server)
			.await;
		Mock::given(method("POST"))
			.and(path("/oauth/token"))
			.respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
				"access_token": "access-secret",
				"refresh_token": "refresh-secret",
				"expires_in": "60"
			})))
			.mount(&server)
			.await;

		let endpoint = HttpTokenEndpoint::for_test(Url::parse(&server.uri()).unwrap());
		let device = endpoint
			.begin_device_authorization(DeviceAuthorizationRequest {
				code_challenge: "challenge".into(),
			})
			.await
			.unwrap();
		assert_eq!(device.user_code, "USER-CODE");
		assert_eq!(device.interval, Duration::from_secs(3));
		assert_eq!(device.expires_in, Duration::from_secs(120));

		let poll = endpoint
			.poll_device_authorization(&device.device_code, &device.user_code)
			.await
			.unwrap();
		let DevicePoll::Authorized {
			authorization_code,
			pkce_verifier,
		} = poll
		else {
			panic!("expected authorization")
		};
		let token = endpoint
			.exchange_authorization_code(&authorization_code, &pkce_verifier)
			.await
			.unwrap();
		assert_eq!(token.expires_in, Some(Duration::from_secs(60)));

		let token = endpoint
			.refresh(&SecretString::from("refresh-secret"))
			.await
			.unwrap();
		assert_eq!(token.expires_in, Some(Duration::from_secs(60)));

		let requests = server.received_requests().await.unwrap();
		let forms: Vec<_> = requests
			.iter()
			.filter(|request| request.url.path() == "/oauth/token")
			.map(|request| String::from_utf8(request.body.clone()).unwrap())
			.collect();
		assert!(forms.iter().any(|form| {
			form.contains("grant_type=authorization_code")
				&& form.contains("code=authorization-secret")
				&& form.contains("code_verifier=server-verifier")
		}));
		assert!(forms.iter().any(|form| {
			form.contains("grant_type=refresh_token") && form.contains("refresh_token=refresh-secret")
		}));
	}

	#[tokio::test]
	async fn http_endpoint_maps_open_code_poll_statuses_to_pending() {
		let server = MockServer::start().await;
		Mock::given(method("POST"))
			.and(path("/api/accounts/deviceauth/token"))
			.respond_with(ResponseTemplate::new(403))
			.mount(&server)
			.await;
		let endpoint = HttpTokenEndpoint::for_test(Url::parse(&server.uri()).unwrap());
		assert!(matches!(
			endpoint
				.poll_device_authorization(&SecretString::from("device-secret"), "USER-CODE")
				.await
				.unwrap(),
			DevicePoll::Pending { retry_after } if retry_after == DEFAULT_DEVICE_POLL_INTERVAL
		));
	}
}
