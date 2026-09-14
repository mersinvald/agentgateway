//! Timers for adapters that buffer SSE events. Keepalive comments are added only
//! after translation, never between fragments of an upstream SSE frame.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum_core::body::Body;
use bytes::Bytes;
use futures_util::stream;
use http_body::Frame;
use http_body_util::{BodyExt, StreamBody};
use tokio::time::{Instant, sleep_until};

#[derive(Clone)]
pub(crate) struct Progress(Arc<Mutex<ProgressState>>);

struct ProgressState {
	last: Instant,
	timed_out: bool,
}

impl Progress {
	/// The adapter records semantic progress. Raw fragments, SSE comments, and
	/// empty data events must not keep a stalled response alive indefinitely.
	pub(crate) fn record(&self) {
		self.0.lock().unwrap().last = Instant::now();
	}

	pub(crate) fn timed_out(&self) -> bool {
		self.0.lock().unwrap().timed_out
	}
}

/// Bound the gap between upstream progress events independently of
/// downstream keepalives. Dropping this body cancels its timer and upstream;
/// no task is spawned. The caller records progress after it accepts new data.
pub(crate) fn upstream_idle_timeout(body: Body, timeout: Duration) -> (Body, Progress) {
	let progress = Progress(Arc::new(Mutex::new(ProgressState {
		last: Instant::now(),
		timed_out: false,
	})));
	let observed = progress.clone();
	let body = Body::new(StreamBody::new(stream::unfold(Some(body), move |state| {
		let progress = observed.clone();
		async move {
			let mut body = state?;
			let deadline = progress.0.lock().unwrap().last + timeout;
			// A stream of immediately-ready comments must not starve the deadline.
			let frame = tokio::select! {
				biased;
				_ = sleep_until(deadline) => None,
				frame = body.frame() => Some(frame),
			};
			match frame {
				Some(Some(Ok(frame))) => Some((Ok(frame), Some(body))),
				Some(Some(Err(error))) => Some((Err(error), None)),
				Some(None) => None,
				None => {
					progress.0.lock().unwrap().timed_out = true;
					tracing::warn!(
						idle_timeout_secs = timeout.as_secs(),
						"upstream SSE stream idle timeout"
					);
					Some((
						Err(axum_core::Error::new(std::io::Error::new(
							std::io::ErrorKind::TimedOut,
							"upstream SSE stream idle timeout",
						))),
						None,
					))
				},
			}
		}
	})));
	(body, progress)
}

/// Emit comments while translated output is pending. The converter marks its
/// terminal batch so we can release upstream immediately, even without EOF.
pub(crate) fn keepalive(body: Body, interval: Duration, terminal: Arc<AtomicBool>) -> Body {
	Body::new(StreamBody::new(stream::unfold(
		Some((body, Instant::now() + interval)),
		move |state| {
			let terminal = terminal.clone();
			async move {
				let (mut body, deadline) = state?;
				tokio::select! {
					biased;
					frame = body.frame() => {
						let frame = frame?;
						let ended = frame.is_err() || terminal.load(Ordering::Relaxed);
						let next = (!ended).then_some((body, Instant::now() + interval));
						Some((frame, next))
					},
					_ = sleep_until(deadline) => Some((
						Ok(Frame::data(Bytes::from_static(b": keepalive\n\n"))),
						Some((body, Instant::now() + interval)),
					)),
				}
			}
		},
	)))
}
