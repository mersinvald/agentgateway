use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use agent_core::strng;
use axum_core::body::Body;
use rand::RngExt;
use serde_json::{Value, json};

use super::ResponseToolMap;
use crate::parse::sse::SseJsonEvent;
use crate::types::completions::typed as completions;
use crate::{LogContentFields, StreamingUsageGuard, parse, types};

// Buffered reasoning/tools must not trip clients' socket idle timers. Bound the
// gap between upstream progress events to five minutes. Empty data events,
// upstream comments, and downstream keepalives do not reset this timer.
const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(15);
const UPSTREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Default)]
struct ToolCall {
	id: String,
	name: String,
	arguments: String,
}

struct Stream {
	id: String,
	message_id: String,
	model: String,
	created_at: u64,
	sequence: u64,
	created: bool,
	finished: bool,
	saw_token: bool,
	text: String,
	reasoning: String,
	message_output_index: Option<usize>,
	reasoning_output_index: Option<usize>,
	next_output_index: usize,
	reasoning_item_id: String,
	tools: BTreeMap<u32, ToolCall>,
	tool_map: ResponseToolMap,
	stop: Option<completions::FinishReason>,
	usage: Option<completions::Usage>,
	buffered: usize,
	limit: usize,
	log: StreamingUsageGuard,
	log_content: LogContentFields,
}

impl Stream {
	fn event(
		&mut self,
		events: &mut Vec<(&'static str, Value)>,
		kind: &'static str,
		mut data: Value,
	) {
		self.sequence += 1;
		data["type"] = json!(kind);
		data["sequence_number"] = json!(self.sequence);
		events.push((kind, data));
	}

	fn response(&self, status: &str) -> Value {
		let builder = types::responses::ResponseBuilder::new(self.id.clone(), self.model.clone());
		let mut value = serde_json::to_value(builder.response(
			types::responses::typed::Status::InProgress,
			None,
			None,
			None,
		))
		.expect("response is serializable");
		value["status"] = json!(status);
		value["created_at"] = json!(self.created_at);
		value
	}

	fn fail(&mut self, events: &mut Vec<(&'static str, Value)>, message: &str) {
		self.fail_with_code(events, "upstream_protocol_error", message);
	}

	fn fail_with_code(&mut self, events: &mut Vec<(&'static str, Value)>, code: &str, message: &str) {
		if self.finished {
			return;
		}
		self.finished = true;
		let mut response = self.response("failed");
		response["error"] = json!({"code": code, "message": message});
		self.event(events, "response.failed", json!({"response": response}));
	}

	fn output_index(&mut self) -> usize {
		let index = self.next_output_index;
		self.next_output_index += 1;
		index
	}

	fn start_reasoning(&mut self, events: &mut Vec<(&'static str, Value)>) -> usize {
		if let Some(index) = self.reasoning_output_index {
			return index;
		}
		let output_index = self.output_index();
		self.reasoning_output_index = Some(output_index);
		// Open Responses consumers use the summary-part lifecycle to surface
		// reasoning while the model is still thinking. Keep the item open until
		// the upstream finish reason; tool arguments remain buffered separately.
		self.event(
			events,
			"response.output_item.added",
			json!({"output_index": output_index, "item": {
				"type": "reasoning", "id": self.reasoning_item_id, "summary": [], "content": [],
				"status": "in_progress"
			}}),
		);
		self.event(
			events,
			"response.reasoning_summary_part.added",
			json!({"item_id": self.reasoning_item_id, "output_index": output_index,
				"summary_index": 0, "part": {"type": "summary_text", "text": ""}}),
		);
		output_index
	}

	fn start_message(&mut self, events: &mut Vec<(&'static str, Value)>) -> usize {
		if let Some(index) = self.message_output_index {
			return index;
		}
		let output_index = self.output_index();
		self.message_output_index = Some(output_index);
		self.event(events, "response.output_item.added", json!({"output_index": output_index, "item": {
			"type": "message", "id": self.message_id, "role": "assistant", "status": "in_progress", "content": []
		}}));
		self.event(
			events,
			"response.content_part.added",
			json!({
				"item_id": self.message_id, "output_index": output_index, "content_index": 0,
				"part": {"type": "output_text", "text": "", "annotations": []}
			}),
		);
		output_index
	}

	fn chunk(&mut self, chunk: completions::StreamResponse, events: &mut Vec<(&'static str, Value)>) {
		if !self.created {
			self.created = true;
			self.model = chunk.model.clone();
			self.event(
				events,
				"response.created",
				json!({"response": self.response("in_progress")}),
			);
			self.log.update(|r| {
				r.response.provider_model = Some(strng::new(&chunk.model));
				r.response.service_tier = chunk.service_tier.as_deref().map(strng::new);
			});
		}
		if let Some(usage) = chunk.usage {
			self.usage = Some(usage);
		}
		let Some(choice) = chunk.choices.into_iter().next() else {
			return;
		};
		if self.stop.is_some() {
			self.fail(events, "received content after finish_reason");
			return;
		}
		let has_token = choice.delta.content.as_ref().is_some_and(|s| !s.is_empty())
			|| choice
				.delta
				.reasoning_content
				.as_ref()
				.is_some_and(|s| !s.is_empty())
			|| choice
				.delta
				.tool_calls
				.as_ref()
				.is_some_and(|calls| !calls.is_empty());
		if has_token && !self.saw_token {
			self.saw_token = true;
			self
				.log
				.update(|r| r.response.first_token = Some(Instant::now()));
		}
		let content = choice.delta.content.unwrap_or_default();
		self.buffered = self.buffered.saturating_add(content.len());
		let reasoning_delta = choice.delta.reasoning_content.unwrap_or_default();
		if !reasoning_delta.is_empty() {
			self.buffered = self.buffered.saturating_add(reasoning_delta.len());
			self.reasoning.push_str(&reasoning_delta);
		}
		for tool in choice.delta.tool_calls.into_iter().flatten() {
			let entry = self.tools.entry(tool.index).or_default();
			if let Some(id) = tool.id {
				self.buffered += id.len();
				entry.id.push_str(&id);
			}
			if let Some(function) = tool.function {
				if let Some(name) = function.name {
					self.buffered += name.len();
					entry.name.push_str(&name);
				}
				if let Some(args) = function.arguments {
					self.buffered += args.len();
					entry.arguments.push_str(&args);
				}
			}
		}
		if self.buffered > self.limit {
			self.fail(
				events,
				"translated response exceeds the response buffer limit",
			);
			return;
		}
		if !reasoning_delta.is_empty() {
			let output_index = self.start_reasoning(events);
			self.event(
				events,
				"response.reasoning_summary_text.delta",
				json!({"item_id": self.reasoning_item_id, "output_index": output_index,
					"summary_index": 0, "delta": reasoning_delta}),
			);
		}
		if !content.is_empty() {
			let output_index = self.start_message(events);
			self.text.push_str(&content);
			self.event(
				events,
				"response.output_text.delta",
				json!({
					"item_id": self.message_id, "output_index": output_index, "content_index": 0, "delta": content
				}),
			);
		}
		self.stop = choice.finish_reason;
	}

	fn finish(&mut self, events: &mut Vec<(&'static str, Value)>) {
		let Some(stop) = self.stop else {
			self.fail(events, "upstream stream ended without finish_reason");
			return;
		};
		let status = match stop {
			completions::FinishReason::Length => "incomplete",
			completions::FinishReason::ContentFilter => "failed",
			_ => "completed",
		};
		let mut outputs = Vec::new();
		if !self.text.is_empty() {
			let output_index = self
				.message_output_index
				.expect("message output was started with text");
			let part = json!({"type": "output_text", "text": self.text, "annotations": []});
			let item = json!({"type": "message", "id": self.message_id,
				"role": "assistant", "status": if status == "completed" { "completed" } else { "incomplete" }, "content": [part]});
			self.event(
				events,
				"response.output_text.done",
				json!({
					"item_id": self.message_id, "output_index": output_index, "content_index": 0, "text": self.text
				}),
			);
			self.event(
				events,
				"response.content_part.done",
				json!({
					"item_id": self.message_id, "output_index": output_index, "content_index": 0, "part": part
				}),
			);
			self.event(
				events,
				"response.output_item.done",
				json!({"output_index": output_index, "item": item}),
			);
			outputs.push((output_index, item));
		}
		if !self.reasoning.is_empty() {
			let output_index = self
				.reasoning_output_index
				.expect("reasoning output was started with content");
			// `summary` is the streaming Responses representation. Keep the
			// `reasoning_text` content copy for replay into a later Chat
			// Completions request, which is how this compatibility route preserves
			// Kimi's reasoning across tool turns.
			let item = json!({"type": "reasoning", "id": self.reasoning_item_id, "summary": [{"type": "summary_text", "text": self.reasoning}],
				"content": [{"type": "reasoning_text", "text": self.reasoning}],
				"status": if status == "completed" { "completed" } else { "incomplete" }});
			self.event(
				events,
				"response.reasoning_summary_text.done",
				json!({"item_id": self.reasoning_item_id, "output_index": output_index,
					"summary_index": 0, "text": self.reasoning}),
			);
			self.event(
				events,
				"response.reasoning_summary_part.done",
				json!({"item_id": self.reasoning_item_id, "output_index": output_index, "summary_index": 0,
					"part": {"type": "summary_text", "text": self.reasoning}}),
			);
			self.event(
				events,
				"response.output_item.done",
				json!({"output_index": output_index, "item": item}),
			);
			outputs.push((output_index, item));
		}
		// Buffer tool arguments until complete so split JSON escapes, names and parallel calls
		// can be restored without exposing a JSON wrapper as executable free-form input.
		// Validate the entire batch before any executable tool item is emitted.
		let mut validated_tools = Vec::new();
		for (index, tool) in std::mem::take(&mut self.tools) {
			// Codex executes output_item.done tool calls. A truncated/filtered turn
			// must not dispatch a tool, even if its partial arguments happen to parse.
			if status != "completed" {
				continue;
			}
			if tool.id.is_empty() || tool.name.is_empty() {
				self.fail(events, "upstream tool call is missing its id or name");
				return;
			}
			if !tool.arguments.is_empty() && serde_json::from_str::<Value>(&tool.arguments).is_err() {
				self.fail(events, "invalid JSON arguments for a tool call");
				return;
			}
			let mut item = json!({"type": "function_call", "id": tool.id, "call_id": tool.id,
				"name": tool.name, "arguments": tool.arguments, "status": "completed"});
			if self.tool_map.restore_call(&mut item).is_err() {
				self.fail(events, "invalid JSON input for a custom tool call");
				return;
			}
			validated_tools.push((index, tool, item));
		}
		let mut logged_tools = Vec::new();
		for (index, tool, item) in validated_tools {
			let custom = item["type"] == "custom_tool_call";
			let field = if custom { "input" } else { "arguments" };
			let mut added = item.clone();
			added[field] = json!("");
			added["status"] = json!("in_progress");
			let output_index = self.output_index();
			self.event(
				events,
				"response.output_item.added",
				json!({"output_index": output_index, "item": added}),
			);
			let delta_type = if custom {
				"response.custom_tool_call_input.delta"
			} else {
				"response.function_call_arguments.delta"
			};
			let done_type = if custom {
				"response.custom_tool_call_input.done"
			} else {
				"response.function_call_arguments.done"
			};
			self.event(
				events,
				delta_type,
				json!({"item_id": item["id"], "output_index": output_index, "delta": item[field]}),
			);
			let mut done = json!({"item_id": item["id"], "output_index": output_index});
			done[field] = item[field].clone();
			if !custom {
				done["name"] = item["name"].clone();
			}
			self.event(events, done_type, done);
			self.event(
				events,
				"response.output_item.done",
				json!({"output_index": output_index, "item": item}),
			);
			logged_tools.push((
				index,
				Some(tool.id),
				item["name"].as_str().map(str::to_owned),
				item[field].as_str().unwrap_or_default().to_owned(),
			));
			outputs.push((output_index, item));
		}
		outputs.sort_by_key(|(index, _)| *index);
		let output_values = outputs
			.into_iter()
			.map(|(_, item)| item)
			.collect::<Vec<_>>();
		let mut response = self.response(status);
		response["output"] = json!(output_values);
		if status == "incomplete" {
			response["incomplete_details"] = json!({"reason": "max_output_tokens"});
		}
		if status == "failed" {
			response["error"] = json!({"code": "content_filter", "message": "Content filtered"});
		}
		if let Some(u) = &self.usage {
			let output_tokens = super::to_responses::usage_output_tokens(u);
			let cached = u
				.prompt_tokens_details
				.as_ref()
				.and_then(|d| d.cached_tokens);
			let cache_write = u
				.prompt_tokens_details
				.as_ref()
				.and_then(|d| d.cache_write_tokens)
				.or(u.cache_creation_input_tokens);
			let reasoning = u
				.completion_tokens_details
				.as_ref()
				.and_then(|d| d.reasoning_tokens);
			response["usage"] = json!({"input_tokens": u.prompt_tokens, "output_tokens": output_tokens,
				"total_tokens": u.total_tokens, "input_tokens_details": {"cached_tokens": cached.unwrap_or_default()},
				"output_tokens_details": {"reasoning_tokens": reasoning.unwrap_or_default()}});
			if let Some(write) = cache_write {
				response["usage"]["input_tokens_details"]["cache_write_tokens"] = json!(write);
			}
			self.log.update(|r| {
				r.response.input_tokens = Some(u.prompt_tokens as u64);
				r.response.output_tokens = Some(output_tokens as u64);
				r.response.total_tokens = Some(u.total_tokens as u64);
				r.response.cached_input_tokens = cached;
				r.response.cache_creation_input_tokens = cache_write;
				r.response.reasoning_tokens = reasoning;
			});
		}
		let mut tool_parts = self
			.log_content
			.tool_calls
			.then(|| crate::conversion::completions::finalize_streaming_tool_calls(logged_tools))
			.flatten();
		self.log.update(|r| {
			if self.log_content.completion {
				r.response.completion = Some(vec![self.text.clone()]);
			}
			crate::conversion::completions::build_output_messages(
				&mut r.response,
				tool_parts.take(),
				Some(strng::new(status)),
			);
		});
		let event = match status {
			"incomplete" => "response.incomplete",
			"failed" => "response.failed",
			_ => "response.completed",
		};
		self.event(events, event, json!({"response": response}));
		self.finished = true;
	}
}

pub(super) fn translate(
	body: Body,
	buffer_limit: usize,
	log: StreamingUsageGuard,
	log_content: LogContentFields,
	tool_map: ResponseToolMap,
) -> Body {
	let terminal = Arc::new(AtomicBool::new(false));
	let ended = terminal.clone();
	let (body, progress) = parse::sse_liveness::upstream_idle_timeout(body, UPSTREAM_IDLE_TIMEOUT);
	let mut stream = Stream {
		id: format!("resp_{:016x}", rand::rng().random::<u64>()),
		message_id: format!("msg_{:016x}", rand::rng().random::<u64>()),
		model: String::new(),
		created_at: chrono::Utc::now().timestamp() as u64,
		sequence: 0,
		created: false,
		finished: false,
		saw_token: false,
		text: String::new(),
		reasoning: String::new(),
		message_output_index: None,
		reasoning_output_index: None,
		next_output_index: 0,
		reasoning_item_id: format!("rs_{:016x}", rand::rng().random::<u64>()),
		tools: BTreeMap::new(),
		tool_map,
		stop: None,
		usage: None,
		buffered: 0,
		limit: buffer_limit,
		log,
		log_content,
	};
	let body = parse::sse::json_transform_multi::<completions::StreamResponse, Value, _>(
		body,
		buffer_limit,
		move |event| {
			let mut events = Vec::new();
			if stream.finished {
				return events;
			}
			match event {
				SseJsonEvent::Data(Ok(chunk)) => {
					let buffered = stream.buffered;
					let stopped = stream.stop.is_some();
					stream.chunk(chunk, &mut events);
					// Metadata, usage-only chunks, and empty deltas are not model
					// progress. Count new content/tool bytes or the first finish reason.
					if stream.buffered > buffered || (!stopped && stream.stop.is_some()) {
						progress.record();
					}
				},
				SseJsonEvent::Done | SseJsonEvent::Eof => stream.finish(&mut events),
				SseJsonEvent::Error if progress.timed_out() => stream.fail_with_code(
					&mut events,
					"upstream_timeout",
					&format!(
						"upstream SSE stream made no progress for {} seconds",
						UPSTREAM_IDLE_TIMEOUT.as_secs()
					),
				),
				SseJsonEvent::Data(Err(_)) | SseJsonEvent::Error => {
					stream.fail(&mut events, "invalid or interrupted upstream event stream")
				},
			}
			ended.store(stream.finished, Ordering::Relaxed);
			events
		},
	);
	parse::sse_liveness::keepalive(body, KEEPALIVE_INTERVAL, terminal)
}

#[cfg(test)]
mod tests {
	use axum_core::body::Body;
	use http_body_util::BodyExt;
	use serde_json::Value;

	use super::*;

	fn chunk(delta: Value, finish: Value) -> Value {
		json!({"id": "chatcmpl_test", "object": "chat.completion.chunk", "created": 1,
			"model": "moonshotai/Kimi-K3", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
	}

	fn events(wire: &str) -> Vec<Value> {
		wire
			.lines()
			.filter_map(|line| line.strip_prefix("data: "))
			.filter(|data| *data != "[DONE]")
			.map(|data| serde_json::from_str(data).expect("valid translated SSE event"))
			.collect()
	}

	#[tokio::test]
	async fn reasoning_content_is_forwarded_incrementally_as_responses_events() {
		let input = [
			chunk(json!({"reasoning_content": "Think "}), Value::Null),
			chunk(json!({"reasoning_content": "carefully."}), Value::Null),
			chunk(json!({"content": "Answer"}), json!("stop")),
		]
		.into_iter()
		.map(|chunk| format!("data: {chunk}\n\n"))
		.chain(std::iter::once("data: [DONE]\n\n".to_string()))
		.collect::<String>();
		let body = translate(
			Body::from(input),
			64 * 1024,
			Default::default(),
			Default::default(),
			ResponseToolMap::default(),
		);
		let wire = String::from_utf8(body.collect().await.unwrap().to_bytes().to_vec()).unwrap();
		let events = events(&wire);
		let reasoning_deltas: Vec<_> = events
			.iter()
			.filter(|event| event["type"] == "response.reasoning_summary_text.delta")
			.collect();
		assert_eq!(
			reasoning_deltas
				.iter()
				.map(|event| event["delta"].as_str().unwrap())
				.collect::<String>(),
			"Think carefully."
		);
		assert!(
			reasoning_deltas
				.iter()
				.all(|event| event["item_id"].as_str().is_some())
		);
		assert!(
			reasoning_deltas
				.iter()
				.all(|event| event["output_index"] == 0)
		);
		assert!(
			events
				.iter()
				.position(|event| event["type"] == "response.reasoning_summary_text.delta")
				.unwrap()
				< events
					.iter()
					.position(|event| event["type"] == "response.completed")
					.unwrap()
		);
		let completed = events.last().expect("terminal response event");
		assert_eq!(completed["type"], "response.completed");
		assert_eq!(completed["response"]["output"][0]["type"], "reasoning");
		assert_eq!(
			completed["response"]["output"][0]["summary"][0]["text"],
			"Think carefully."
		);
		assert_eq!(
			completed["response"]["output"][1]["content"][0]["text"],
			"Answer"
		);
	}
}
