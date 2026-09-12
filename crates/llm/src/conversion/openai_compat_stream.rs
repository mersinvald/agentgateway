use std::collections::BTreeMap;
use std::time::Instant;

use agent_core::strng;
use axum_core::body::Body;
use rand::RngExt;
use serde_json::{Value, json};

use super::ResponseToolMap;
use crate::parse::sse::SseJsonEvent;
use crate::types::completions::typed as completions;
use crate::{LogContentFields, StreamingUsageGuard, parse, types};

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
		if self.finished {
			return;
		}
		self.finished = true;
		let mut response = self.response("failed");
		response["error"] = json!({"code": "upstream_protocol_error", "message": message});
		self.event(events, "response.failed", json!({"response": response}));
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
		if let Some(reasoning) = choice.delta.reasoning_content {
			self.buffered = self.buffered.saturating_add(reasoning.len());
			self.reasoning.push_str(&reasoning);
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
		if !content.is_empty() {
			if self.text.is_empty() {
				self.event(events, "response.output_item.added", json!({"output_index": 0, "item": {
					"type": "message", "id": self.message_id, "role": "assistant", "status": "in_progress", "content": []
				}}));
				self.event(
					events,
					"response.content_part.added",
					json!({
						"item_id": self.message_id, "output_index": 0, "content_index": 0,
						"part": {"type": "output_text", "text": "", "annotations": []}
					}),
				);
			}
			self.text.push_str(&content);
			self.event(
				events,
				"response.output_text.delta",
				json!({
					"item_id": self.message_id, "output_index": 0, "content_index": 0, "delta": content
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
			let part = json!({"type": "output_text", "text": self.text, "annotations": []});
			let item = json!({"type": "message", "id": self.message_id,
				"role": "assistant", "status": if status == "completed" { "completed" } else { "incomplete" }, "content": [part]});
			self.event(
				events,
				"response.output_text.done",
				json!({
					"item_id": self.message_id, "output_index": 0, "content_index": 0, "text": self.text
				}),
			);
			self.event(
				events,
				"response.content_part.done",
				json!({
					"item_id": self.message_id, "output_index": 0, "content_index": 0, "part": part
				}),
			);
			self.event(
				events,
				"response.output_item.done",
				json!({"output_index": 0, "item": item}),
			);
			outputs.push(item);
		}
		if !self.reasoning.is_empty() {
			let item = json!({"type": "reasoning", "id": format!("rs_{}", self.id), "summary": [],
				"content": [{"type": "reasoning_text", "text": self.reasoning}],
				"status": if status == "completed" { "completed" } else { "incomplete" }});
			let mut added = item.clone();
			added["content"] = json!([]);
			added["status"] = json!("in_progress");
			self.event(
				events,
				"response.output_item.added",
				json!({"output_index": outputs.len(), "item": added}),
			);
			self.event(
				events,
				"response.output_item.done",
				json!({"output_index": outputs.len(), "item": item}),
			);
			outputs.push(item);
		}
		// Buffer tool arguments until complete so split JSON escapes, names and parallel calls
		// can be restored without exposing a JSON wrapper as executable free-form input.
		let mut logged_tools = Vec::new();
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
			let mut item = json!({"type": "function_call", "id": tool.id, "call_id": tool.id,
				"name": tool.name, "arguments": tool.arguments, "status": "completed"});
			if self.tool_map.restore_call(&mut item).is_err() {
				self.fail(events, "invalid JSON input for a custom tool call");
				return;
			}
			let custom = item["type"] == "custom_tool_call";
			let field = if custom { "input" } else { "arguments" };
			let mut added = item.clone();
			added[field] = json!("");
			added["status"] = json!("in_progress");
			let output_index = outputs.len();
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
			outputs.push(item);
		}
		let mut response = self.response(status);
		response["output"] = json!(outputs);
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
		tools: BTreeMap::new(),
		tool_map,
		stop: None,
		usage: None,
		buffered: 0,
		limit: buffer_limit,
		log,
		log_content,
	};
	parse::sse::json_transform_multi::<completions::StreamResponse, Value, _>(
		body,
		buffer_limit,
		move |event| {
			let mut events = Vec::new();
			if stream.finished {
				return events;
			}
			match event {
				SseJsonEvent::Data(Ok(chunk)) => stream.chunk(chunk, &mut events),
				SseJsonEvent::Done | SseJsonEvent::Eof => stream.finish(&mut events),
				SseJsonEvent::Data(Err(_)) | SseJsonEvent::Error => {
					stream.fail(&mut events, "invalid or interrupted upstream event stream")
				},
			}
			events
		},
	)
}
