//! Chat Completions facade for the Codex subscription Responses endpoint.
use std::collections::BTreeMap;
use std::sync::{
	Arc,
	atomic::{AtomicBool, Ordering},
};

use axum_core::body::Body;
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio_sse_codec::Frame;
use tokio_util::codec::BytesCodec;

use crate::parse::sse::SseDecoder;
use crate::parse::transform::TransformEvent;
use crate::types::ResponseType;
use crate::{AIError, LogContentFields, StreamingUsageGuard, parse, types};

fn unsupported(field: &str) -> AIError {
	AIError::UnsupportedConversion(
		format!("Codex Chat Completions: unsupported or invalid {field}").into(),
	)
}

fn fields(value: &Value, allowed: &[&str]) -> Result<(), AIError> {
	let object = value.as_object().ok_or_else(|| unsupported("object"))?;
	for (key, value) in object {
		if !value.is_null() && !allowed.contains(&key.as_str()) {
			return Err(unsupported(key));
		}
	}
	Ok(())
}

fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, AIError> {
	value[key].as_str().ok_or_else(|| unsupported(key))
}

pub fn translate(req: &types::completions::Request) -> Result<Vec<u8>, AIError> {
	let mut req = serde_json::to_value(req).map_err(AIError::RequestMarshal)?;
	// SDKs may send explicit defaults. Accept only neutral values for controls
	// that the subscription endpoint cannot implement; reject active controls below.
	for key in [
		"logprobs",
		"logit_bias",
		"frequency_penalty",
		"presence_penalty",
		"top_p",
	] {
		let neutral = match key {
			"logprobs" => req[key] == false,
			"logit_bias" => req[key].as_object().is_some_and(|value| value.is_empty()),
			"top_p" => req[key].as_f64() == Some(1.0),
			_ => req[key].as_f64() == Some(0.0),
		};
		if neutral {
			req
				.as_object_mut()
				.expect("serialized Chat request is an object")
				.remove(key);
		}
	}
	fields(
		&req,
		&[
			"model",
			"messages",
			"stream",
			"stream_options",
			"n",
			"temperature",
			"max_tokens",
			"max_completion_tokens",
			"tools",
			"tool_choice",
			"parallel_tool_calls",
			"response_format",
			"reasoning_effort",
			"metadata",
			"service_tier",
			"prompt_cache_key",
			"store",
			"user",
		],
	)?;
	if !req["n"].is_null() && req["n"].as_u64() != Some(1) {
		return Err(unsupported("n (only 1 is supported)"));
	}
	if !req["stream_options"].is_null() {
		fields(&req["stream_options"], &["include_usage"])?;
	}
	let mut out = json!({"model": req["model"], "input": [], "stream": req["stream"]});
	let mut input = Vec::new();
	for message in req["messages"]
		.as_array()
		.ok_or_else(|| unsupported("messages"))?
	{
		fields(
			message,
			&[
				"role",
				"content",
				"tool_calls",
				"tool_call_id",
				"reasoning_content",
				"refusal",
				"name",
			],
		)?;
		let role = string(message, "role")?;
		if !matches!(role, "system" | "developer" | "user" | "assistant" | "tool") {
			return Err(unsupported("message role"));
		}
		if !message["name"].is_null() {
			if role != "tool" {
				return Err(unsupported("name on non-tool message"));
			}
			// Hermes retains this redundant tool label. Responses links results by
			// call_id; forwarding name would be invalid, and changing content is unnecessary.
			string(message, "name")?;
		}
		if role != "assistant" && !message["tool_calls"].is_null() {
			return Err(unsupported("tool_calls on non-assistant message"));
		}
		if role != "assistant" && !message["refusal"].is_null() {
			return Err(unsupported("refusal on non-assistant message"));
		}
		if role == "tool" {
			let content = match &message["content"] {
				Value::String(text) => text.clone(),
				Value::Array(parts) => {
					let mut text = String::new();
					for part in parts {
						fields(part, &["type", "text"])?;
						if part["type"] != "text" {
							return Err(unsupported("tool result content"));
						}
						text.push_str(string(part, "text")?);
					}
					text
				},
				_ => return Err(unsupported("tool result content")),
			};
			input.push(json!({"type": "function_call_output", "call_id": string(message, "tool_call_id")?, "output": content}));
			continue;
		}
		let mut content = Vec::new();
		let text_type = if role == "assistant" {
			"output_text"
		} else {
			"input_text"
		};
		match &message["content"] {
			Value::String(text) => content.push(json!({"type": text_type, "text": text})),
			Value::Array(parts) => {
				for part in parts {
					match part["type"].as_str() {
						Some("text") => {
							fields(part, &["type", "text"])?;
							content.push(json!({"type": text_type, "text": string(part, "text")?}));
						},
						Some("refusal") if role == "assistant" => {
							fields(part, &["type", "refusal"])?;
							content.push(json!({"type": "refusal", "refusal": string(part, "refusal")?}));
						},
						Some("image_url") if role == "user" => {
							fields(part, &["type", "image_url"])?;
							fields(&part["image_url"], &["url", "detail"])?;
							let mut image =
								json!({"type": "input_image", "image_url": string(&part["image_url"], "url")?});
							if !part["image_url"]["detail"].is_null() {
								if !matches!(
									part["image_url"]["detail"].as_str(),
									Some("auto" | "low" | "high")
								) {
									return Err(unsupported("image detail"));
								}
								image["detail"] = part["image_url"]["detail"].clone();
							}
							content.push(image);
						},
						_ => return Err(unsupported("message content part")),
					}
				}
			},
			Value::Null
				if role == "assistant"
					&& (message["tool_calls"].is_array() || message["refusal"].is_string()) => {},
			_ => return Err(unsupported("message content")),
		}
		if !message["refusal"].is_null() {
			content.push(json!({"type": "refusal", "refusal": string(message, "refusal")?}));
		}
		if !content.is_empty() {
			// Assistant history uses output parts, without opaque item IDs or reasoning replay.
			input.push(json!({"role": role, "content": content}));
		}
		if !message["tool_calls"].is_null() {
			for call in message["tool_calls"]
				.as_array()
				.ok_or_else(|| unsupported("tool_calls"))?
			{
				fields(call, &["id", "type", "function"])?;
				fields(&call["function"], &["name", "arguments"])?;
				if call["type"] != "function" {
					return Err(unsupported("tool call type"));
				}
				input.push(json!({"type": "function_call", "call_id": string(call, "id")?, "name": string(&call["function"], "name")?, "arguments": string(&call["function"], "arguments")?}));
			}
		}
	}
	out["input"] = json!(input);
	if !req["tools"].is_null() {
		let mut tools = Vec::new();
		for tool in req["tools"]
			.as_array()
			.ok_or_else(|| unsupported("tools"))?
		{
			fields(tool, &["type", "function"])?;
			if tool["type"] != "function" {
				return Err(unsupported("tool type"));
			}
			fields(
				&tool["function"],
				&["name", "description", "parameters", "strict"],
			)?;
			string(&tool["function"], "name")?;
			let mut tool = tool["function"].clone();
			tool["type"] = json!("function");
			tools.push(tool);
		}
		out["tools"] = json!(tools);
	}
	if !req["tool_choice"].is_null() {
		out["tool_choice"] = match &req["tool_choice"] {
			Value::String(choice) if matches!(choice.as_str(), "auto" | "none" | "required") => {
				json!(choice)
			},
			choice if choice["type"] == "function" => {
				fields(choice, &["type", "function"])?;
				fields(&choice["function"], &["name"])?;
				json!({"type": "function", "name": string(&choice["function"], "name")?})
			},
			_ => return Err(unsupported("tool_choice")),
		};
	}
	if !req["parallel_tool_calls"].is_null() && !req["parallel_tool_calls"].is_boolean() {
		return Err(unsupported("parallel_tool_calls"));
	}
	for key in [
		"parallel_tool_calls",
		"metadata",
		"service_tier",
		"prompt_cache_key",
	] {
		if !req[key].is_null() {
			out[key] = req[key].clone();
		}
	}
	if !req["reasoning_effort"].is_null() {
		out["reasoning"] = json!({"effort": req["reasoning_effort"]});
	}
	if !req["response_format"].is_null() {
		let format = &req["response_format"];
		out["text"] = json!({"format": match format["type"].as_str() {
			Some("text" | "json_object") => { fields(format, &["type"])?; format.clone() },
			Some("json_schema") => {
				fields(format, &["type", "json_schema"])?;
				fields(&format["json_schema"], &["name", "description", "schema", "strict"])?;
				string(&format["json_schema"], "name")?;
				if !format["json_schema"]["schema"].is_object() { return Err(unsupported("json_schema.schema")); }
				let mut schema = format["json_schema"].clone();
				schema["type"] = json!("json_schema");
				schema
			},
			_ => return Err(unsupported("response_format")),
		}});
	}
	// Subscription normalization intentionally ignores temperature and both token limits,
	// forces store=false/stream=true, and removes the user safety identifier.
	serde_json::to_vec(&out).map_err(AIError::RequestMarshal)
}

fn invalid(message: &str) -> AIError {
	AIError::InvalidResponse(message.to_owned().into())
}

fn usage(response: &Value) -> Value {
	let u = &response["usage"];
	if u.is_null() {
		return Value::Null;
	}
	json!({"prompt_tokens": u["input_tokens"], "completion_tokens": u["output_tokens"], "total_tokens": u["total_tokens"],
		"prompt_tokens_details": u["input_tokens_details"], "completion_tokens_details": u["output_tokens_details"]})
}

fn finish(response: &Value, tools: bool, refusal: bool) -> Result<&'static str, AIError> {
	match response["status"].as_str() {
		Some("completed") => Ok(if tools {
			"tool_calls"
		} else if refusal {
			"content_filter"
		} else {
			"stop"
		}),
		Some("incomplete") => match response["incomplete_details"]["reason"].as_str() {
			Some("max_output_tokens") => Ok("length"),
			Some("content_filter") => Ok("content_filter"),
			_ => Err(invalid("unknown Responses incomplete reason")),
		},
		_ => Err(invalid("Responses request did not complete")),
	}
}

pub fn translate_response(bytes: &Bytes) -> Result<Box<dyn ResponseType>, AIError> {
	let response: Value = serde_json::from_slice(bytes).map_err(AIError::ResponseParsing)?;
	let mut state = StreamState::default();
	state.metadata(&response)?;
	let mut ignored = Vec::new();
	for (index, item) in response["output"]
		.as_array()
		.ok_or(AIError::IncompleteResponse)?
		.iter()
		.enumerate()
	{
		state.item(index as u64, item, &mut ignored)?;
	}
	let mut message = json!({"role": "assistant", "content": null});
	let text: String = state
		.text
		.iter()
		.filter(|((_, _, refusal), _)| !refusal)
		.map(|(_, text)| text.as_str())
		.collect();
	if !text.is_empty() {
		message["content"] = json!(text);
	}
	let refusal: String = state
		.text
		.iter()
		.filter(|((_, _, refusal), _)| *refusal)
		.map(|(_, text)| text.as_str())
		.collect();
	if !refusal.is_empty() {
		message["refusal"] = json!(refusal);
	}
	if !state.tools.is_empty() {
		message["tool_calls"] = json!(state.tools.values().map(|tool| json!({"id": tool.id, "type": "function", "function": {"name": tool.name, "arguments": tool.arguments}})).collect::<Vec<_>>());
	}
	let result = json!({"id": state.id, "object": "chat.completion", "model": state.model, "created": state.created,
		"choices": [{"index": 0, "message": message, "finish_reason": finish(&response, !state.tools.is_empty(), !refusal.is_empty())?, "logprobs": null}], "usage": usage(&response), "service_tier": response["service_tier"]});
	Ok(Box::new(
		serde_json::from_value::<types::completions::Response>(result)
			.map_err(AIError::ResponseParsing)?,
	))
}

#[derive(Default)]
struct Tool {
	index: usize,
	id: String,
	name: String,
	arguments: String,
}

#[derive(Default)]
struct StreamState {
	id: String,
	model: String,
	created: u64,
	started: bool,
	ended: bool,
	include_usage: bool,
	tools: BTreeMap<u64, Tool>,
	text: BTreeMap<(u64, u64, bool), String>,
}

impl StreamState {
	fn metadata(&mut self, response: &Value) -> Result<(), AIError> {
		if !self.id.is_empty() {
			return Ok(());
		}
		self.id = response["id"]
			.as_str()
			.filter(|s| !s.is_empty())
			.ok_or(AIError::IncompleteResponse)?
			.to_owned();
		self.model = response["model"]
			.as_str()
			.ok_or(AIError::IncompleteResponse)?
			.to_owned();
		self.created = response["created_at"]
			.as_u64()
			.ok_or(AIError::IncompleteResponse)?;
		Ok(())
	}

	fn chunk(&self, delta: Value, finish: Option<&str>) -> Value {
		let mut chunk = json!({"id": self.id, "object": "chat.completion.chunk", "model": self.model, "created": self.created,
			"choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": null}]});
		if self.include_usage {
			chunk["usage"] = Value::Null;
		}
		chunk
	}

	fn text(
		&mut self,
		key: (u64, u64, bool),
		text: &str,
		complete: bool,
		out: &mut Vec<Value>,
	) -> Result<(), AIError> {
		let current = self.text.entry(key).or_default();
		let delta = if complete {
			text
				.strip_prefix(current.as_str())
				.ok_or_else(|| invalid("inconsistent Responses text"))?
		} else {
			text
		};
		current.push_str(delta);
		if !delta.is_empty() {
			out.push(self.chunk(
				if key.2 {
					json!({"refusal": delta})
				} else {
					json!({"content": delta})
				},
				None,
			));
		}
		Ok(())
	}

	fn item(&mut self, index: u64, item: &Value, out: &mut Vec<Value>) -> Result<(), AIError> {
		match item["type"].as_str() {
			Some("function_call") => {
				let id = item["call_id"]
					.as_str()
					.filter(|s| !s.is_empty())
					.ok_or(AIError::IncompleteResponse)?;
				let name = item["name"]
					.as_str()
					.filter(|s| !s.is_empty())
					.ok_or(AIError::IncompleteResponse)?;
				if !self.tools.contains_key(&index) {
					let tool_index = self.tools.len();
					self.tools.insert(
						index,
						Tool {
							index: tool_index,
							id: id.into(),
							name: name.into(),
							arguments: String::new(),
						},
					);
					out.push(self.chunk(json!({"tool_calls": [{"index": tool_index, "id": id, "type": "function", "function": {"name": name, "arguments": ""}}]}), None));
				}
				let tool = &self.tools[&index];
				if tool.id != id || tool.name != name {
					return Err(invalid("inconsistent Responses tool identity"));
				}
				self.arguments(
					index,
					item["arguments"]
						.as_str()
						.ok_or(AIError::IncompleteResponse)?,
					true,
					out,
				)?;
			},
			Some("message") => {
				for (content_index, part) in item["content"]
					.as_array()
					.ok_or(AIError::IncompleteResponse)?
					.iter()
					.enumerate()
				{
					let refusal = match part["type"].as_str() {
						Some("output_text") => false,
						Some("refusal") => true,
						_ => return Err(invalid("unsupported Responses output content")),
					};
					self.text(
						(index, content_index as u64, refusal),
						part[if refusal { "refusal" } else { "text" }]
							.as_str()
							.ok_or(AIError::IncompleteResponse)?,
						true,
						out,
					)?;
				}
			},
			Some("reasoning") => {}, // Never expose opaque encrypted reasoning on the Chat API.
			_ => return Err(invalid("unsupported Responses output item")),
		}
		Ok(())
	}

	fn arguments(
		&mut self,
		index: u64,
		arguments: &str,
		complete: bool,
		out: &mut Vec<Value>,
	) -> Result<(), AIError> {
		let tool = self
			.tools
			.get_mut(&index)
			.ok_or_else(|| invalid("tool arguments before tool identity"))?;
		let delta = if complete {
			arguments
				.strip_prefix(&tool.arguments)
				.ok_or_else(|| invalid("inconsistent Responses tool arguments"))?
		} else {
			arguments
		};
		tool.arguments.push_str(delta);
		let index = tool.index;
		if !delta.is_empty() {
			out.push(self.chunk(
				json!({"tool_calls": [{"index": index, "function": {"arguments": delta}}]}),
				None,
			));
		}
		Ok(())
	}

	fn event(&mut self, event: Value) -> Result<Vec<Value>, AIError> {
		let mut out = Vec::new();
		let kind = event["type"].as_str().ok_or(AIError::IncompleteResponse)?;
		if kind == "error" || kind == "response.failed" {
			self.ended = true;
			let error = if kind == "error" {
				event.get("error").unwrap_or(&event)
			} else {
				&event["response"]["error"]
			};
			return Ok(vec![json!({"error": error})]);
		}
		if event["response"].is_object() {
			self.metadata(&event["response"])?;
		}
		if self.id.is_empty() {
			return Err(invalid("Responses stream missing response metadata"));
		}
		if !self.started {
			self.started = true;
			out.push(self.chunk(json!({"role": "assistant", "content": ""}), None));
		}
		let index = || {
			event["output_index"]
				.as_u64()
				.ok_or(AIError::IncompleteResponse)
		};
		match kind {
			"response.output_item.added" if event["item"]["type"] == "function_call" => {
				self.item(index()?, &event["item"], &mut out)?
			},
			"response.output_item.done" => self.item(index()?, &event["item"], &mut out)?,
			"response.function_call_arguments.delta" | "response.function_call_arguments.done" => {
				let complete = kind.ends_with(".done");
				self.arguments(
					index()?,
					event[if complete { "arguments" } else { "delta" }]
						.as_str()
						.ok_or(AIError::IncompleteResponse)?,
					complete,
					&mut out,
				)?;
			},
			"response.output_text.delta"
			| "response.output_text.done"
			| "response.refusal.delta"
			| "response.refusal.done" => {
				let complete = kind.ends_with(".done");
				let refusal = kind.starts_with("response.refusal.");
				self.text(
					(
						index()?,
						event["content_index"]
							.as_u64()
							.ok_or(AIError::IncompleteResponse)?,
						refusal,
					),
					event[if !complete {
						"delta"
					} else if refusal {
						"refusal"
					} else {
						"text"
					}]
					.as_str()
					.ok_or(AIError::IncompleteResponse)?,
					complete,
					&mut out,
				)?;
			},
			"response.completed" | "response.incomplete" => {
				let response = &event["response"];
				if let Some(items) = response["output"].as_array() {
					for (index, item) in items.iter().enumerate() {
						self.item(index as u64, item, &mut out)?;
					}
				} else if !response["output"].is_null() {
					return Err(AIError::IncompleteResponse);
				}
				let reason = finish(
					response,
					!self.tools.is_empty(),
					self.text.keys().any(|key| key.2),
				)?;
				out.push(self.chunk(json!({}), Some(reason)));
				if self.include_usage {
					let mut chunk = self.chunk(json!({}), None);
					chunk["choices"] = json!([]);
					chunk["usage"] = usage(response);
					out.push(chunk);
				}
				self.ended = true;
			},
			_ => {},
		}
		Ok(out)
	}
}

pub fn translate_stream(
	body: Body,
	buffer_limit: usize,
	log: StreamingUsageGuard,
	log_content: LogContentFields,
	include_usage: bool,
) -> Body {
	let body = super::passthrough_stream(body, buffer_limit, log, log_content);
	stream(body, buffer_limit, include_usage)
}

fn stream(body: Body, buffer_limit: usize, include_usage: bool) -> Body {
	let ended = Arc::new(AtomicBool::new(false));
	let terminal = ended.clone();
	let mut state = StreamState {
		include_usage,
		..Default::default()
	};
	let body = parse::transform::parser(
		body,
		SseDecoder::<Bytes>::with_max_size(buffer_limit),
		BytesCodec::new(),
		move |event| {
			if state.ended {
				return Vec::new();
			}
			let result = match event {
				TransformEvent::Item(Frame::Event(event)) if event.data.is_empty() => return Vec::new(),
				TransformEvent::Item(Frame::Event(event)) if event.data.as_ref() != b"[DONE]" => {
					serde_json::from_slice(&event.data)
						.map_err(AIError::ResponseParsing)
						.and_then(|v| state.event(v))
				},
				TransformEvent::Item(Frame::Comment(_) | Frame::Retry(_)) => return Vec::new(),
				_ => Err(invalid("Responses stream ended before terminal response")),
			};
			let values = match result {
				Ok(values) => values,
				Err(error) => {
					state.ended = true;
					vec![json!({"error": {"type": "api_error", "message": error.to_string()}})]
				},
			};
			let success = state.ended && values.last().is_some_and(|v| v.get("error").is_none());
			let mut frames: Vec<_> = values
				.into_iter()
				.map(|v| parse::encode_sse_event("", Bytes::from(v.to_string())))
				.collect();
			if success {
				frames.push(parse::encode_sse_event("", Bytes::from_static(b"[DONE]")));
			}
			terminal.store(state.ended, Ordering::Relaxed);
			frames
		},
	);
	// Drop the upstream immediately after yielding the terminal batch, rather than
	// waiting for EOF (or leaking a later transport error after our DONE marker).
	Body::from_stream(futures_util::stream::unfold(
		Some(body.into_data_stream()),
		move |body| {
			let ended = ended.clone();
			async move {
				let mut body = body?;
				let frame = body.next().await?;
				Some((frame, (!ended.load(Ordering::Relaxed)).then_some(body)))
			}
		},
	))
}

#[cfg(test)]
mod tests {
	use super::*;
	use http_body_util::BodyExt;

	fn response() -> Value {
		json!({"id": "resp_test", "model": "gpt-5.6-luna", "created_at": 123, "status": "completed", "output": [],
			"usage": {"input_tokens": 30, "output_tokens": 10, "total_tokens": 40, "input_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 2}, "output_tokens_details": {"reasoning_tokens": 3}}})
	}

	fn state() -> StreamState {
		let mut state = StreamState::default();
		state
			.event(json!({"type": "response.created", "response": response()}))
			.unwrap();
		state
	}

	fn tool(id: &str, name: &str, arguments: &str) -> Value {
		json!({"type": "function_call", "id": format!("fc_{id}"), "call_id": id, "name": name, "arguments": arguments})
	}

	#[test]
	fn rejects_unsupported_and_malformed_controls() {
		for (field, value) in [
			("n", json!(0)),
			("n", json!(2)),
			("n", json!(-1)),
			("n", json!(1.5)),
			("n", json!("1")),
			("logprobs", json!(true)),
			("logit_bias", json!({"42": 1})),
			("logit_bias", json!([])),
			("logprobs", json!("false")),
			("presence_penalty", json!(0.1)),
			("top_logprobs", json!(2)),
			("audio", json!({})),
			("modalities", json!(["audio"])),
			("stop", json!(["END"])),
			("seed", json!(1)),
			("top_p", json!(0.9)),
			("frequency_penalty", json!(1)),
			("prediction", json!({})),
			("functions", json!([])),
			("parallel_tool_calls", json!("yes")),
			(
				"stream_options",
				json!({"include_usage": true, "other": true}),
			),
		] {
			let mut request = json!({"messages": [{"role": "user", "content": "Hello"}]});
			request[field] = value;
			let request = serde_json::from_value(request).unwrap();
			assert!(translate(&request).is_err(), "{field}");
		}
		for part in [
			json!({"type": "input_audio", "input_audio": {"data": "test", "format": "wav"}}),
			json!({"type": "file", "file": {"file_id": "file_1"}}),
		] {
			let request =
				serde_json::from_value(json!({"messages": [{"role": "user", "content": [part]}]})).unwrap();
			assert!(translate(&request).is_err());
		}
	}

	#[test]
	fn public_tool_history_does_not_require_encrypted_reasoning() {
		let request = serde_json::from_value(json!({"model": "gpt-5.6-luna", "messages": [
			{"role": "assistant", "content": null, "reasoning_content": "I should look this up", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
			{"role": "tool", "name": "lookup", "tool_call_id": "call_1", "content": "found"}], "temperature": 0.7, "max_tokens": 256})).unwrap();
		let value: Value = serde_json::from_slice(&translate(&request).unwrap()).unwrap();
		assert_eq!(value["input"].as_array().unwrap().len(), 2);
		assert_eq!(value["input"][0]["call_id"], "call_1");
		assert_eq!(value["input"][0]["name"], "lookup");
		assert_eq!(
			value["input"][1],
			json!({"type": "function_call_output", "call_id": "call_1", "output": "found"})
		);
		assert!(value.get("temperature").is_none());
		assert!(value.get("max_output_tokens").is_none());
		assert!(!value.to_string().contains("reasoning_content"));
	}

	#[test]
	fn rejects_invalid_tool_names_and_named_participants() {
		for name in [json!(42), json!(false), json!([]), json!({})] {
			let request = serde_json::from_value::<types::completions::Request>(json!({"messages": [
				{"role": "tool", "name": name, "tool_call_id": "call_1", "content": "found"}
			]}));
			assert!(request.is_err());
		}
		// Participant names have meaning beyond a tool call ID; do not silently discard them.
		for role in ["system", "developer", "user", "assistant"] {
			let request = serde_json::from_value(json!({"messages": [
				{"role": role, "name": "participant", "content": "Hello"}
			]}))
			.unwrap();
			assert!(translate(&request).is_err());
		}
	}

	#[test]
	fn tool_deltas_done_and_terminal_are_not_duplicated() {
		let mut state = state();
		let mut chunks = Vec::new();
		for (index, id, name) in [(4, "call_a", "lookup"), (7, "call_b", "search")] {
			chunks.extend(state.event(json!({"type": "response.output_item.added", "output_index": index, "item": tool(id, name, "")})).unwrap());
			for delta in ["{\"q\":", "\"cat\"}"] {
				chunks.extend(state.event(json!({"type": "response.function_call_arguments.delta", "output_index": index, "delta": delta})).unwrap());
			}
			chunks.extend(state.event(json!({"type": "response.function_call_arguments.done", "output_index": index, "arguments": "{\"q\":\"cat\"}"})).unwrap());
			for _ in 0..2 {
				chunks.extend(state.event(json!({"type": "response.output_item.done", "output_index": index, "item": tool(id, name, "{\"q\":\"cat\"}")})).unwrap());
			}
		}
		let terminal = state
			.event(json!({"type": "response.completed", "response": response()}))
			.unwrap();
		assert_eq!(terminal[0]["choices"][0]["finish_reason"], "tool_calls");
		assert_eq!(chunks.len(), 6);
		for index in 0..2 {
			let tools: Vec<_> = chunks
				.iter()
				.map(|c| &c["choices"][0]["delta"]["tool_calls"][0])
				.filter(|t| t["index"] == index)
				.collect();
			assert_eq!(tools.iter().filter(|t| t.get("id").is_some()).count(), 1);
			assert_eq!(
				tools
					.iter()
					.filter(|t| t["function"].get("name").is_some())
					.count(),
				1
			);
			assert_eq!(
				tools
					.iter()
					.map(|t| t["function"]["arguments"].as_str().unwrap())
					.collect::<String>(),
				"{\"q\":\"cat\"}"
			);
		}
	}

	#[test]
	fn done_only_and_partial_arguments_recover_only_missing_suffix() {
		let mut state = state();
		let mut out = Vec::new();
		state
			.item(3, &tool("call_a", "lookup", "{"), &mut out)
			.unwrap();
		state
			.item(3, &tool("call_a", "lookup", "{}"), &mut out)
			.unwrap();
		assert_eq!(out.len(), 3);
		assert_eq!(
			out[2]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
			"}"
		);
		assert!(
			state
				.item(3, &tool("call_a", "lookup", "[]"), &mut out)
				.is_err()
		);
		assert!(
			state
				.item(3, &tool("call_a", "changed", "{}"), &mut out)
				.is_err()
		);
		assert!(state.arguments(9, "{}", false, &mut out).is_err());
	}

	#[test]
	fn text_terminal_fallback_metadata_and_finish_reasons() {
		for (status, reason, expected) in [
			("completed", Value::Null, "stop"),
			("incomplete", json!("max_output_tokens"), "length"),
			("incomplete", json!("content_filter"), "content_filter"),
		] {
			let mut state = state();
			state.include_usage = true;
			state.event(json!({"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "Hel"})).unwrap();
			let mut response = response();
			response["status"] = json!(status);
			response["incomplete_details"] = json!({"reason": reason});
			response["model"] = json!("different-terminal-model");
			response["output"] =
				json!([{"type": "message", "content": [{"type": "output_text", "text": "Hello"}]}]);
			let out = state
				.event(json!({"type": format!("response.{status}"), "response": response}))
				.unwrap();
			assert_eq!(out[0]["choices"][0]["delta"]["content"], "lo");
			assert_eq!(out[1]["choices"][0]["finish_reason"], expected);
			assert_eq!(out[2]["choices"], json!([]));
			assert_eq!(
				out[2]["usage"]["completion_tokens_details"]["reasoning_tokens"],
				3
			);
			for chunk in &out {
				assert_eq!(chunk["id"], "resp_test");
				assert_eq!(chunk["model"], "gpt-5.6-luna");
				assert_eq!(chunk["created"], 123);
			}
		}
	}

	async fn wire(events: Vec<Value>, include_usage: bool) -> String {
		let body = events
			.iter()
			.map(|e| format!("data: {e}\r\n\r\n"))
			.collect::<String>();
		// Split every UTF-8 byte and SSE delimiter across HTTP body frames.
		let fragments = body
			.bytes()
			.map(|b| Ok::<_, std::io::Error>(Bytes::from(vec![b])))
			.collect::<Vec<_>>();
		let body = Body::from_stream(futures_util::stream::iter(fragments));
		String::from_utf8(
			stream(body, 65536, include_usage)
				.collect()
				.await
				.unwrap()
				.to_bytes()
				.to_vec(),
		)
		.unwrap()
	}

	#[tokio::test]
	async fn reasoning_then_streamed_tools_replay_as_standard_chat_history() {
		let reasoning = json!({"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque-not-for-chat"});
		let mut events = vec![
			json!({"type": "response.created", "response": response()}),
			json!({"type": "response.output_item.added", "output_index": 0, "item": reasoning}),
			json!({"type": "response.reasoning_summary_text.delta", "output_index": 0, "delta": "Checking"}),
			json!({"type": "response.output_item.done", "output_index": 0, "item": reasoning}),
		];
		for (index, id, name) in [(1, "call_a", "lookup"), (2, "call_b", "search")] {
			events.push(json!({"type": "response.output_item.added", "output_index": index, "item": tool(id, name, "")}));
			for delta in ["{\"q\":", "\"cat\"}"] {
				events.push(json!({"type": "response.function_call_arguments.delta", "output_index": index, "delta": delta}));
			}
			events.push(json!({"type": "response.function_call_arguments.done", "output_index": index, "arguments": "{\"q\":\"cat\"}"}));
			events.push(json!({"type": "response.output_item.done", "output_index": index, "item": tool(id, name, "{\"q\":\"cat\"}")}));
		}
		// Repeated terminal data must not repeat the finish, usage, or DONE chunks.
		for _ in 0..2 {
			events.push(json!({"type": "response.completed", "response": response()}));
		}
		events.push(json!({"type": "response.output_text.delta", "output_index": 3, "content_index": 0, "delta": "late-output-must-not-escape"}));
		for include_usage in [false, true] {
			let wire = wire(events.clone(), include_usage).await;
			assert_eq!(wire.matches("[DONE]").count(), 1);
			assert!(!wire.contains("opaque-not-for-chat"));
			assert!(!wire.contains("late-output-must-not-escape"));
			let chunks: Vec<Value> = wire
				.lines()
				.filter_map(|line| line.strip_prefix("data: "))
				.filter(|line| *line != "[DONE]")
				.map(|line| serde_json::from_str(line).unwrap())
				.collect();
			assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
			let finishes: Vec<_> = chunks
				.iter()
				.filter_map(|c| c["choices"][0]["finish_reason"].as_str())
				.collect();
			assert_eq!(finishes, ["tool_calls"]);
			let usage: Vec<_> = chunks
				.iter()
				.filter(|c| c["choices"] == json!([]))
				.collect();
			assert_eq!(usage.len(), usize::from(include_usage));
			if include_usage {
				assert_eq!(
					usage[0]["usage"]["completion_tokens_details"]["reasoning_tokens"],
					3
				);
			}
			let mut calls = BTreeMap::new();
			let mut finished = false;
			for chunk in &chunks {
				if !chunk["choices"].as_array().unwrap().is_empty() {
					assert!(!finished, "choice output after finish");
					finished = chunk["choices"][0]["finish_reason"].is_string();
				} else {
					assert!(finished, "usage before finish");
				}
				assert_eq!(chunk["object"], "chat.completion.chunk");
				assert_eq!(chunk["id"], "resp_test");
				assert_eq!(chunk["model"], "gpt-5.6-luna");
				assert_eq!(chunk["created"], 123);
				if let Some(deltas) = chunk["choices"][0]["delta"]["tool_calls"].as_array() {
					for delta in deltas {
						let call = calls
							.entry(delta["index"].as_u64().unwrap())
							.or_insert_with(
								|| json!({"id": "", "type": null, "function": {"name": "", "arguments": ""}}),
							);
						if let Some(kind) = delta["type"].as_str() {
							assert_eq!(kind, "function");
							call["type"] = json!(kind);
						}
						if let Some(id) = delta["id"].as_str() {
							call["id"] = json!(format!("{}{id}", call["id"].as_str().unwrap()));
						}
						for field in ["name", "arguments"] {
							if let Some(text) = delta["function"][field].as_str() {
								call["function"][field] = json!(format!(
									"{}{text}",
									call["function"][field].as_str().unwrap()
								));
							}
						}
					}
				}
			}
			assert_eq!(calls.keys().copied().collect::<Vec<_>>(), [0, 1]);
			assert_eq!(calls[&0]["id"], "call_a");
			assert_eq!(calls[&0]["function"]["name"], "lookup");
			assert_eq!(calls[&1]["id"], "call_b");
			assert_eq!(calls[&1]["function"]["name"], "search");
			for call in calls.values() {
				assert_eq!(call["type"], "function");
				assert_eq!(call["function"]["arguments"], "{\"q\":\"cat\"}");
			}
			let mut request: Value = serde_json::from_str(include_str!(
				"../../tests/requests/completions/codex/full.json"
			))
			.unwrap();
			let full: Value = serde_json::from_slice(
				&translate(&serde_json::from_value(request.clone()).unwrap()).unwrap(),
			)
			.unwrap();
			assert_eq!(full["input"][0]["role"], "system");
			assert_eq!(full["input"][1]["role"], "developer");
			for key in [
				"logprobs",
				"logit_bias",
				"frequency_penalty",
				"presence_penalty",
				"top_p",
			] {
				assert!(full.get(key).is_none());
			}
			request["messages"] = json!([
				{"role": "assistant", "content": null, "tool_calls": calls.into_values().collect::<Vec<_>>()},
				{"role": "tool", "tool_call_id": "call_a", "content": "found"},
				{"role": "tool", "tool_call_id": "call_b", "content": "confirmed"}
			]);
			let translated: Value =
				serde_json::from_slice(&translate(&serde_json::from_value(request).unwrap()).unwrap())
					.unwrap();
			assert_eq!(translated["input"].as_array().unwrap().len(), 4);
			assert_eq!(translated["input"][0]["call_id"], "call_a");
			assert_eq!(translated["input"][1]["call_id"], "call_b");
			assert_eq!(
				translated["input"][2],
				json!({"type": "function_call_output", "call_id": "call_a", "output": "found"})
			);
			assert_eq!(
				translated["input"][3],
				json!({"type": "function_call_output", "call_id": "call_b", "output": "confirmed"})
			);
			assert_eq!(translated["reasoning"]["effort"], "high");
			assert_eq!(translated["text"]["format"]["type"], "json_schema");
			assert_eq!(translated["text"]["format"]["strict"], true);
		}
	}

	#[tokio::test]
	async fn terminal_drops_upstream_before_delayed_error_or_pending_forever() {
		for failed in [false, true] {
			for suffix_kind in ["transport_error", "decoder_error", "pending"] {
				let event = if failed {
					json!({"type": "error", "message": "upstream failed"})
				} else {
					json!({"type": "response.completed", "response": response()})
				};
				let polls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
				let observed = polls.clone();
				let suffix = futures_util::stream::once(async move {
					observed.fetch_add(1, Ordering::Relaxed);
					if suffix_kind == "pending" {
						std::future::pending::<()>().await;
					}
					tokio::time::sleep(std::time::Duration::from_millis(10)).await;
					if suffix_kind == "decoder_error" {
						return Ok(Bytes::from(format!("data: {}\n\n", "x".repeat(65537))));
					}
					Err::<Bytes, _>(std::io::Error::other("late transport error"))
				});
				let body = Body::from_stream(
					futures_util::stream::iter([Ok(Bytes::from(format!("data: {event}\n\n")))]).chain(suffix),
				);
				let result = tokio::time::timeout(
					std::time::Duration::from_secs(1),
					stream(body, 65536, true).collect(),
				)
				.await
				.expect("must not wait for upstream EOF")
				.expect("must not leak late transport errors")
				.to_bytes();
				let wire = String::from_utf8(result.to_vec()).unwrap();
				assert_eq!(wire.matches("[DONE]").count(), usize::from(!failed));
				assert_eq!(wire.contains("\"error\""), failed);
				assert_eq!(
					polls.load(Ordering::Relaxed),
					0,
					"upstream was polled after terminal"
				);
			}
		}
	}

	#[test]
	fn assistant_history_uses_output_text_for_scalar_and_array_content() {
		for content in [
			json!("MEMORY_91823"),
			json!([{"type": "text", "text": "MEMORY_91823"}]),
		] {
			let request = serde_json::from_value(json!({"messages": [
				{"role": "system", "content": "Remember the conversation."},
				{"role": "developer", "content": [{"type": "text", "text": "Repeat faithfully."}]},
				{"role": "user", "content": "What is the marker?"},
				{"role": "assistant", "content": content},
				{"role": "user", "content": "Repeat the marker."}
			]}))
			.unwrap();
			let result: Value = serde_json::from_slice(&translate(&request).unwrap()).unwrap();
			for index in [0, 1, 2, 4] {
				assert_eq!(result["input"][index]["content"][0]["type"], "input_text");
			}
			assert_eq!(
				result["input"][3],
				json!({"role": "assistant", "content": [{"type": "output_text", "text": "MEMORY_91823"}]})
			);
		}
	}

	#[test]
	fn refusal_history_round_trips_and_is_assistant_only() {
		let mut response = response();
		response["output"] =
			json!([{"type": "message", "content": [{"type": "refusal", "refusal": "Cannot help."}]}]);
		let translated = translate_response(&Bytes::from(response.to_string())).unwrap();
		let completion: Value = serde_json::from_slice(&translated.serialize().unwrap()).unwrap();
		for content in [Value::Null, json!("Context.")] {
			let mut message = completion["choices"][0]["message"].clone();
			message["content"] = content.clone();
			let request = serde_json::from_value(json!({"messages": [message]})).unwrap();
			let request: Value = serde_json::from_slice(&translate(&request).unwrap()).unwrap();
			if content.is_string() {
				assert_eq!(
					request["input"][0]["content"][0],
					json!({"type": "output_text", "text": "Context."})
				);
			}
			assert_eq!(
				request["input"][0]["content"]
					.as_array()
					.unwrap()
					.last()
					.unwrap()["refusal"],
				"Cannot help."
			);
		}
		for role in ["assistant", "user", "system", "developer", "tool"] {
			for message in [
				json!({"role": role, "content": null, "refusal": "Cannot help."}),
				json!({"role": role, "content": [{"type": "refusal", "refusal": "Cannot help."}]}),
			] {
				let request = serde_json::from_value(json!({"messages": [message]})).unwrap();
				let result = translate(&request);
				assert_eq!(result.is_ok(), role == "assistant");
				if let Ok(bytes) = result {
					let request: Value = serde_json::from_slice(&bytes).unwrap();
					assert_eq!(
						request["input"][0],
						json!({"role": "assistant", "content": [{"type": "refusal", "refusal": "Cannot help."}]})
					);
				}
			}
		}
	}

	#[tokio::test]
	async fn wire_errors_and_truncated_streams_never_emit_done_or_success() {
		for events in [
			vec![],
			vec![json!({"type": "response.created", "response": response()})],
			vec![json!({"type": "response.output_text.delta", "delta": "missing metadata"})],
			vec![json!({"type": "error", "code": "server_error", "message": "failed"})],
			vec![
				json!({"type": "response.failed", "response": {"error": {"code": "server_error", "message": "failed"}}}),
			],
			vec![json!({"type": "response.completed", "response": {"status": "completed"}})],
		] {
			let wire = wire(events, false).await;
			assert!(wire.contains("\"error\""), "{wire}");
			assert!(!wire.contains("[DONE]"));
			assert!(!wire.contains("\"finish_reason\":\"stop\""));
		}
		for bad in ["data: invalid-json\n\n", "data: [DONE]\n\n"] {
			let bytes = stream(Body::from(bad), 65536, false)
				.collect()
				.await
				.unwrap()
				.to_bytes();
			let text = String::from_utf8(bytes.to_vec()).unwrap();
			assert!(text.contains("\"error\""));
			assert!(!text.contains("[DONE]"));
		}
	}

	#[tokio::test]
	async fn response_and_stream_goldens() {
		for name in ["basic", "tool", "reasoning"] {
			let path = format!("src/tests/response/responses/{name}.json");
			let bytes = Bytes::from(std::fs::read(path).unwrap());
			let translated = translate_response(&bytes).unwrap();
			let json: Value = serde_json::from_slice(&translated.serialize().unwrap()).unwrap();
			let response: Value = serde_json::from_slice(&bytes).unwrap();
			let mut created = response.clone();
			created["output"] = json!([]);
			let mut events = vec![json!({"type": "response.created", "response": created})];
			for (index, item) in response["output"].as_array().unwrap().iter().enumerate() {
				events
					.push(json!({"type": "response.output_item.done", "output_index": index, "item": item}));
			}
			events.push(json!({"type": "response.completed", "response": response}));
			let with_usage = wire(events.clone(), true).await;
			let without_usage = wire(events, false).await;
			assert_eq!(
				with_usage.matches("[DONE]").count(),
				1,
				"{name}: {with_usage}"
			);
			assert_eq!(without_usage.matches("[DONE]").count(), 1);
			assert!(!without_usage.contains("\"usage\""));
			insta::with_settings!({snapshot_path => "../../tests/response/responses", prepend_module_to_snapshot => false}, {
				insta::assert_json_snapshot!(format!("{name}.codex-completions"), json!({"response": json, "parsed": translated.to_llm_response(LogContentFields::default()), "stream": with_usage}));
			});
		}
	}
}
