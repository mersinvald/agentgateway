use std::collections::BTreeMap;

use bytes::{Bytes, BytesMut};
use serde_json::{Value, json};
use tokio_sse_codec::{Frame, SseDecoder};
use tokio_util::codec::Decoder;

use super::{AIError, BufferedResponse};

/// Maximum amount of an initial Codex SSE stream that may be held while looking for an
/// overload error. Once this limit is reached, the stream is forwarded as-is.
pub(super) const OVERLOAD_PREFIX_LIMIT: usize = 64 * 1024;

/// A complete Responses event can be classified without consuming the rest of the stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StreamEventDisposition {
	Continue,
	OutputStarted,
	Overloaded,
}

fn classify_named_stream_event(event_name: Option<&str>, data: &[u8]) -> StreamEventDisposition {
	if data.is_empty() {
		return StreamEventDisposition::Continue;
	}
	if data == b"[DONE]" {
		return StreamEventDisposition::OutputStarted;
	}
	let Ok(event) = serde_json::from_slice::<Value>(data) else {
		// Malformed or future data cannot be assumed to be harmless prelude. Forward it and
		// let the normal translator handle the stream without risking a replay.
		return StreamEventDisposition::OutputStarted;
	};
	let event_type = event
		.get("type")
		.and_then(Value::as_str)
		.or(event_name)
		.unwrap_or_default();
	if overload_error_named(event_name, &event).is_some() {
		StreamEventDisposition::Overloaded
	} else if matches!(
		event_type,
		"response.created" | "response.in_progress" | "response.queued"
	) {
		StreamEventDisposition::Continue
	} else {
		// Only documented lifecycle prelude events are safe to wait through. All output,
		// terminal, unknown, or future event types end lookahead so later errors cannot replay.
		StreamEventDisposition::OutputStarted
	}
}

pub(super) fn overload_error_named(event_name: Option<&str>, event: &Value) -> Option<Value> {
	let event_type = event.get("type").and_then(Value::as_str).or(event_name)?;
	let error = match event_type {
		"error" => event.get("error").unwrap_or(event),
		"response.failed" => event
			.get("response")
			.and_then(|r| r.get("error"))
			.unwrap_or(event),
		_ => return None,
	};
	(error.get("code").and_then(Value::as_str) == Some("server_is_overloaded")).then(|| error.clone())
}

/// Incrementally decodes complete SSE events while retaining the raw prefix for replay into
/// the normal streaming translator. The caller stops reading as soon as output or overload
/// is seen, or when the prefix limit is reached.
pub(super) struct OverloadStreamInspector {
	decoder: SseDecoder<Bytes>,
	buffer: BytesMut,
	decision: Option<StreamEventDisposition>,
}

impl OverloadStreamInspector {
	pub(super) fn new() -> Self {
		Self {
			decoder: SseDecoder::with_max_size(OVERLOAD_PREFIX_LIMIT),
			buffer: BytesMut::new(),
			decision: None,
		}
	}

	pub(super) fn push(&mut self, bytes: &[u8]) -> StreamEventDisposition {
		if let Some(decision) = self.decision {
			return decision;
		}
		self.buffer.extend_from_slice(bytes);
		loop {
			match self.decoder.decode(&mut self.buffer) {
				Ok(Some(Frame::Event(event))) => {
					let disposition = classify_named_stream_event(Some(&event.name), &event.data);
					if disposition != StreamEventDisposition::Continue {
						self.decision = Some(disposition);
						return disposition;
					}
				},
				Ok(Some(_)) => continue,
				Ok(None) => return StreamEventDisposition::Continue,
				Err(_) => {
					self.decision = Some(StreamEventDisposition::OutputStarted);
					return StreamEventDisposition::OutputStarted;
				},
			}
		}
	}
}

/// Apply subscription-only constraints after translation and policy overrides. Do not use an
/// allowlist: Codex supports Responses tools, images, reasoning, text and stream controls, and
/// new fields must be able to pass through without waiting for a gateway schema update.
pub(super) fn normalize_request(bytes: &[u8]) -> Result<Vec<u8>, AIError> {
	let mut body: Value = serde_json::from_slice(bytes).map_err(AIError::RequestParsing)?;
	let obj = body.as_object_mut().ok_or(AIError::UnsupportedContent)?;
	// Codex CLI's ResponsesApiRequest omits token limits; OpenCode's Codex auth plugin likewise
	// clears maxOutputTokens. These other advisory fields are rejected by the subscription API.
	for key in [
		"max_output_tokens",
		"temperature",
		"prompt_cache_breakpoint",
		"prompt_cache_retention",
		"safety_identifier",
	] {
		obj.remove(key);
	}
	// AI SDK emits this unsupported hint on message and tool-result content parts.
	// Do not recursively strip keys from opaque history, tool schemas or user data.
	if let Some(Value::Array(input)) = obj.get_mut("input") {
		for item in input {
			let field = match item.get("type").and_then(Value::as_str) {
				Some("function_call_output" | "custom_tool_call_output") => "output",
				Some("message") | None if item.get("role").is_some() => "content",
				_ => continue,
			};
			if let Some(Value::Array(parts)) = item.get_mut(field) {
				for part in parts {
					if let Some(part) = part.as_object_mut() {
						part.remove("prompt_cache_breakpoint");
					}
				}
			}
		}
	}
	obj.insert("store".into(), json!(false));
	// Keep LLMRequest.streaming unchanged so unary callers still receive a JSON response.
	obj.insert("stream".into(), json!(true));
	if obj.get("instructions").is_none_or(Value::is_null) {
		obj.insert("instructions".into(), json!(""));
	}
	if let Some(Value::String(text)) = obj.get("input") {
		obj.insert(
			"input".into(),
			json!([{
				"role": "user", "content": [{"type": "input_text", "text": text}]
			}]),
		);
	}
	if obj.get("include").is_none_or(Value::is_null) {
		obj.insert("include".into(), json!([]));
	}
	if let Some(Value::Array(include)) = obj.get_mut("include") {
		let encrypted = json!("reasoning.encrypted_content");
		if !include.contains(&encrypted) {
			include.push(encrypted);
		}
	}
	serde_json::to_vec(&body).map_err(AIError::RequestMarshal)
}

/// The upstream is always SSE. Keep completed items as well as the terminal response: Codex
/// can omit terminal output even after emitting full output_item.done events. Do not rebuild
/// items from text deltas, which would lose opaque reasoning, tools and future fields.
/// The caller has already decompressed and enforced the configured response buffer limit.
pub(super) fn normalize_unary_response(buffered: &mut BufferedResponse) -> Result<(), AIError> {
	let mut bytes = BytesMut::from(buffered.bytes.as_ref());
	let mut decoder = SseDecoder::<Bytes>::with_max_size(bytes.len().max(8));
	let mut output = BTreeMap::new();
	while let Some(frame) = decoder
		.decode_eof(&mut bytes)
		.map_err(|_| AIError::InvalidResponse("invalid Codex SSE response".into()))?
	{
		let Frame::Event(event) = frame else { continue };
		if event.data.is_empty() || event.data.as_ref() == b"[DONE]" {
			continue;
		}
		let mut event: Value = serde_json::from_slice(&event.data).map_err(AIError::ResponseParsing)?;
		match event.get("type").and_then(Value::as_str) {
			Some("response.output_item.done") => {
				let index = event
					.get("output_index")
					.and_then(Value::as_u64)
					.ok_or(AIError::IncompleteResponse)?;
				let item = event
					.get_mut("item")
					.filter(|v| v.is_object())
					.ok_or(AIError::IncompleteResponse)?
					.take();
				output.insert(index, item);
				continue;
			},
			Some("response.completed" | "response.incomplete" | "response.failed") => {
				let mut response = event
					.get_mut("response")
					.filter(|v| v.is_object())
					.ok_or(AIError::IncompleteResponse)?
					.take();
				if response
					.get("output")
					.is_none_or(|v| v.is_null() || v.as_array().is_some_and(Vec::is_empty))
				{
					response["output"] = Value::Array(output.into_values().collect());
				}
				buffered.bytes = serde_json::to_vec(&response)
					.map_err(AIError::ResponseMarshal)?
					.into();
			},
			Some("error") => {
				let error = event.get_mut("error").map(Value::take).unwrap_or(event);
				buffered.parts.status = http::StatusCode::BAD_GATEWAY;
				buffered.bytes = serde_json::to_vec(&json!({"error": error}))
					.map_err(AIError::ResponseMarshal)?
					.into();
			},
			_ => continue,
		}
		buffered.parts.headers.insert(
			http::header::CONTENT_TYPE,
			http::HeaderValue::from_static("application/json"),
		);
		return Ok(());
	}
	Err(AIError::IncompleteResponse)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn defaults_and_string_input_are_normalized_idempotently() {
		for value in [None, Some(Value::Null), Some(json!(4096))] {
			let mut body = json!({"model": "gpt-test", "input": "hello"});
			if let Some(value) = value {
				body["max_output_tokens"] = value;
			}
			let normalized = normalize_request(&serde_json::to_vec(&body).unwrap()).unwrap();
			assert_eq!(
				serde_json::from_slice::<Value>(&normalized).unwrap(),
				json!({
					"model": "gpt-test", "input": [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
					"instructions": "", "store": false, "stream": true, "include": ["reasoning.encrypted_content"]
				})
			);
			assert_eq!(normalize_request(&normalized).unwrap(), normalized);
		}
	}

	#[test]
	fn cache_breakpoints_are_removed_only_from_wire_content_parts() {
		for hint in [Value::Null, json!({"mode": "explicit"})] {
			let mut body = json!({
				"model": "gpt-test", "instructions": "", "store": false, "stream": true,
				"include": ["reasoning.encrypted_content"], "prompt_cache_key": "session-test",
				"input": [
					{"role": "system", "content": [{"type": "input_text", "text": "system"}]},
					{"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "developer"}]},
					{"role": "user", "content": [
						{"type": "input_text", "text": "hello"},
						{"type": "input_image", "image_url": "https://example.com/image.png"},
						{"type": "input_file", "file_id": "file_1"}
					]},
					{"type": "function_call_output", "call_id": "call_1", "output": [{"type": "input_text", "text": "found"}]},
					{"type": "custom_tool_call_output", "call_id": "call_2", "output": [{"type": "input_text", "text": "done"}]},
					{"type": "message", "role": "assistant", "id": "msg_1", "phase": "commentary", "content": [{"type": "output_text", "text": "working"}]},
					{"role": "user", "content": "prompt_cache_breakpoint"},
					{"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque", "summary": [], "future": {"prompt_cache_breakpoint": "keep"}},
					{"type": "function_call", "call_id": "call_1", "arguments": "{\"prompt_cache_breakpoint\":true}"},
					{"type": "function_call_output", "call_id": "call_3", "output": "{\"prompt_cache_breakpoint\":true}"}
				],
				"metadata": {"prompt_cache_breakpoint": "keep"},
				"tools": [{"type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {"prompt_cache_breakpoint": {"type": "boolean"}}}}]
			});
			let expected = body.clone();
			body["prompt_cache_breakpoint"] = hint.clone();
			for item in body["input"].as_array_mut().unwrap().iter_mut().take(6) {
				let field = if item.get("role").is_some() {
					"content"
				} else {
					"output"
				};
				for part in item[field].as_array_mut().unwrap() {
					part["prompt_cache_breakpoint"] = hint.clone();
				}
			}
			let normalized = normalize_request(&serde_json::to_vec(&body).unwrap()).unwrap();
			assert_eq!(
				serde_json::from_slice::<Value>(&normalized).unwrap(),
				expected
			);
			assert_eq!(normalize_request(&normalized).unwrap(), normalized);
		}
	}

	fn buffered(sse: &str) -> BufferedResponse {
		let (parts, _) = http::Response::builder()
			.status(200)
			.header("content-type", "text/event-stream")
			.body(())
			.unwrap()
			.into_parts();
		BufferedResponse {
			parts,
			bytes: Bytes::copy_from_slice(sse.as_bytes()),
		}
	}

	#[test]
	fn stateless_reasoning_extends_include_without_replacing_client_controls() {
		for store in [Value::Null, json!(false), json!(true)] {
			let body = json!({"model": "gpt-test", "input": [], "store": store,
				"instructions": null, "include": ["web_search_call.action.sources"],
				"stream_options": {"reasoning_summary_delivery": "sequential_cutoff"}});
			let actual: Value =
				serde_json::from_slice(&normalize_request(&body.to_string().into_bytes()).unwrap())
					.unwrap();
			assert_eq!(actual["store"], false);
			assert_eq!(actual["instructions"], "");
			assert_eq!(actual["input"], body["input"]);
			assert_eq!(actual["stream_options"], body["stream_options"]);
			assert_eq!(
				actual["include"],
				json!([
					"web_search_call.action.sources",
					"reasoning.encrypted_content"
				])
			);
		}
	}

	#[test]
	fn unary_terminal_responses_preserve_all_fields() {
		for status in ["completed", "incomplete", "failed"] {
			for output in [None, Some(Value::Null), Some(json!([]))] {
				let mut response = json!({"status": status, "error": {"code": "test"}, "future": true});
				if let Some(output) = output {
					response["output"] = output;
				}
				let mut buffered = buffered(&format!(
					": heartbeat\n\ndata: {{\"type\":\"response.created\"}}\n\nevent: response.{status}\ndata: {}\n\n",
					json!({"type": format!("response.{status}"), "response": response})
				));
				normalize_unary_response(&mut buffered).unwrap();
				response["output"] = json!([]);
				assert_eq!(
					serde_json::from_slice::<Value>(&buffered.bytes).unwrap(),
					response
				);
				assert_eq!(buffered.parts.headers["content-type"], "application/json");
			}
		}
	}

	#[test]
	fn unary_recovers_completed_items_in_output_index_order() {
		let items = json!([
			{"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque", "future": {"keep": true}},
			{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "NONSTREAM_OK", "annotations": []}]},
			{"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup", "arguments": "{}", "status": "completed"}
		]);
		for status in ["completed", "incomplete", "failed"] {
			for output in [
				None,
				Some(Value::Null),
				Some(json!([])),
				Some(json!([{"type": "message", "id": "terminal-wins"}])),
			] {
				let mut terminal =
					json!({"id": "resp_1", "status": status, "usage": {"output_tokens": 7}, "future": true});
				if let Some(output) = output {
					terminal["output"] = output;
				}
				let mut expected = terminal.clone();
				if expected["output"].as_array().is_none_or(Vec::is_empty) {
					expected["output"] = items.clone();
				}
				let mut sse = String::new();
				// Out-of-order events and a repeated index must not reorder or duplicate output.
				for index in [2, 1, 0, 1] {
					sse.push_str(&format!(
						"data: {}\n\n",
						json!({"type": "response.output_item.done", "output_index": index, "item": items[index]})
					));
				}
				sse.push_str(&format!(
					"data: {}\n\n",
					json!({"type": format!("response.{status}"), "response": terminal})
				));
				let mut buffered = buffered(&sse);
				normalize_unary_response(&mut buffered).unwrap();
				assert_eq!(
					serde_json::from_slice::<Value>(&buffered.bytes).unwrap(),
					expected
				);
			}
		}
	}

	#[test]
	fn unary_errors_and_truncation_do_not_become_success() {
		for sse in [
			"",
			"data: [DONE]\n\n",
			"data: {\"type\":\"response.created\"}\n\n",
			"data: {\"type\":\"response.completed\"}\n\n",
			"data: invalid-json\n\n",
			"data: {\"type\":\"response.output_item.done\",\"item\":{}}\n\n",
			"data: {\"type\":\"response.output_item.done\",\"output_index\":0}\n\n",
			"data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{}}\n\n",
		] {
			assert!(normalize_unary_response(&mut buffered(sse)).is_err());
		}
		let mut buffered =
			buffered("data: {\"type\":\"error\",\"code\":\"server_error\",\"message\":\"failed\"}\n\n");
		normalize_unary_response(&mut buffered).unwrap();
		assert_eq!(buffered.parts.status, http::StatusCode::BAD_GATEWAY);
		assert_eq!(
			serde_json::from_slice::<Value>(&buffered.bytes).unwrap()["error"]["code"],
			"server_error"
		);
	}

	#[test]
	fn overload_inspector_waits_through_empty_prelude_and_split_chunks() {
		let mut inspector = OverloadStreamInspector::new();
		assert_eq!(
			inspector
				.push(b": keepalive\n\nevent: response.created\ndata: {\"type\":\"response.created\"}\n\n"),
			StreamEventDisposition::Continue
		);
		assert_eq!(
			inspector.push(b"event: error\ndata: {\"type\":\"error\",\"error\":{\"code\":"),
			StreamEventDisposition::Continue
		);
		assert_eq!(
			inspector.push(b"\"server_is_overloaded\",\"message\":\"busy\"}}\n\n"),
			StreamEventDisposition::Overloaded
		);

		let mut named_error = OverloadStreamInspector::new();
		assert_eq!(
			named_error.push(b"event: error\ndata: {\"error\":{\"code\":\"server_is_overloaded\"}}\n\n"),
			StreamEventDisposition::Overloaded
		);
	}

	#[test]
	fn tool_output_and_later_overload_are_not_retryable() {
		let mut inspector = OverloadStreamInspector::new();
		assert_eq!(
			inspector.push(b"data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"function_call\",\"name\":\"lookup\"}}\n\n"),
			StreamEventDisposition::OutputStarted
		);
		assert_eq!(
			inspector
				.push(b"data: {\"type\":\"error\",\"error\":{\"code\":\"server_is_overloaded\"}}\n\n"),
			StreamEventDisposition::OutputStarted
		);
	}

	#[test]
	fn terminal_and_unknown_events_end_lookahead() {
		for data in [
			br#"{"type":"response.completed"}"#.as_slice(),
			br#"{"type":"response.future_output_type"}"#.as_slice(),
			b"[DONE]".as_slice(),
			b"not-json".as_slice(),
		] {
			assert_eq!(
				classify_named_stream_event(None, data),
				StreamEventDisposition::OutputStarted
			);
		}
		assert_eq!(
			classify_named_stream_event(
				None,
				br#"{"type":"response.failed","response":{"error":{"code":"invalid_request"}}}"#
			),
			StreamEventDisposition::OutputStarted
		);
		let mut inspector = OverloadStreamInspector::new();
		let mut oversized = b"data: ".to_vec();
		oversized.extend(vec![b'x'; OVERLOAD_PREFIX_LIMIT]);
		assert_eq!(
			inspector.push(&oversized),
			StreamEventDisposition::OutputStarted
		);
	}
}
