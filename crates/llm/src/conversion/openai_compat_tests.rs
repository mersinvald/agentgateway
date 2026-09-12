use bytes::Bytes;
use serde_json::{Value, json};

use super::{from_responses, to_responses};

fn translate(input: &[u8]) -> Value {
	let response = to_responses::translate_response(&Bytes::copy_from_slice(input), "input-model")
		.expect("Chat Completions response should translate");
	serde_json::from_slice(&response.serialize().expect("response should serialize"))
		.expect("translated response should be JSON")
}

#[test]
fn buffered_failed_response_preserves_error() {
	let mut input: Value =
		serde_json::from_slice(include_bytes!("../tests/response/completions/basic.json")).unwrap();
	input["choices"][0]["finish_reason"] = json!("content_filter");
	let input = serde_json::to_vec(&input).unwrap();

	let response = translate(&input);

	assert_eq!(response["status"], json!("failed"));
	assert_eq!(
		response["error"],
		json!({"code": "content_filter", "message": "Content filtered"})
	);
}

fn codex_request() -> crate::types::responses::Request {
	serde_json::from_slice(include_bytes!(
		"../tests/requests/responses/codex-tools.json"
	))
	.unwrap()
}

fn chunk(delta: Value, finish: Value) -> Value {
	json!({"id": "chatcmpl_test", "object": "chat.completion.chunk", "created": 1,
		"model": "moonshotai/Kimi-K3", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
}

async fn stream(chunks: Vec<Value>, done: bool, limit: usize) -> Vec<Value> {
	use http_body_util::BodyExt;
	let (_, tools) = from_responses::translate_request_with_context(&codex_request()).unwrap();
	let mut input: String = chunks
		.into_iter()
		.map(|c| format!("data: {c}\n\n"))
		.collect();
	if done {
		input.push_str("data: [DONE]\n\n");
	}
	let body = to_responses::translate_stream_with_context(
		axum_core::body::Body::from(input),
		limit,
		Default::default(),
		Default::default(),
		tools,
	);
	let bytes = body.collect().await.unwrap().to_bytes();
	String::from_utf8(bytes.to_vec())
		.unwrap()
		.split("\n\n")
		.filter(|s| !s.is_empty())
		.map(|frame| {
			let mut lines = frame.lines();
			let event = lines.next().unwrap().strip_prefix("event: ").unwrap();
			let data: Value =
				serde_json::from_str(lines.next().unwrap().strip_prefix("data: ").unwrap()).unwrap();
			assert_eq!(
				event, data["type"],
				"SSE event names must match their JSON type"
			);
			data
		})
		.collect()
}

#[test]
fn custom_tool_history_uses_call_id_and_json_object_input() {
	let (request, _) = from_responses::translate_request_with_context(&codex_request()).unwrap();
	let request = serde_json::to_value(request).unwrap();
	assert_eq!(
		request["messages"][2]["content"],
		"I'll update the file and check the directory."
	);
	assert_eq!(request["messages"][2]["tool_calls"][0]["id"], "call_patch");
	assert_eq!(request["messages"][3]["tool_call_id"], "call_patch");
	assert_eq!(request["messages"][4]["tool_call_id"], "call_exec");
	let args: Value = serde_json::from_str(
		request["messages"][2]["tool_calls"][0]["function"]["arguments"]
			.as_str()
			.unwrap(),
	)
	.unwrap();
	assert!(args["input"].as_str().unwrap().contains("+привет\n"));
	let names: Vec<_> = request["tools"]
		.as_array()
		.unwrap()
		.iter()
		.map(|t| t["function"]["name"].as_str().unwrap())
		.collect();
	assert_eq!(
		names.iter().collect::<std::collections::HashSet<_>>().len(),
		names.len()
	);
}

#[test]
fn custom_tool_unary_response_restores_namespace_and_raw_input() {
	let (_, tools) = from_responses::translate_request_with_context(&codex_request()).unwrap();
	let input = "*** Begin Patch\n+\"привет\\世界\"\n*** End Patch";
	let bytes = Bytes::from(
		serde_json::to_vec(&json!({
			"id": "chatcmpl_test", "object": "chat.completion", "created": 1, "model": "kimi",
			"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {"role": "assistant",
				"tool_calls": [{"type": "function", "id": "call_upstream", "function": {
					"name": "agw_tool_2", "arguments": json!({"input": input}).to_string()
				}}]}}]
		}))
		.unwrap(),
	);
	let result = to_responses::translate_response_with_context(&bytes, "kimi", &tools).unwrap();
	let result: Value = serde_json::from_slice(&result.serialize().unwrap()).unwrap();
	let call = &result["output"][0];
	assert_eq!(call["type"], "custom_tool_call");
	assert_eq!(call["namespace"], "functions");
	assert_eq!(call["name"], "apply_patch");
	assert_eq!(call["call_id"], "call_upstream");
	assert_eq!(call["input"], input);
	insta::assert_json_snapshot!("codex_custom_tool_response", call);
}

#[tokio::test]
async fn custom_tool_stream_handles_parallel_calls_split_names_and_json_escapes() {
	let input = "*** Begin Patch\n+\"привет\\世界\"\n*** End Patch";
	let args = json!({"input": input}).to_string();
	let mut chunks = vec![chunk(json!({"content": "Checking"}), Value::Null)];
	chunks.push(chunk(json!({"tool_calls": [{"index": 1, "id": "call_exec", "function": {"name": "agw_", "arguments": "{\"cmd\":"}}, {"index": 0, "id": "call_patch", "function": {"name": "agw_tool_2"}}]}), Value::Null));
	for character in args.chars() {
		chunks.push(chunk(
			json!({"tool_calls": [{"index": 0, "function": {"arguments": character.to_string()}}]}),
			Value::Null,
		));
	}
	chunks.push(chunk(
		json!({"tool_calls": [{"index": 1, "function": {"name": "tool_3", "arguments": "\"pwd\"}"}}]}),
		json!("tool_calls"),
	));
	chunks.push(json!({"id": "chatcmpl_test", "object": "chat.completion.chunk", "created": 1,
		"model": "kimi", "choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}));
	let events = stream(chunks, true, 1024 * 1024).await;
	for (i, event) in events.iter().enumerate() {
		assert_eq!(event["sequence_number"], i + 1);
	}
	let final_response = &events.last().unwrap()["response"];
	assert_eq!(final_response["status"], "completed");
	assert_eq!(final_response["usage"]["total_tokens"], 15);
	assert_eq!(
		final_response["output"][0]["content"][0]["text"],
		"Checking"
	);
	assert_eq!(final_response["output"][1]["input"], input);
	assert_eq!(final_response["output"][1]["call_id"], "call_patch");
	assert_eq!(final_response["output"][2]["name"], "exec_command");
	assert_eq!(final_response["output"][2]["namespace"], "functions");
	assert_eq!(final_response["output"][2]["call_id"], "call_exec");
	let delta: String = events
		.iter()
		.filter(|e| e["type"] == "response.custom_tool_call_input.delta")
		.map(|e| e["delta"].as_str().unwrap())
		.collect();
	assert_eq!(delta, input);
	let done_items: Vec<_> = events
		.iter()
		.filter(|e| e["type"] == "response.output_item.done")
		.map(|e| e["item"].clone())
		.collect();
	assert_eq!(json!(done_items), final_response["output"]);
}

#[tokio::test]
async fn stream_requires_finish_reason_and_preserves_text_without_logging() {
	for done in [true, false] {
		let events = stream(
			vec![chunk(json!({"content": "Hello"}), json!("stop"))],
			done,
			4096,
		)
		.await;
		assert_eq!(
			events.last().unwrap()["response"]["output"][0]["content"][0]["text"],
			"Hello"
		);
		let events = stream(
			vec![chunk(json!({"content": "Truncated"}), Value::Null)],
			done,
			4096,
		)
		.await;
		assert_eq!(events.last().unwrap()["type"], "response.failed");
		assert!(!events.iter().any(|e| e["type"] == "response.completed"));
	}
}

#[tokio::test]
async fn invalid_custom_arguments_fail_instead_of_executing_json() {
	let events = stream(
		vec![chunk(
			json!({"tool_calls": [{"index": 0, "id": "call_bad",
		"function": {"name": "agw_tool_2", "arguments": "{\"input\": 42}"}}]}),
			json!("tool_calls"),
		)],
		true,
		4096,
	)
	.await;
	assert_eq!(events.last().unwrap()["type"], "response.failed");
	assert!(
		!events
			.iter()
			.any(|e| e["type"] == "response.output_item.done")
	);
}

#[test]
fn unsupported_hosted_tools_and_server_state_are_rejected() {
	for extra in [
		json!({"tools": [{"type": "web_search"}]}),
		json!({"previous_response_id": "resp_old"}),
	] {
		let mut request = json!({"model": "kimi", "input": "hi"});
		request
			.as_object_mut()
			.unwrap()
			.extend(extra.as_object().unwrap().clone());
		let request = serde_json::from_value(request).unwrap();
		assert!(from_responses::translate_request(&request).is_err());
	}
}

#[tokio::test]
async fn reasoning_survives_tool_call_replay() {
	let events = stream(
		vec![
			chunk(
				json!({"reasoning_content": "Inspect the file before editing."}),
				Value::Null,
			),
			chunk(
				json!({"tool_calls": [{"index": 0, "id": "call_reasoning", "function": {
			"name": "agw_tool_3", "arguments": "{\"cmd\":\"pwd\"}"}}]}),
				json!("tool_calls"),
			),
		],
		true,
		4096,
	)
	.await;
	let output = events.last().unwrap()["response"]["output"]
		.as_array()
		.unwrap();
	assert_eq!(output[0]["type"], "reasoning");
	let mut history = output.clone();
	history.push(
		json!({"type": "function_call_output", "call_id": "call_reasoning", "output": "/workspace"}),
	);
	let request = serde_json::from_value(json!({"model": "kimi", "input": history})).unwrap();
	let replay = from_responses::translate_request(&request).unwrap();
	let replay = serde_json::to_value(replay).unwrap();
	assert_eq!(
		replay["messages"][0]["reasoning_content"],
		"Inspect the file before editing."
	);
	assert_eq!(
		replay["messages"][0]["tool_calls"][0]["id"],
		"call_reasoning"
	);
	assert_eq!(replay["messages"][1]["tool_call_id"], "call_reasoning");
}

#[tokio::test]
async fn accumulated_stream_is_bounded_and_incomplete_tools_are_not_completed() {
	let events = stream(
		vec![
			chunk(json!({"content": "a".repeat(1500)}), Value::Null),
			chunk(json!({"content": "a".repeat(1500)}), Value::Null),
			chunk(json!({"content": "a".repeat(1500)}), json!("stop")),
		],
		true,
		4096,
	)
	.await;
	assert_eq!(events.last().unwrap()["type"], "response.failed");
	let events = stream(
		vec![chunk(
			json!({"tool_calls": [{"index": 0, "id": "partial", "function": {
		"name": "agw_tool_2", "arguments": "{\"input\":\"unfinished"}}]}),
			json!("length"),
		)],
		true,
		4096,
	)
	.await;
	assert_eq!(events.last().unwrap()["type"], "response.incomplete");
	assert_eq!(
		events.last().unwrap()["response"]["incomplete_details"]["reason"],
		"max_output_tokens"
	);
	assert!(
		!events
			.iter()
			.any(|e| e["type"] == "response.output_item.done")
	);
}
