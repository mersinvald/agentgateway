use std::collections::BTreeMap;

use agent_core::strng;
use serde_json::{Value, json};

use crate::AIError;

/// Request-local mapping: Chat Completions has neither namespaces nor free-form tools.
/// Keep this with the request, never in a provider-global registry.
#[derive(Debug, Clone, Default)]
pub struct ResponseToolMap(BTreeMap<String, ToolIdentity>);

#[derive(Debug, Clone, PartialEq, Eq)]
struct ToolIdentity {
	name: String,
	namespace: Option<String>,
	custom: bool,
}

fn unsupported(message: &str) -> AIError {
	AIError::UnsupportedConversion(strng::new(message))
}

impl ResponseToolMap {
	fn register(&mut self, name: &str, namespace: Option<&str>, custom: bool) -> String {
		let identity = ToolIdentity {
			name: name.into(),
			namespace: namespace.map(str::to_owned),
			custom,
		};
		if let Some((wire_name, _)) = self.0.iter().find(|(_, value)| **value == identity) {
			return wire_name.clone();
		}
		let valid_name = !name.is_empty()
			&& name.len() <= 64
			&& name
				.bytes()
				.all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-'));
		let wire_name = if namespace.is_none() && valid_name && !self.0.contains_key(name) {
			name.to_owned()
		} else {
			let mut index = self.0.len();
			loop {
				let candidate = format!("agw_tool_{index}");
				if !self.0.contains_key(&candidate) {
					break candidate;
				}
				index += 1;
			}
		};
		self.0.insert(wire_name.clone(), identity);
		wire_name
	}

	fn flatten_tools(
		&mut self,
		tools: &[Value],
		namespace: Option<&str>,
		out: &mut Vec<Value>,
	) -> Result<(), AIError> {
		for tool in tools {
			let kind = tool["type"].as_str().unwrap_or_default();
			if kind == "namespace" {
				if namespace.is_some() {
					return Err(unsupported("nested tool namespaces are not supported"));
				}
				let name = tool["name"]
					.as_str()
					.ok_or_else(|| unsupported("missing namespace name"))?;
				let children = tool["tools"]
					.as_array()
					.ok_or_else(|| unsupported("missing namespace tools"))?;
				self.flatten_tools(children, Some(name), out)?;
				continue;
			}
			if !matches!(kind, "function" | "custom") {
				return Err(unsupported(&format!(
					"Responses tool {kind:?} requires provider-native support; disable it for Chat Completions backends"
				)));
			}
			let name = tool["name"]
				.as_str()
				.ok_or_else(|| unsupported("missing tool name"))?;
			let custom = kind == "custom";
			let wire_name = self.register(name, namespace, custom);
			if out.iter().any(|t| t["name"] == wire_name) {
				return Err(unsupported("duplicate tool name in the same namespace"));
			}
			let mut converted = tool.clone();
			converted["type"] = json!("function");
			converted["name"] = json!(wire_name);
			if custom {
				let mut description = tool["description"].as_str().unwrap_or_default().to_owned();
				description.push_str(
					"\nPass the complete raw tool input in the input string, without Markdown fences.",
				);
				if tool["format"]["type"] == "grammar" {
					// JSON-only providers cannot enforce a CFG. Preserve the grammar as instructions.
					description.push_str(&format!(
						"\nThe input must follow this grammar: {}",
						tool["format"]
					));
				}
				converted["description"] = json!(description);
				converted["parameters"] = json!({
					"type": "object", "properties": {"input": {"type": "string"}},
					"required": ["input"], "additionalProperties": false
				});
				converted["strict"] = json!(true);
			}
			if let Some(namespace) = namespace {
				converted["description"] = json!(format!(
					"Tool: {namespace}.{name}\n{}",
					converted["description"].as_str().unwrap_or_default()
				));
			}
			out.push(converted);
		}
		Ok(())
	}

	pub(super) fn normalize_request(request: &mut Value) -> Result<Self, AIError> {
		for key in ["previous_response_id", "conversation", "prompt"] {
			if !request[key].is_null() {
				return Err(unsupported(&format!(
					"{key} requires provider-side Responses state; send the full input history"
				)));
			}
		}
		let mut map = Self::default();
		if let Some(tools) = request["tools"].as_array() {
			let mut flattened = Vec::new();
			map.flatten_tools(tools, None, &mut flattened)?;
			request["tools"] = json!(flattened);
		}
		if let Some(items) = request["input"].as_array_mut() {
			for item in items {
				match item["type"].as_str() {
					Some("function_call" | "custom_tool_call") => {
						let custom = item["type"] == "custom_tool_call";
						let name = item["name"]
							.as_str()
							.ok_or_else(|| unsupported("missing tool call name"))?;
						let wire_name = map.register(name, item["namespace"].as_str(), custom);
						if custom {
							let input = item["input"]
								.as_str()
								.ok_or_else(|| unsupported("custom tool input must be text"))?;
							item["arguments"] = json!(json!({"input": input}).to_string());
							item.as_object_mut().unwrap().remove("input");
						}
						item["type"] = json!("function_call");
						item["name"] = json!(wire_name);
						item.as_object_mut().unwrap().remove("namespace");
					},
					Some("custom_tool_call_output") => item["type"] = json!("function_call_output"),
					Some("item_reference") => {
						return Err(unsupported(
							"item references require provider-side Responses state",
						));
					},
					_ => {},
				}
			}
		}
		let choice = &mut request["tool_choice"];
		if matches!(choice["type"].as_str(), Some("function" | "custom")) {
			let name = choice["name"]
				.as_str()
				.ok_or_else(|| unsupported("missing tool choice name"))?;
			let namespace = choice["namespace"].as_str();
			let custom = choice["type"] == "custom";
			let wire_name = map
				.0
				.iter()
				.find(|(_, identity)| {
					identity.name == name
						&& identity.namespace.as_deref() == namespace
						&& identity.custom == custom
				})
				.map(|(wire_name, _)| wire_name.clone())
				.ok_or_else(|| unsupported("tool choice does not name a declared tool"))?;
			*choice = json!({"type": "function", "name": wire_name});
		} else if choice.is_object() {
			return Err(unsupported(
				"this tool choice requires provider-native Responses support",
			));
		}
		Ok(map)
	}

	/// Restore the client-visible name, namespace and free-form input after inference.
	pub(super) fn restore_call(&self, item: &mut Value) -> Result<(), AIError> {
		let Some(identity) = item["name"].as_str().and_then(|name| self.0.get(name)) else {
			return Ok(());
		};
		item["name"] = json!(identity.name);
		if let Some(namespace) = &identity.namespace {
			item["namespace"] = json!(namespace);
		}
		if identity.custom {
			let arguments = item["arguments"].as_str().unwrap_or_default();
			let args: Value = serde_json::from_str(arguments).map_err(AIError::ResponseParsing)?;
			let input = args["input"].as_str().ok_or_else(|| {
				AIError::ResponseParsing(<serde_json::Error as serde::de::Error>::custom(
					"custom tool arguments must contain a string input",
				))
			})?;
			item["input"] = json!(input);
			item["type"] = json!("custom_tool_call");
			item.as_object_mut().unwrap().remove("arguments");
		}
		Ok(())
	}
}
