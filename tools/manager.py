"""Parse, validate, and execute untrusted model-requested tools."""
from __future__ import annotations
import json
import logging
import math
import re
import copy
from datetime import date, datetime, time
from dataclasses import dataclass
from typing import Any
from . import config
from .common import ToolError, validate_public_url
from .registry import ToolDefinition, ToolRegistry, build_default_registry
LOG = logging.getLogger(__name__)

class ToolValidationError(ValueError): pass

@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str = "tool_call_1"

@dataclass(frozen=True)
class ToolExecutionResult:
    call: ToolCall
    ok: bool
    data: dict[str, Any] | None = None
    error_details: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None
    validated_arguments: dict[str, Any] | None = None
    @property
    def result(self):
        if self.data and "value" in self.data: return self.data["value"]
        return self.data
    @property
    def error(self) -> str | None: return self.error_details.get("message") if self.error_details else None
    def payload(self) -> dict[str, Any]:
        if self.ok: return {"ok": True, "tool": self.call.name, "data": self.data or {}, "meta": self.meta or {}}
        return {"ok": False, "tool": self.call.name, "error": self.error_details or {"code": "internal_error", "message": "Tool execution failed.", "details": {}}}

class ToolManager:
    def __init__(self, registry: ToolRegistry | None = None) -> None: self.registry = registry or build_default_registry()
    def schemas(self) -> list[dict[str, Any]]: return self.registry.schemas()
    def provenance(self, tool_name: str) -> str:
        definition = self.registry.get(tool_name)
        return definition.provenance if definition is not None else "unknown"
    @staticmethod
    def _strict_json_loads(value: str) -> Any:
        def reject_constant(constant: str):
            raise ValueError(f"non-standard JSON number: {constant}")
        return json.loads(value, parse_constant=reject_constant)
    @staticmethod
    def parse_tool_calls(model_output: str) -> list[ToolCall]:
        text = (model_output or "").strip()
        tagged = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.DOTALL)
        incomplete_tag = text.count("<tool_call>") != text.count("</tool_call>")
        recovered_nested_wrapper = False
        if not tagged and "<tool_call" in text:
            return [ToolCall("__invalid_tool_call__", {"__invalid_arguments__": "[incomplete tool call]"})]
        candidates = tagged or ([text] if text.startswith("{") and text.endswith("}") else [])
        calls: list[ToolCall] = []
        for index, candidate in enumerate(candidates, 1):
            normalized_candidate = candidate.strip()
            while normalized_candidate.startswith("<tool_call>"):
                recovered_nested_wrapper = True
                normalized_candidate = normalized_candidate[len("<tool_call>"):].strip()
            try: payload = ToolManager._strict_json_loads(normalized_candidate)
            except (TypeError, ValueError, json.JSONDecodeError):
                if tagged: calls.append(ToolCall("__invalid_tool_call__", {"__invalid_arguments__": "[malformed JSON]"}, f"tool_call_{index}"))
                continue
            for item in payload if isinstance(payload, list) else [payload]:
                if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip():
                    if tagged: calls.append(ToolCall("__invalid_tool_call__", {"__invalid_arguments__": "[missing tool name]"}, f"tool_call_{index}"))
                    continue
                if not tagged and "arguments" not in item:
                    continue
                arguments = item.get("arguments", {})
                if isinstance(arguments, str):
                    try: arguments = ToolManager._strict_json_loads(arguments)
                    except (TypeError, ValueError, json.JSONDecodeError): arguments = {"__invalid_arguments__": "[malformed JSON]"}
                if not isinstance(arguments, dict): arguments = {"__invalid_arguments__": arguments}
                name = item["name"].strip()
                if len(name) > 128:
                    name = "__invalid_tool_call__"
                    arguments = {"__invalid_arguments__": "[tool name too long]"}
                call_id = str(item.get("id") or f"tool_call_{index}")
                if len(call_id) > 128:
                    call_id = f"tool_call_{index}"
                calls.append(ToolCall(name, arguments, call_id))
        if tagged and incomplete_tag and not recovered_nested_wrapper:
            calls.append(ToolCall("__invalid_tool_call__", {"__invalid_arguments__": "[incomplete tool call]"}, f"tool_call_{len(calls) + 1}"))
        return calls
    @classmethod
    def parse_tool_call(cls, model_output: str) -> ToolCall | None:
        calls = cls.parse_tool_calls(model_output)
        return calls[0] if calls else None
    def execute(self, call: ToolCall) -> ToolExecutionResult:
        validated_call, failure = self.validate_call(call)
        if failure is not None:
            return failure
        return self.execute_validated(validated_call)

    def validate_call(self, call: ToolCall) -> tuple[ToolCall | None, ToolExecutionResult | None]:
        """Validate and normalize without executing the registered function."""
        definition = self.registry.get(call.name)
        if definition is None:
            return None, self._failure(call, "unknown_tool", f"Unknown tool: {call.name}")
        try:
            arguments = self._validate(definition.parameters, call.arguments, "arguments")
            arguments = self._normalize_arguments(call.name, arguments)
            return ToolCall(call.name, arguments, call.call_id), None
        except ToolValidationError as exc:
            return None, self._failure(call, "validation_error", str(exc))
        except ToolError as exc:
            return None, self._failure(call, exc.code, exc.message, exc.details)
        except (ValueError, TypeError, OSError) as exc:
            return None, self._failure(call, "validation_error", str(exc)[:500])

    def execute_validated(self, call: ToolCall) -> ToolExecutionResult:
        """Execute a call returned by ``validate_call``."""
        definition = self.registry.get(call.name)
        if definition is None:
            return self._failure(call, "unknown_tool", f"Unknown tool: {call.name}")
        arguments = call.arguments
        try:
            data = definition.function(**arguments)
            if not isinstance(data, dict): data = {"value": data}
            data = self._json_safe(data)
            data = self._bound_data(data)
            return ToolExecutionResult(call, True, data=data, meta={}, validated_arguments=arguments)
        except ToolError as exc: return self._failure(call, exc.code, exc.message, exc.details, arguments)
        except (ValueError, TypeError, OSError) as exc: return self._failure(call, "execution_error", str(exc)[:500], validated_arguments=arguments)
        except Exception:
            LOG.exception("Unexpected tool failure for %s", call.name)
            return self._failure(call, "internal_error", "The tool failed unexpectedly.", validated_arguments=arguments)
    @staticmethod
    def _failure(
        call: ToolCall,
        code: str,
        message: str,
        details: dict | None = None,
        validated_arguments: dict[str, Any] | None = None,
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            call,
            False,
            error_details={"code": code, "message": message, "details": details or {}},
            validated_arguments=validated_arguments,
        )
    @staticmethod
    def _normalize_arguments(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        output = dict(arguments)
        path_fields = ["path"] if "path" in output else []
        if "paths" in output:
            path_fields.append("paths")
        for field in path_fields:
            values = output[field] if isinstance(output[field], list) else [output[field]]
            normalized = []
            for value in values:
                cleaned = value.strip()
                if not cleaned or "\x00" in cleaned:
                    raise ToolValidationError(f"arguments.{field} contains a malformed path")
                normalized.append(cleaned)
            output[field] = normalized if isinstance(output[field], list) else normalized[0]
        if tool_name == "fetch_webpage":
            output["url"] = validate_public_url(output["url"].strip())
        elif tool_name == "weather" and "place" in output:
            output["place"] = output["place"].strip()
        elif tool_name == "random_number" and output["minimum"] > output["maximum"]:
            raise ToolValidationError("minimum must be less than or equal to maximum")
        elif tool_name == "analyze_spreadsheet":
            for index, operation in enumerate(output["operations"]):
                if operation["op"] == "filter" and operation["operator"] == "in" and not isinstance(operation["value"], list):
                    raise ToolValidationError(f"arguments.operations[{index}].value must be an array for operator 'in'")
                if operation["op"] == "aggregate" and operation["function"] != "count" and not operation.get("column"):
                    raise ToolValidationError(f"arguments.operations[{index}].column is required for numeric aggregation")
        return output
    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if value is None or type(value) in (bool, int, str):
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise ToolError("invalid_result", "The tool produced a non-finite numeric result.")
            return value
        if isinstance(value, (datetime, date, time)):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(key): cls._json_safe(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(child) for child in value]
        raise ToolError("invalid_result", "The tool produced a value that cannot be returned safely.", {"type": type(value).__name__})
    @classmethod
    def _bound_data(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Bound the complete result data, including metadata, while keeping valid JSON."""
        budget = max(500, config.MAX_RESULT_CHARS - 256)
        initial = json.dumps(data, ensure_ascii=False)
        if len(initial) <= budget:
            return data
        bounded = copy.deepcopy(data)
        def mark_truncated() -> None:
            if isinstance(bounded.get("metadata"), dict):
                bounded["metadata"]["truncated"] = True
            else:
                bounded["truncated"] = True
        mark_truncated()

        def largest_shrinkable(container: Any):
            candidates: list[tuple[int, Any, Any, str]] = []
            dict_candidates: list[tuple[int, Any, Any, str]] = []
            def visit(value: Any, parent: Any = None, key: Any = None, depth: int = 0) -> None:
                if isinstance(value, str) and len(value) > 32:
                    candidates.append((len(value), parent, key, "string"))
                elif isinstance(value, list):
                    if len(value) > 1:
                        candidates.append((len(json.dumps(value, ensure_ascii=False)), parent, key, "list"))
                    for index, child in enumerate(value):
                        visit(child, value, index, depth + 1)
                elif isinstance(value, dict):
                    if depth > 0 and len(value) > 4:
                        dict_candidates.append((len(json.dumps(value, ensure_ascii=False)), parent, key, "dict"))
                    for child_key, child in list(value.items()):
                        visit(child, value, child_key, depth + 1)
            visit(container)
            pool = candidates or dict_candidates
            return max(pool, default=None, key=lambda item: item[0])

        for _ in range(128):
            rendered = json.dumps(bounded, ensure_ascii=False)
            if len(rendered) <= budget:
                mark_truncated()
                if len(json.dumps(bounded, ensure_ascii=False)) <= budget:
                    return bounded
            candidate = largest_shrinkable(bounded)
            if candidate is None:
                break
            _, parent, key, kind = candidate
            value = parent[key]
            if kind == "string":
                excess = len(rendered) - budget
                parent[key] = value[: max(16, len(value) - excess - 16)]
            elif kind == "list":
                parent[key] = value[: max(1, len(value) // 2)]
            else:
                keys = [child_key for child_key in value if child_key != "truncated"]
                for child_key in keys[max(1, len(keys) // 2):]:
                    value.pop(child_key, None)
        return {"truncated": True, "original_characters": len(initial)}
    @classmethod
    def _validate(cls, schema: dict, value: Any, path: str) -> Any:
        if path == "arguments" and (not isinstance(value, dict) or "__invalid_arguments__" in value): raise ToolValidationError("arguments must be a JSON object")
        if "oneOf" in schema:
            successes, errors = [], []
            for branch in schema["oneOf"]:
                try: successes.append(cls._validate(branch, value, path))
                except ToolValidationError as exc: errors.append(str(exc))
            if not successes: raise ToolValidationError(errors[0] if errors else f"{path} does not match a supported schema")
            if len(successes) != 1: raise ToolValidationError(f"{path} matches more than one schema")
            return successes[0]
        expected = schema.get("type")
        if type(value) is float and not math.isfinite(value):
            raise ToolValidationError(f"{path} must be a finite number")
        valid = {"object": isinstance(value, dict), "array": isinstance(value, list), "string": isinstance(value, str), "integer": type(value) is int,
                 "number": type(value) in (int, float), "boolean": type(value) is bool}
        if expected and not valid.get(expected, False): raise ToolValidationError(f"{path} must be a {expected}")
        if "enum" in schema and value not in schema["enum"]: raise ToolValidationError(f"{path} has an invalid value")
        if expected == "object":
            props = schema.get("properties", {})
            unexpected = set(value) - set(props)
            if unexpected and schema.get("additionalProperties") is False: raise ToolValidationError(f"unexpected argument(s) at {path}: {', '.join(sorted(unexpected))}")
            missing = set(schema.get("required", [])) - set(value)
            if missing: raise ToolValidationError(f"missing required argument(s) at {path}: {', '.join(sorted(missing))}")
            output = {}
            for name, child in props.items():
                if name in value: output[name] = cls._validate(child, value[name], f"{path}.{name}")
                elif "default" in child: output[name] = child["default"]
            return output
        if expected == "array":
            if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", float("inf")): raise ToolValidationError(f"{path} has an invalid number of items")
            return [cls._validate(schema.get("items", {}), item, f"{path}[{i}]") for i, item in enumerate(value)]
        if expected == "string" and not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", float("inf")): raise ToolValidationError(f"{path} has an invalid length")
        if type(value) in (int, float) and (value < schema.get("minimum", value) or value > schema.get("maximum", value)): raise ToolValidationError(f"{path} is outside the allowed range")
        return value
