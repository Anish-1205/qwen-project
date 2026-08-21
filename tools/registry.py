"""The sole executable allowlist and Qwen-facing schema registry."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
from . import config
from .calculator import calculator
from .directory_listing import list_directory
from .file_reader import read_file
from .random_tools import random_number, roll_die
from .spreadsheet import analyze_spreadsheet
from .weather import weather
from .web_fetch import fetch_webpage

@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    function: Callable[..., Any]
    parameters: dict[str, Any]
    def as_qwen_schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}

class ToolRegistry:
    def __init__(self) -> None: self._tools: dict[str, ToolDefinition] = {}
    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._tools: raise ValueError(f"tool is already registered: {definition.name}")
        self._tools[definition.name] = definition
    def get(self, name: str) -> ToolDefinition | None: return self._tools.get(name)
    def schemas(self) -> list[dict[str, Any]]: return [item.as_qwen_schema() for item in self._tools.values()]

def _object(properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": properties, "required": required or [], "additionalProperties": False}

def build_default_registry() -> ToolRegistry:
    r = ToolRegistry()
    integer = {"type": "integer", "minimum": -1_000_000_000, "maximum": 1_000_000_000}
    r.register(ToolDefinition("roll_die", "Roll a die.", roll_die, _object({"sides": {"type": "integer", "minimum": 2, "maximum": 1_000_000, "default": 6}})))
    r.register(ToolDefinition("random_number", "Return a random integer in an inclusive range.", random_number, _object({"minimum": integer, "maximum": integer}, ["minimum", "maximum"])))
    r.register(ToolDefinition("calculator", "Evaluate safe arithmetic or aggregate a supplied numeric list.", calculator, {"oneOf": [
        _object({"expression": {"type": "string", "minLength": 1, "maxLength": config.CALCULATOR_MAX_EXPRESSION_LENGTH}}, ["expression"]),
        _object({"aggregate": {"type": "string", "enum": ["sum", "min", "max", "mean", "count"]}, "values": {"type": "array", "items": {"type": "number"}, "maxItems": config.CALCULATOR_MAX_VALUES}}, ["aggregate", "values"])]}))
    r.register(ToolDefinition("fetch_webpage", "Fetch an HTTP(S) page and return bounded readable text.", fetch_webpage, _object({"url": {"type": "string", "minLength": 8, "maxLength": config.WEB_MAX_URL_CHARS}}, ["url"])))
    r.register(ToolDefinition("weather", "Get current weather and a short forecast by place or coordinates.", weather, {"oneOf": [
        _object({"place": {"type": "string", "minLength": 1, "maxLength": 200}, "forecast_days": {"type": "integer", "minimum": 1, "maximum": config.WEATHER_MAX_FORECAST_DAYS, "default": 3}}, ["place"]),
        _object({"latitude": {"type": "number", "minimum": -90, "maximum": 90}, "longitude": {"type": "number", "minimum": -180, "maximum": 180}, "forecast_days": {"type": "integer", "minimum": 1, "maximum": config.WEATHER_MAX_FORECAST_DAYS, "default": 3}}, ["latitude", "longitude"])]}))
    r.register(ToolDefinition("read_file", "Read one local TXT, CSV, JSON, DOCX, PDF, or XLSX file without ingestion.", read_file, _object({"path": {"type": "string", "minLength": 1, "maxLength": 4096}}, ["path"])))
    r.register(ToolDefinition("list_directory", "List a local directory with bounded optional recursion and filtering.", list_directory, _object({
        "path": {"type": "string", "minLength": 1, "maxLength": 4096}, "recursive": {"type": "boolean", "default": False},
        "max_depth": {"type": "integer", "minimum": 0, "maximum": config.DIRECTORY_MAX_DEPTH, "default": min(1, config.DIRECTORY_MAX_DEPTH)}, "extensions": {"type": "array", "items": {"type": "string", "maxLength": 20}, "maxItems": 50},
        "pattern": {"type": "string", "minLength": 1, "maxLength": 200}}, ["path"])))
    operation_schemas = [
        _object({"op": {"type": "string", "enum": ["select"]}, "columns": {"type": "array", "items": {"type": "string"}, "minItems": 1}}, ["op", "columns"]),
        _object({"op": {"type": "string", "enum": ["filter"]}, "column": {"type": "string", "minLength": 1}, "operator": {"type": "string", "enum": ["eq", "ne", "gt", "gte", "lt", "lte", "contains", "in"]}, "value": {}}, ["op", "column", "operator", "value"]),
        _object({"op": {"type": "string", "enum": ["group_by"]}, "columns": {"type": "array", "items": {"type": "string"}, "minItems": 1}}, ["op", "columns"]),
        _object({"op": {"type": "string", "enum": ["aggregate"]}, "function": {"type": "string", "enum": ["sum", "min", "max", "mean", "count"]}, "column": {"type": "string", "minLength": 1}, "as": {"type": "string", "minLength": 1}}, ["op", "function"]),
        _object({"op": {"type": "string", "enum": ["sort"]}, "column": {"type": "string"}, "direction": {"type": "string", "enum": ["asc", "desc"], "default": "asc"}}, ["op", "column"]),
        _object({"op": {"type": "string", "enum": ["limit"]}, "count": {"type": "integer", "minimum": 1, "maximum": config.SPREADSHEET_MAX_RETURNED_ROWS}}, ["op", "count"]),
    ]
    r.register(ToolDefinition("analyze_spreadsheet", "Declaratively filter, group, aggregate, sort, and select CSV/XLSX data.", analyze_spreadsheet, _object({
        "paths": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 4096}, "minItems": 1, "maxItems": 100},
        "sheet": {"type": "string", "minLength": 1, "maxLength": 200}, "include_source": {"type": "boolean", "default": False},
        "operations": {"type": "array", "items": {"oneOf": operation_schemas}, "maxItems": 20}}, ["paths", "operations"])))
    return r
