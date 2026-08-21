"""Stateless, registry-allowlisted general-purpose tools."""
from .calculator import calculator
from .directory_listing import list_directory
from .file_reader import read_file
from .manager import ToolCall, ToolExecutionResult, ToolManager
from .random_tools import random_number, roll_die
from .registry import ToolDefinition, ToolRegistry, build_default_registry
from .spreadsheet import analyze_spreadsheet
from .weather import weather
from .web_fetch import fetch_webpage
__all__ = ["ToolCall", "ToolDefinition", "ToolExecutionResult", "ToolManager", "ToolRegistry", "build_default_registry",
           "calculator", "random_number", "roll_die", "fetch_webpage", "weather", "read_file", "analyze_spreadsheet", "list_directory"]
