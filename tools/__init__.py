"""Stateless, registry-allowlisted general-purpose tools."""
from .calculator import calculator
from .currency_exchange import currency_exchange
from .directory_listing import list_directory
from .file_reader import read_file
from .manager import ToolCall, ToolExecutionResult, ToolManager
from .random_tools import random_number, roll_die
from .registry import ToolDefinition, ToolRegistry, build_default_registry
from .spreadsheet import analyze_spreadsheet
from .weather import weather
from .web_fetch import fetch_webpage
from .web_search import search_web
__all__ = ["ToolCall", "ToolDefinition", "ToolExecutionResult", "ToolManager", "ToolRegistry", "build_default_registry",
           "calculator", "currency_exchange", "random_number", "roll_die", "fetch_webpage", "search_web", "weather", "read_file", "analyze_spreadsheet", "list_directory"]
