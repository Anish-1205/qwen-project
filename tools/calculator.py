from __future__ import annotations

import ast
import math
import operator
from statistics import mean

from . import config
from .common import ToolError

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

def _bounded_number(value: int | float) -> int | float:
    if type(value) not in (int, float):
        raise ToolError("numeric_domain_error", "The expression must produce a real number.")
    if type(value) is float and not math.isfinite(value):
        raise ToolError("numeric_overflow", "The result must be finite.")
    if type(value) is int and value.bit_length() > config.CALCULATOR_MAX_RESULT_BITS:
        raise ToolError("numeric_overflow", "The numeric result is too large to calculate safely.")
    return value


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return _bounded_number(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _bounded_number(_UNARY[type(node.op)](_evaluate(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > config.CALCULATOR_MAX_ABS_EXPONENT:
            raise ToolError("invalid_expression", f"Exponents must be between -{config.CALCULATOR_MAX_ABS_EXPONENT} and {config.CALCULATOR_MAX_ABS_EXPONENT}.")
        if isinstance(node.op, ast.Pow) and type(left) is int and type(right) is int and right > 0:
            if max(1, abs(left).bit_length()) * right > config.CALCULATOR_MAX_RESULT_BITS:
                raise ToolError("numeric_overflow", "The exponentiation result would be too large to calculate safely.")
        try:
            return _bounded_number(_BINARY[type(node.op)](left, right))
        except (ArithmeticError, OverflowError) as exc:
            raise ToolError("arithmetic_error", str(exc) or "Arithmetic operation failed.") from exc
    raise ToolError("invalid_expression", "Expression contains unsupported syntax.")


def calculator(expression: str | None = None, aggregate: str | None = None, values: list[int | float] | None = None) -> int | float:
    arithmetic_mode = expression is not None
    aggregate_mode = aggregate is not None or values is not None
    if arithmetic_mode == aggregate_mode:
        raise ToolError("ambiguous_mode", "Provide either expression, or aggregate with values, but not both.")
    if arithmetic_mode:
        source = expression.strip()
        if not source or len(source) > config.CALCULATOR_MAX_EXPRESSION_LENGTH:
            raise ToolError("invalid_expression", f"Expression must contain 1 to {config.CALCULATOR_MAX_EXPRESSION_LENGTH} characters.")
        try:
            result = _evaluate(ast.parse(source, mode="eval"))
        except SyntaxError as exc:
            raise ToolError("invalid_expression", "Expression is not valid arithmetic.") from exc
    else:
        if aggregate not in {"sum", "min", "max", "mean", "count"}:
            raise ToolError("invalid_aggregate", "Aggregate must be sum, min, max, mean, or count.")
        if not isinstance(values, list) or len(values) > config.CALCULATOR_MAX_VALUES or any(type(v) not in (int, float) for v in values):
            raise ToolError("invalid_values", "Values must be a bounded list of numbers.")
        if aggregate != "count" and not values:
            raise ToolError("invalid_values", "Values must not be empty for this aggregate.")
        funcs = {"sum": sum, "min": min, "max": max, "mean": mean, "count": len}
        result = funcs[aggregate](values)
    return _bounded_number(result)
