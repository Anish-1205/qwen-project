from __future__ import annotations

import random

from .common import ToolError


def roll_die(sides: int = 6) -> dict:
    return {"value": random.randint(1, sides), "sides": sides}


def random_number(minimum: int, maximum: int) -> dict:
    if minimum > maximum:
        raise ToolError("invalid_range", "minimum must be less than or equal to maximum")
    return {"value": random.randint(minimum, maximum), "minimum": minimum, "maximum": maximum}
