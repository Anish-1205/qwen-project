from __future__ import annotations

import requests

from . import config
from .common import ToolError

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog", 48: "Rime fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle", 56: "Light freezing drizzle", 57: "Heavy freezing drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain", 66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains", 80: "Light rain showers", 81: "Rain showers",
    82: "Heavy rain showers", 85: "Light snow showers", 86: "Heavy snow showers", 95: "Thunderstorm",
    96: "Thunderstorm with hail", 99: "Severe thunderstorm with hail",
}


def _get_json(url: str, params: dict) -> dict:
    try:
        response = requests.get(url, params=params, timeout=config.WEB_REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout as exc:
        raise ToolError("network_timeout", "The weather request timed out.") from exc
    except (requests.RequestException, ValueError) as exc:
        raise ToolError("weather_upstream_error", "The weather service request failed.", {"exception": type(exc).__name__}) from exc
    if not isinstance(payload, dict):
        raise ToolError("weather_upstream_error", "The weather service returned malformed data.")
    return payload


def weather(place: str | None = None, latitude: float | None = None, longitude: float | None = None,
            forecast_days: int = 3) -> dict:
    if place is not None:
        geocoded = _get_json(GEOCODING_URL, {"name": place, "count": 1, "language": "en", "format": "json"})
        results = geocoded.get("results") or []
        if not results:
            raise ToolError("location_not_found", "No matching location was found.", {"place": place})
        location = results[0]
        latitude, longitude = location.get("latitude"), location.get("longitude")
        resolved = {key: location.get(key) for key in ("name", "admin1", "country", "timezone") if location.get(key) is not None}
    else:
        resolved = {"name": "Coordinates"}
    if type(latitude) not in (int, float) or not -90 <= latitude <= 90 or type(longitude) not in (int, float) or not -180 <= longitude <= 180:
        raise ToolError("invalid_coordinates", "Latitude must be -90..90 and longitude must be -180..180.")
    days = min(forecast_days, config.WEATHER_MAX_FORECAST_DAYS)
    payload = _get_json(FORECAST_URL, {
        "latitude": latitude, "longitude": longitude, "forecast_days": days, "timezone": "auto",
        "current": "temperature_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,wind_direction_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
    })
    current, daily = payload.get("current") or {}, payload.get("daily") or {}
    required_current = {"temperature_2m", "precipitation", "weather_code", "wind_speed_10m"}
    dates = daily.get("time")
    required_daily = ("weather_code", "temperature_2m_max", "temperature_2m_min")
    if not required_current.issubset(current) or not isinstance(dates, list) or not dates:
        raise ToolError("incomplete_weather_data", "The weather service returned incomplete data.")
    returned_days = min(days, len(dates))
    if any(not isinstance(daily.get(field), list) or len(daily[field]) < returned_days for field in required_daily):
        raise ToolError("incomplete_weather_data", "The weather service returned incomplete daily forecast data.")
    probability = daily.get("precipitation_probability_max")
    if probability is not None and (not isinstance(probability, list) or len(probability) < returned_days):
        raise ToolError("incomplete_weather_data", "The weather service returned incomplete precipitation data.")
    forecast = []
    for index, date in enumerate(dates[:returned_days]):
        code = daily["weather_code"][index]
        forecast.append({
            "date": date, "condition": WEATHER_CODES.get(code, f"Weather code {code}"),
            "high_c": daily["temperature_2m_max"][index],
            "low_c": daily["temperature_2m_min"][index],
            "precipitation_probability_percent": probability[index] if probability is not None else None,
        })
    code = current.get("weather_code")
    return {"location": {**resolved, "latitude": latitude, "longitude": longitude}, "timezone": payload.get("timezone"),
            "current": {"time": current.get("time"), "temperature_c": current.get("temperature_2m"),
                        "apparent_temperature_c": current.get("apparent_temperature"), "precipitation_mm": current.get("precipitation"),
                        "condition": WEATHER_CODES.get(code, f"Weather code {code}"), "wind_speed_kmh": current.get("wind_speed_10m"),
                        "wind_direction_degrees": current.get("wind_direction_10m")}, "forecast": forecast}
