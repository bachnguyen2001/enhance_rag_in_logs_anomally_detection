"""Prompt, OpenAI-compatible HTTP call, strict JSON parsing and one JSONL cache."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

import httpx


SYSTEM_PROMPT = """You are a system-log anomaly classifier.
Classify one complete HDFS BlockId event sequence as NORMAL or ANOMALY.
Historical references, when provided, are training sequences with their ground-truth labels.
Use each reference's NORMAL or ANOMALY label when comparing it with the query.
Pay attention to missing, additional, repeated, or differently ordered events.
Return valid JSON only."""


def load_env(path):
    """Load simple KEY=VALUE entries without overwriting shell variables."""
    path = Path(path)
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def build_messages(query_text, context_texts):
    rendered = []
    for index, context in enumerate(context_texts):
        if isinstance(context, dict):
            label = context.get("label", "UNKNOWN")
            text = context["text"]
            rendered.append(f"REFERENCE {index + 1} — LABEL: {label}\n{text}")
        else:
            rendered.append(f"REFERENCE {index + 1}:\n{context}")
    references = "\n\n".join(rendered) or "NONE"
    user = (
        f"LABELED HISTORICAL REFERENCES:\n{references}\n\n"
        f"QUERY BLOCK SEQUENCE:\n{query_text}\n\n"
        'Return: {"label":"NORMAL or ANOMALY","reason":"one short sentence"}'
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_prediction(text):
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    elif "{" in text and "}" in text:
        text = text[text.find("{"):text.rfind("}") + 1]
    try:
        value = json.loads(text)
        label = value["label"].strip().upper()
        reason = value.get("reason", "").strip()
        if label in {"NORMAL", "ANOMALY"} and reason:
            return label, reason
    except (ValueError, TypeError, KeyError, AttributeError):
        pass
    return "INVALID_OUTPUT", ""


def messages_to_input(messages):
    parts = []
    for message in messages:
        role = message.get("role", "user").upper()
        parts.append(f"{role}:\n{message.get('content', '')}")
    return "\n\n".join(parts)


def response_content(data, payload_format):
    if payload_format == "input_text":
        for item in data.get("output", []):
            if item.get("type") == "message" and item.get("content"):
                return item["content"]
        return ""
    return data["choices"][0]["message"].get("content") or ""


class ChatLLM:
    def __init__(self, config, cache_path, client=None):
        if not config.get("model") or not config.get("base_url"):
            raise ValueError("Set LLM_BASE_URL and LLM_MODEL in experiments/.env")
        self.config = config
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["cache_key"]] = row["result"]
        self.client = client or httpx.Client(timeout=config.get("timeout_seconds", 90))
        self.last_request_at = None

    def _chat_url(self):
        path = self.config.get("chat_path", "/chat/completions")
        if not path.startswith("/"):
            path = "/" + path
        return self.config["base_url"].rstrip("/") + path

    def close(self):
        self.client.close()

    def _wait_between_requests(self):
        delay = float(self.config.get("request_delay_seconds", 0))
        if self.last_request_at is not None and delay > 0:
            remaining = delay - (time.perf_counter() - self.last_request_at)
            if remaining > 0:
                time.sleep(remaining)

    def _post(self, body, key):
        """Pace requests and retry bounded rate-limit/network/server failures."""
        max_rate_retries = int(self.config.get("rate_limit_retries", 5))
        rate_backoff = float(self.config.get("rate_limit_backoff_seconds", 10))
        max_transient_retries = int(self.config.get("transient_retries", 3))
        transient_backoff = float(self.config.get("transient_backoff_seconds", 5))
        rate_retries = 0
        transient_retries = 0
        while True:
            self._wait_between_requests()
            headers = ({"Authorization": f"Bearer {key}"}
                       if self.config.get("api_key_required", True) and key else {})
            try:
                response = self.client.post(
                    self._chat_url(),
                    headers=headers, json=body,
                )
                self.last_request_at = time.perf_counter()
            except (httpx.TimeoutException, httpx.NetworkError) as error:
                if transient_retries >= max_transient_retries:
                    raise
                transient_retries += 1
                wait_seconds = min(transient_backoff * (2 ** (transient_retries - 1)), 60.0)
                print(
                    f"[LLM] {type(error).__name__}; waiting {wait_seconds:.1f}s before "
                    f"network retry {transient_retries}/{max_transient_retries}...",
                    flush=True,
                )
                time.sleep(wait_seconds)
                continue

            if response.status_code == 429:
                if rate_retries >= max_rate_retries:
                    response.raise_for_status()
                retry_after = response.headers.get("Retry-After")
                try:
                    wait_seconds = float(retry_after) if retry_after else rate_backoff * (2 ** rate_retries)
                except ValueError:
                    wait_seconds = rate_backoff * (2 ** rate_retries)
                rate_retries += 1
                wait_seconds = min(wait_seconds, 120.0)
                print(
                    f"[LLM] rate limit 429; waiting {wait_seconds:.1f}s before "
                    f"retry {rate_retries}/{max_rate_retries}...",
                    flush=True,
                )
                time.sleep(wait_seconds)
                continue

            if response.status_code in {500, 502, 503, 504}:
                if transient_retries >= max_transient_retries:
                    response.raise_for_status()
                transient_retries += 1
                wait_seconds = min(transient_backoff * (2 ** (transient_retries - 1)), 60.0)
                print(
                    f"[LLM] HTTP {response.status_code}; waiting {wait_seconds:.1f}s before "
                    f"server retry {transient_retries}/{max_transient_retries}...",
                    flush=True,
                )
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            return response, rate_retries, transient_retries

    def classify(self, messages):
        payload = {
            "provider": self.config.get("provider", "openai-compatible"),
            "base_url": self.config["base_url"], "model": self.config["model"],
            "chat_path": self.config.get("chat_path", "/chat/completions"),
            "payload_format": self.config.get("payload_format", "chat_completions"),
            "messages": messages, "temperature": self.config.get("temperature"),
            "max_output_tokens": self.config.get("max_output_tokens", 128),
            "output_token_parameter": self.config.get("output_token_parameter", "max_tokens"),
            "json_mode": self.config.get("json_mode", True),
            "reasoning_effort": self.config.get("reasoning_effort"),
            "cache_namespace": self.config.get("cache_namespace", "main"),
        }
        cache_key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if cache_key in self.cache:
            return {**self.cache[cache_key], "cache_hit": True, "latency_seconds": None, "new_api_calls": 0}
        key = os.environ.get(self.config.get("api_key_env", "LLM_API_KEY"), "")
        if self.config.get("api_key_required", True) and not key:
            raise ValueError("Set LLM_API_KEY in experiments/.env")

        conversation, attempts = list(messages), []
        rate_limit_retries = 0
        transient_retries = 0
        start = time.perf_counter()
        label, reason = "INVALID_OUTPUT", ""
        for attempt in range(2):
            payload_format = self.config.get("payload_format", "chat_completions")
            if payload_format == "input_text":
                body = {"model": self.config["model"], "input": messages_to_input(conversation),
                        "max_output_tokens": self.config.get("max_output_tokens", 128)}
            else:
                body = {"model": self.config["model"], "messages": conversation,
                        self.config.get("output_token_parameter", "max_completion_tokens"): self.config.get("max_output_tokens", 128)}
            if self.config.get("temperature") is not None:
                body["temperature"] = self.config["temperature"]
            if payload_format != "input_text" and self.config.get("reasoning_effort") is not None:
                body["reasoning_effort"] = self.config["reasoning_effort"]
            if payload_format != "input_text" and self.config.get("json_mode", True):
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "hdfs_anomaly_classification",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string", "enum": ["NORMAL", "ANOMALY"]},
                                "reason": {"type": "string"},
                            },
                            "required": ["label", "reason"],
                            "additionalProperties": False,
                        },
                    },
                }
            response, rate_retries, network_retries = self._post(body, key)
            rate_limit_retries += rate_retries
            transient_retries += network_retries
            data = response.json()
            content = response_content(data, payload_format)
            usage = data.get("usage", {})
            if payload_format == "input_text" and "stats" in data:
                usage = {"prompt_tokens": data["stats"].get("input_tokens"),
                         "completion_tokens": data["stats"].get("total_output_tokens")}
            attempts.append({"content": content, "usage": usage,
                             "response_model": data.get("model") or data.get("model_instance_id")})
            label, reason = parse_prediction(content)
            if label != "INVALID_OUTPUT":
                break
            if attempt == 0:
                conversation.extend([{"role": "assistant", "content": content}, {"role": "user", "content":
                    'Return valid JSON only: {"label":"NORMAL or ANOMALY","reason":"one short sentence"}.'}])
        prompt_tokens = [item["usage"].get("prompt_tokens") for item in attempts]
        result = {
            "prediction": label, "reason": reason, "attempts": attempts,
            "run_date": datetime.now(timezone.utc).isoformat(),
            "prompt_tokens": sum(prompt_tokens) if all(value is not None for value in prompt_tokens) else None,
            "request_count": len(attempts), "latency_seconds": time.perf_counter() - start,
            "cache_hit": False, "new_api_calls": len(attempts),
            "rate_limit_retries": rate_limit_retries,
            "transient_retries": transient_retries,
        }
        with self.cache_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"cache_key": cache_key, "result": result}, ensure_ascii=False) + "\n")
        self.cache[cache_key] = result
        return result
