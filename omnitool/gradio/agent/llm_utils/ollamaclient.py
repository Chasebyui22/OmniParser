import requests
import logging
import os
from .utils import is_image_path, encode_image


def _normalize_base_url(base_url: str) -> str:
    url = (base_url or "").strip().rstrip("/")
    if not url:
        return "http://localhost:11434"
    if "://" not in url:
        return f"http://{url}"
    return url


def _candidate_base_urls(base_url: str) -> list[str]:
    primary = _normalize_base_url(base_url)
    candidates = [primary]

    # 0.0.0.0 is a server bind address, not a client destination.
    if "0.0.0.0" in primary:
        candidates.extend(
            [
                primary.replace("0.0.0.0", "localhost"),
                primary.replace("0.0.0.0", "127.0.0.1"),
            ]
        )

    # In WSL, Windows host is commonly the default gateway.
    try:
        if os.path.exists("/proc/sys/kernel/osrelease"):
            with open("/proc/sys/kernel/osrelease", "r", encoding="utf-8") as f:
                if "microsoft" in f.read().lower():
                    with open("/proc/net/route", "r", encoding="utf-8") as f:
                        for line in f.readlines()[1:]:
                            cols = line.strip().split()
                            if len(cols) > 2 and cols[1] == "00000000":
                                gateway_hex = cols[2]
                                gateway_ip = ".".join(
                                    str(int(gateway_hex[i:i + 2], 16))
                                    for i in range(6, -1, -2)
                                )
                                parsed = requests.utils.urlparse(primary)
                                port = parsed.port or 11434
                                scheme = parsed.scheme or "http"
                                candidates.append(f"{scheme}://{gateway_ip}:{port}")
                                break
    except Exception:
        pass

    for fallback in ("http://localhost:11434", "http://127.0.0.1:11434"):
        if fallback not in candidates:
            candidates.append(fallback)
    # Deduplicate while preserving order.
    deduped = []
    for c in candidates:
        if c not in deduped:
            deduped.append(c)
    return deduped


def _get_first_reachable_base_url(base_url: str, timeout: int = 3) -> str | None:
    for candidate in _candidate_base_urls(base_url):
        try:
            response = requests.get(f"{candidate}/api/tags", timeout=timeout)
            if response.status_code == 200:
                return candidate
        except Exception:
            continue
    return None


def get_ollama_models(base_url: str = "http://localhost:11434") -> list[str]:
    """Fetch available models from a running Ollama instance."""
    normalized = _normalize_base_url(base_url)
    try:
        reachable = _get_first_reachable_base_url(normalized, timeout=3)
        if not reachable:
            logging.warning(f"Could not fetch Ollama models from {normalized}: no reachable Ollama endpoint found")
            return []
        response = requests.get(f"{reachable}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [m["name"] for m in models]
    except Exception as e:
        logging.warning(f"Could not fetch Ollama models from {normalized}: {e}")
    return []


def check_ollama_connection(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama is running and reachable."""
    return _get_first_reachable_base_url(base_url, timeout=3) is not None


def run_ollama_interleaved(
    messages: list,
    system: str,
    model_name: str,
    base_url: str = "http://localhost:11434",
    max_tokens: int = 4096,
    temperature: float = 0.1,
    supports_vision: bool = True,
):
    """
    Run a chat completion through Ollama's OpenAI-compatible API.
    Falls back to text-only mode if vision is not supported by the model.
    """
    normalized_base_url = _normalize_base_url(base_url)
    reachable_base_url = _get_first_reachable_base_url(normalized_base_url, timeout=3)
    if not reachable_base_url:
        error = f"Cannot connect to Ollama at {normalized_base_url}. Is Ollama running?"
        print(error)
        return error, 0
    # Prefer Ollama native chat endpoint for maximum compatibility, especially vision.
    api_url = f"{reachable_base_url}/api/chat"
    headers = {"Content-Type": "application/json"}
    final_messages = [{"role": "system", "content": system or ""}]

    if isinstance(messages, str):
        final_messages.append({"role": "user", "content": messages})
    elif isinstance(messages, list):
        for item in messages:
            role = "user"
            raw_content = item
            if isinstance(item, dict):
                role = str(item.get("role", "user"))
                if "user" in role.lower():
                    role = "user"
                elif "assistant" in role.lower():
                    role = "assistant"
                else:
                    role = "user"
                raw_content = item.get("content", "")

            if isinstance(raw_content, str):
                final_messages.append({"role": role, "content": raw_content or "(no content)"})
                continue

            if not isinstance(raw_content, list):
                raw_content = [raw_content]

            text_parts = []
            images = []
            for cnt in raw_content:
                if hasattr(cnt, "text") and not isinstance(cnt, (str, dict)):
                    cnt_text = cnt.text
                elif isinstance(cnt, str):
                    cnt_text = cnt
                else:
                    cnt_text = str(cnt)
                    if "BetaToolUseBlock" in cnt_text or "ToolUseBlock" in cnt_text:
                        continue

                if isinstance(cnt_text, str) and is_image_path(cnt_text):
                    if supports_vision:
                        try:
                            images.append(encode_image(cnt_text))
                        except Exception as e:
                            print(f"Warning: could not encode image {cnt_text}: {e}")
                    continue

                if cnt_text:
                    text_parts.append(str(cnt_text))

            message_payload = {
                "role": role,
                "content": "\n".join(text_parts) if text_parts else "(no content)",
            }
            if images:
                # Some Ollama vision models accept only one image per message.
                # Keep the latest image to avoid request rejection.
                if len(images) > 1:
                    images = [images[-1]]
                message_payload["images"] = images
            final_messages.append(message_payload)

    payload = {
        "model": model_name,
        "messages": final_messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=300)
        response_json = response.json()

        if "error" in response_json:
            error_obj = response_json["error"]
            error_msg = error_obj.get("message", str(error_obj)) if isinstance(error_obj, dict) else str(error_obj)
            print(f"Ollama API error: {error_msg}")
            return f"Error from Ollama: {error_msg}", 0

        text = (
            response_json.get("message", {}).get("content")
            or response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        token_usage = int(
            response_json.get("eval_count", 0) + response_json.get("prompt_eval_count", 0)
        )

        # Handle thinking models (DeepSeek R1, etc.) that wrap output in <think> tags
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        if "<output>" in text:
            text = text.replace("<output>", "").replace("</output>", "").strip()

        return text, token_usage

    except requests.exceptions.ConnectionError:
        error = f"Cannot connect to Ollama at {reachable_base_url}. Is Ollama running?"
        print(error)
        return error, 0
    except requests.exceptions.Timeout:
        error = f"Ollama request timed out (model: {model_name}). The model may be loading or the request is too complex."
        print(error)
        return error, 0
    except Exception as e:
        # Surface raw response details when JSON decoding fails or payload is rejected.
        try:
            status = response.status_code
            body = response.text[:500]
            print(f"Error in Ollama interleaved: {e}; status={status}; body={body}")
            return f"{e} (status={status})", 0
        except Exception:
            print(f"Error in Ollama interleaved: {e}")
            return str(e), 0
