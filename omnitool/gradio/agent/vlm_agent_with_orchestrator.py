import json
from collections.abc import Callable
from typing import cast, Callable
import uuid
from PIL import Image, ImageDraw
import base64
from io import BytesIO
import copy
from pathlib import Path
from datetime import datetime
from anthropic import APIResponse
from anthropic.types import ToolResultBlockParam
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock, BetaMessageParam, BetaUsage

from agent.llm_utils.oaiclient import run_oai_interleaved
from agent.llm_utils.groqclient import run_groq_interleaved
from agent.llm_utils.ollamaclient import run_ollama_interleaved
from agent.llm_utils.utils import is_image_path
import time
import re
import os
OUTPUT_DIR = "./tmp/outputs"
TRACE_LOG_PATH = "./uploads/decision_trace.jsonl"
ORCHESTRATOR_LEDGER_PROMPT = """
Recall we are working on the following request:

{task}

To make progress on the request, please answer the following questions, including necessary reasoning:

    - Is the request fully satisfied? (True if complete, or False if the original request has yet to be SUCCESSFULLY and FULLY addressed)
    - Are we in a loop where we are repeating the same requests and / or getting the same responses as before? Loops can span multiple turns, and can include repeated actions like scrolling up or down more than a handful of times.
    - Are we making forward progress? (True if just starting, or recent messages are adding value. False if recent messages show evidence of being stuck in a loop or if there is evidence of significant barriers to success such as the inability to read from a required file)
    - What instruction or question would you give in order to complete the task? 

Please output an answer in pure JSON format according to the following schema. The JSON object must be parsable as-is. DO NOT OUTPUT ANYTHING OTHER THAN JSON, AND DO NOT DEVIATE FROM THIS SCHEMA:

    {{
       "is_request_satisfied": {{
            "reason": string,
            "answer": boolean
        }},
        "is_in_loop": {{
            "reason": string,
            "answer": boolean
        }},
        "is_progress_being_made": {{
            "reason": string,
            "answer": boolean
        }},
        "instruction_or_question": {{
            "reason": string,
            "answer": string
        }}
    }}
"""

def extract_data(input_string, data_type):
    # Regular expression to extract content starting from '```python' until the end if there are no closing backticks
    pattern = f"```{data_type}" + r"(.*?)(```|$)"
    # Extract content
    # re.DOTALL allows '.' to match newlines as well
    matches = re.findall(pattern, input_string, re.DOTALL)
    # Return the first match if exists, trimming whitespace and ignoring potential closing backticks
    return matches[0][0].strip() if matches else input_string

class VLMOrchestratedAgent:
    def __init__(
        self,
        model: str, 
        provider: str, 
        api_key: str,
        output_callback: Callable, 
        api_response_callback: Callable,
        max_tokens: int = 4096,
        only_n_most_recent_images: int | None = None,
        print_usage: bool = True,
        save_folder: str = None,
        ollama_model_name: str = "",
        ollama_json_model_name: str = "",
        ollama_base_url: str = "http://localhost:11434",
        ollama_supports_vision: bool = True,
    ):
        if model == "omniparser + gpt-4o" or model == "omniparser + gpt-4o-orchestrated":
            self.model = "gpt-4o-2024-11-20"
        elif model == "omniparser + R1" or model == "omniparser + R1-orchestrated":
            self.model = "deepseek-r1-distill-llama-70b"
        elif model == "omniparser + qwen2.5vl" or model == "omniparser + qwen2.5vl-orchestrated":
            self.model = "qwen2.5-vl-72b-instruct"
        elif model == "omniparser + o1" or model == "omniparser + o1-orchestrated":
            self.model = "o1"
        elif model == "omniparser + o3-mini" or model == "omniparser + o3-mini-orchestrated":
            self.model = "o3-mini"
        elif model == "omniparser + ollama" or model == "omniparser + ollama-orchestrated":
            self.model = ollama_model_name if ollama_model_name else "llama3.2-vision"
        else:
            raise ValueError(f"Model {model} not supported")
        
        self.ollama_base_url = ollama_base_url
        self.ollama_supports_vision = ollama_supports_vision
        self.ollama_json_model_name = (ollama_json_model_name or "").strip()
        self.is_ollama = ("ollama" in model)

        self.provider = provider
        self.api_key = api_key
        self.api_response_callback = api_response_callback
        self.max_tokens = max_tokens
        self.only_n_most_recent_images = only_n_most_recent_images
        self.output_callback = output_callback
        self.save_folder = save_folder
        if self.save_folder:
            os.makedirs(self.save_folder, exist_ok=True)
        
        self.print_usage = print_usage
        self.total_token_usage = 0
        self.total_cost = 0
        self.step_count = 0
        self.plan, self.ledger = None, None
        self.completion_guard_enabled = True
        self.task_objective = ""
        self.action_history: list[dict] = []
        self.trace_log_path = Path(TRACE_LOG_PATH)
        self.trace_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.system = ''

    def _extract_task_text(self, messages: list) -> str:
        if isinstance(getattr(self, "_task", None), str) and self._task.strip():
            return self._task
        if not messages:
            return ""
        first = messages[0]
        content = first.get("content", "") if isinstance(first, dict) else ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if hasattr(item, "text"):
                    parts.append(str(item.text))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        if hasattr(content, "text"):
            return str(content.text)
        return str(content)

    def _short(self, value, limit: int = 2000) -> str:
        text = str(value)
        if len(text) <= limit:
            return text
        return text[:limit] + "...(truncated)"

    def _debug_log(self, stage: str, payload: dict):
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "agent": "vlm_orchestrated",
            "step": self.step_count,
            "stage": stage,
            **payload,
        }
        log_line = json.dumps(event, ensure_ascii=True, default=str)
        print(f"[decision_trace] {log_line}")
        with open(self.trace_log_path, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")

    def _build_screen_context(self, parsed_screen: dict, max_items: int = 200) -> str:
        parsed_list = parsed_screen.get("parsed_content_list", []) if isinstance(parsed_screen, dict) else []
        lines = []
        for idx, item in enumerate(parsed_list[:max_items]):
            if isinstance(item, dict):
                parts = []
                for key in ("text", "content", "label", "caption", "type"):
                    value = item.get(key)
                    if value:
                        parts.append(f"{key}={value}")
                line = ", ".join(parts) if parts else str(item)
            else:
                line = str(item)
            lines.append(f"ID {idx}: {line}")
        return "\n".join(lines)

    def _build_action_history_context(self, max_items: int = 20) -> str:
        if not self.action_history:
            return "[]"
        return json.dumps(self.action_history[-max_items:], ensure_ascii=True)

    def _build_json_planner_prompt(self, vision_response: str) -> str:
        skills = (
            "Skill 1 - Goal Alignment: prioritize actions that directly progress the final objective.\n"
            "Skill 2 - State Tracking: avoid repeating recent actions unless state changed.\n"
            "Skill 3 - Grounding: only use Box ID values that exist in detected IDs.\n"
            "Skill 4 - Interaction Choice: use double_click for desktop icons/files; left_click for buttons/links.\n"
            "Skill 5 - Safety: if target cannot be grounded to an ID, return Next Action as None."
        )
        return (
            "You are an action planner for computer-use automation.\n"
            "Pick exactly one best next action.\n\n"
            f"FINAL OBJECTIVE:\n{self.task_objective}\n\n"
            f"ACTIONS ALREADY DONE (JSON list):\n{self._build_action_history_context()}\n\n"
            f"AVAILABLE DETECTED TARGETS (ID inventory):\n{self._build_screen_context(self._latest_parsed_screen)}\n\n"
            f"VISION MODEL PROPOSAL:\n{vision_response}\n\n"
            f"{skills}\n\n"
            "Output STRICT JSON only with keys:\n"
            "{\"Reasoning\": string, \"Next Action\": string, \"Box ID\": integer(optional), \"value\": string(optional)}\n"
            "Allowed Next Action values: type, left_click, right_click, double_click, hover, scroll_up, scroll_down, wait, None.\n"
            "Rules:\n"
            "- If action needs a target, Box ID must be an integer from the detected IDs.\n"
            "- Never output label strings for Box ID.\n"
            "- Only include value for type action.\n"
            "- If no valid grounded action exists, return Next Action as None.\n"
        )

    def _run_model(self, messages: list, system: str, use_json_bridge: bool = True, stage: str = "action"):
        if self.is_ollama:
            vision_response, token_usage = run_ollama_interleaved(
                messages=messages,
                system=system,
                model_name=self.model,
                base_url=self.ollama_base_url,
                max_tokens=self.max_tokens,
                temperature=0.1,
                supports_vision=self.ollama_supports_vision,
            )
            self.total_token_usage += token_usage
            self.total_cost += 0
            self._debug_log(
                "vision_model_output",
                {
                    "stage_name": stage,
                    "model": self.model,
                    "token_usage": token_usage,
                    "response": self._short(vision_response),
                },
            )
            if (
                use_json_bridge
                and
                self.ollama_json_model_name
                and self.ollama_json_model_name != self.model
                and not str(vision_response).startswith("Error from Ollama:")
            ):
                json_prompt = self._build_json_planner_prompt(str(vision_response))
                self._debug_log(
                    "json_planner_input",
                    {
                        "stage_name": stage,
                        "model": self.ollama_json_model_name,
                        "prompt": self._short(json_prompt, 4000),
                    },
                )
                json_response, json_tokens = run_ollama_interleaved(
                    messages=json_prompt,
                    system="You are a strict JSON planner for GUI action selection.",
                    model_name=self.ollama_json_model_name,
                    base_url=self.ollama_base_url,
                    max_tokens=self.max_tokens,
                    temperature=0,
                    supports_vision=False,
                )
                self.total_token_usage += json_tokens
                self._debug_log(
                    "json_planner_output",
                    {
                        "stage_name": stage,
                        "model": self.ollama_json_model_name,
                        "token_usage": json_tokens,
                        "response": self._short(json_response),
                    },
                )
                if not str(json_response).startswith("Error from Ollama:"):
                    return json_response
            return vision_response
        if "gpt" in self.model or "o1" in self.model or "o3-mini" in self.model:
            vlm_response, token_usage = run_oai_interleaved(
                messages=messages,
                system=system,
                model_name=self.model,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                provider_base_url="https://api.openai.com/v1",
                temperature=0,
            )
            self.total_token_usage += token_usage
            if 'gpt' in self.model:
                self.total_cost += (token_usage * 2.5 / 1000000)
            elif 'o1' in self.model:
                self.total_cost += (token_usage * 15 / 1000000)
            elif 'o3-mini' in self.model:
                self.total_cost += (token_usage * 1.1 / 1000000)
            return vlm_response
        if "r1" in self.model:
            vlm_response, token_usage = run_groq_interleaved(
                messages=messages,
                system=system,
                model_name=self.model,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
            )
            self.total_token_usage += token_usage
            self.total_cost += (token_usage * 0.99 / 1000000)
            return vlm_response
        if "qwen" in self.model:
            vlm_response, token_usage = run_oai_interleaved(
                messages=messages,
                system=system,
                model_name=self.model,
                api_key=self.api_key,
                max_tokens=min(2048, self.max_tokens),
                provider_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                temperature=0,
            )
            self.total_token_usage += token_usage
            self.total_cost += (token_usage * 2.2 / 1000000)
            return vlm_response
        raise ValueError(f"Model {self.model} not supported")

    def _resolve_box_id(self, box_id_value, parsed_screen: dict):
        if isinstance(box_id_value, int):
            return box_id_value
        if isinstance(box_id_value, str):
            raw = box_id_value.strip()
            if raw.isdigit():
                return int(raw)
        else:
            raw = str(box_id_value or "").strip()
            if raw.isdigit():
                return int(raw)

        parsed_list = parsed_screen.get("parsed_content_list", [])
        query = raw.lower()
        if not query:
            return None
        match = re.search(r"\b(?:id|box)\s*[:#]?\s*(\d+)\b", query)
        if match:
            return int(match.group(1))

        query_tokens = [t for t in re.split(r"[^a-z0-9]+", query) if t]
        if not query_tokens:
            return None

        best_idx = None
        best_score = -1.0
        for idx, item in enumerate(parsed_list):
            hay = ""
            if isinstance(item, dict):
                parts = []
                for key in ("text", "content", "label", "caption", "type"):
                    value = item.get(key)
                    if value:
                        parts.append(str(value))
                hay = " ".join(parts).lower()
            else:
                hay = str(item).lower()
            if not hay:
                continue
            hits = sum(1 for token in query_tokens if token in hay)
            if hits <= 0:
                continue
            score = hits / max(1, len(query_tokens))
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_score >= 0.5:
            return best_idx
        return None

    def _normalize_action_json(self, vlm_response_json: dict, parsed_screen: dict):
        normalized = dict(vlm_response_json)
        action = str(normalized.get("Next Action", "None")).strip().lower().replace(" ", "_").replace("-", "_")
        action_map = {
            "click": "left_click",
            "leftclick": "left_click",
            "doubleclick": "double_click",
            "rightclick": "right_click",
            "scrollup": "scroll_up",
            "scrolldown": "scroll_down",
            "none": "None",
        }
        action = action_map.get(action, action)
        if action not in {"type", "left_click", "right_click", "double_click", "hover", "scroll_up", "scroll_down", "wait", "None"}:
            action = "None"
        normalized["Next Action"] = action if action != "none" else "None"

        if "Box ID" in normalized:
            resolved = self._resolve_box_id(normalized.get("Box ID"), parsed_screen)
            if resolved is None:
                normalized["Reasoning"] = (
                    f"{normalized.get('Reasoning', '')}\n"
                    "System note: box id was not resolvable; action converted to None."
                ).strip()
                normalized["Next Action"] = "None"
                normalized.pop("Box ID", None)
                normalized.pop("value", None)
            else:
                normalized["Box ID"] = int(resolved)

        if normalized.get("Next Action") == "left_click" and isinstance(normalized.get("Box ID"), int):
            box_idx = normalized["Box ID"]
            parsed_list = parsed_screen.get("parsed_content_list", [])
            if 0 <= box_idx < len(parsed_list):
                item = parsed_list[box_idx]
                text_blob = str(item).lower()
                icon_keywords = ("icon", "recycle bin", "this pc", "folder", "shortcut")
                if any(token in text_blob for token in icon_keywords):
                    normalized["Next Action"] = "double_click"

        return normalized

    def _completion_check_override(
        self,
        vlm_response_json: dict,
        planner_messages: list,
        system: str,
        screenshot_uuid: str,
    ) -> dict:
        if not self.completion_guard_enabled:
            return vlm_response_json
        if str(vlm_response_json.get("Next Action", "")).strip() != "None":
            return vlm_response_json

        task_text = self._extract_task_text(planner_messages)
        verifier_prompt = (
            f"Original task:\n{task_text}\n\n"
            "You previously returned Next Action as None. "
            "Check the current screenshot and decide if the task is truly complete.\n"
            "Return strict JSON only with this schema:\n"
            "{"
            "\"is_task_complete\": boolean, "
            "\"reason\": string, "
            "\"next_action\": \"type|left_click|right_click|double_click|hover|scroll_up|scroll_down|wait|None\", "
            "\"box_id\": number|null, "
            "\"value\": string|null"
            "}\n"
            "If task is not complete, next_action must not be None."
        )
        verify_messages = list(planner_messages)
        verify_messages.append(
            {
                "role": "user",
                "content": [
                    verifier_prompt,
                    f"{OUTPUT_DIR}/screenshot_{screenshot_uuid}.png",
                ],
            }
        )
        verify_response = self._run_model(
            messages=verify_messages,
            system=system,
            use_json_bridge=False,
            stage="completion_check",
        )
        verify_json_str = extract_data(verify_response, "json")
        try:
            verify_json = json.loads(verify_json_str)
        except Exception:
            return vlm_response_json

        is_complete = bool(verify_json.get("is_task_complete", True))
        next_action = str(verify_json.get("next_action", "None")).strip()
        if is_complete or next_action == "None":
            return vlm_response_json

        override = dict(vlm_response_json)
        override["Reasoning"] = (
            f"{vlm_response_json.get('Reasoning', '')}\n"
            f"Completion-check override: {verify_json.get('reason', 'Task not complete')}"
        ).strip()
        override["Next Action"] = next_action
        if verify_json.get("box_id") is not None:
            override["Box ID"] = verify_json.get("box_id")
        if verify_json.get("value") is not None:
            override["value"] = verify_json.get("value")
        self.output_callback("<i>Completion check: task not complete, continuing with another action.</i>")
        return override
    
    def __call__(self, messages: list, parsed_screen: list[str, list, dict]):
        self._latest_parsed_screen = parsed_screen
        if not self.task_objective:
            self.task_objective = self._extract_task_text(messages)
        if self.step_count == 0:
            plan = self._initialize_task(messages)
            self.output_callback(f'-- Plan: {plan} --', )
            # update messages with the plan
            messages.append({"role": "assistant", "content": plan})
        else:
            updated_ledger = self._update_ledger(messages)
            self.output_callback(
                f'<details>'
                f'  <summary><strong>Task Progress Ledger (click to expand)</strong></summary>'
                f'  <div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-top: 5px;">'
                f'    <pre>{updated_ledger}</pre>'
                f'  </div>'
                f'</details>',
            )
            # update messages with the ledger
            messages.append({"role": "assistant", "content": updated_ledger})
            self.ledger = updated_ledger

        self.step_count += 1
        # save the image to the output folder
        with open(f"{self.save_folder}/screenshot_{self.step_count}.png", "wb") as f:
            f.write(base64.b64decode(parsed_screen['original_screenshot_base64']))
        with open(f"{self.save_folder}/som_screenshot_{self.step_count}.png", "wb") as f:
            f.write(base64.b64decode(parsed_screen['som_image_base64']))

        latency_omniparser = parsed_screen['latency']
        screen_info = str(parsed_screen['screen_info'])
        screenshot_uuid = parsed_screen['screenshot_uuid']
        screen_width, screen_height = parsed_screen['width'], parsed_screen['height']

        boxids_and_labels = parsed_screen["screen_info"]
        system = self._get_system_prompt(boxids_and_labels)

        # drop looping actions msg, byte image etc
        planner_messages = messages
        _remove_som_images(planner_messages)
        _maybe_filter_to_n_most_recent_images(planner_messages, self.only_n_most_recent_images)

        # Ensure screenshots go into a user message (not an assistant message).
        # The orchestrator appends plan/ledger as assistant messages, so the last
        # message may not be from the user.
        last_is_user = (
            isinstance(planner_messages[-1], dict)
            and "user" in str(planner_messages[-1].get("role", "")).lower()
        )
        if last_is_user:
            if not isinstance(planner_messages[-1]["content"], list):
                planner_messages[-1]["content"] = [planner_messages[-1]["content"]]
            planner_messages[-1]["content"].append(f"{OUTPUT_DIR}/screenshot_{screenshot_uuid}.png")
            planner_messages[-1]["content"].append(f"{OUTPUT_DIR}/screenshot_som_{screenshot_uuid}.png")
        else:
            planner_messages.append({
                "role": "user",
                "content": [
                    "Here is the current screenshot. What is the next action?",
                    f"{OUTPUT_DIR}/screenshot_{screenshot_uuid}.png",
                    f"{OUTPUT_DIR}/screenshot_som_{screenshot_uuid}.png",
                ],
            })

        start = time.time()
        before_tokens = self.total_token_usage
        vlm_response = self._run_model(messages=planner_messages, system=system, use_json_bridge=True, stage="action")
        used_tokens = self.total_token_usage - before_tokens
        if self.is_ollama:
            print(f"ollama token usage: {used_tokens}")
        elif "gpt" in self.model or "o1" in self.model or "o3-mini" in self.model:
            print(f"oai token usage: {used_tokens}")
        elif "r1" in self.model:
            print(f"groq token usage: {used_tokens}")
        elif "qwen" in self.model:
            print(f"qwen token usage: {used_tokens}")
        latency_vlm = time.time() - start
        
        # Update step counter with both latencies
        self.output_callback(f'<i>Step {self.step_count} | OmniParser: {latency_omniparser:.2f}s | LLM: {latency_vlm:.2f}s</i>', )

        print(f"{vlm_response}")
        
        if self.print_usage:
            print(f"Total token so far: {self.total_token_usage}. Total cost so far: $USD{self.total_cost:.5f}")
        
        vlm_response_json = extract_data(vlm_response, "json")
        try:
            vlm_response_json = json.loads(vlm_response_json)
        except json.JSONDecodeError:
            # Model didn't return valid JSON — try to find a JSON object in the raw response
            json_match = re.search(r'\{[^{}]*\}', vlm_response, re.DOTALL)
            if json_match:
                try:
                    vlm_response_json = json.loads(json_match.group())
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse JSON from model response, treating as 'None' action")
                    vlm_response_json = {"Reasoning": vlm_response, "Next Action": "None"}
            else:
                print(f"Warning: No JSON found in model response, treating as 'None' action")
                vlm_response_json = {"Reasoning": vlm_response, "Next Action": "None"}
        vlm_response_json = self._completion_check_override(
            vlm_response_json=vlm_response_json,
            planner_messages=planner_messages,
            system=system,
            screenshot_uuid=screenshot_uuid,
        )
        vlm_response_json = self._normalize_action_json(vlm_response_json, parsed_screen)

        img_to_show_base64 = parsed_screen["som_image_base64"]
        if "Box ID" in vlm_response_json:
            try:
                bbox = parsed_screen["parsed_content_list"][int(vlm_response_json["Box ID"])]["bbox"]
                vlm_response_json["box_centroid_coordinate"] = [int((bbox[0] + bbox[2]) / 2 * screen_width), int((bbox[1] + bbox[3]) / 2 * screen_height)]
                self._debug_log(
                    "action_grounding",
                    {
                        "action": vlm_response_json.get("Next Action"),
                        "box_id": vlm_response_json.get("Box ID"),
                        "bbox_norm": bbox,
                        "coordinate_px": vlm_response_json["box_centroid_coordinate"],
                        "target": self._short(parsed_screen["parsed_content_list"][int(vlm_response_json["Box ID"])]),
                    },
                )
                img_to_show_data = base64.b64decode(img_to_show_base64)
                img_to_show = Image.open(BytesIO(img_to_show_data))

                draw = ImageDraw.Draw(img_to_show)
                x, y = vlm_response_json["box_centroid_coordinate"] 
                radius = 10
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red')
                draw.ellipse((x - radius*3, y - radius*3, x + radius*3, y + radius*3), fill=None, outline='red', width=2)

                buffered = BytesIO()
                img_to_show.save(buffered, format="PNG")
                img_to_show_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            except:
                print(f"Error parsing: {vlm_response_json}")
                pass
        self._debug_log(
            "action_selected",
            {
                "next_action": vlm_response_json.get("Next Action"),
                "box_id": vlm_response_json.get("Box ID"),
                "value": vlm_response_json.get("value"),
                "reasoning": self._short(vlm_response_json.get("Reasoning", ""), 1200),
            },
        )
        self.action_history.append(
            {
                "step": self.step_count,
                "action": vlm_response_json.get("Next Action"),
                "box_id": vlm_response_json.get("Box ID"),
                "value": vlm_response_json.get("value"),
            }
        )
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]
        self.output_callback(f'<img src="data:image/png;base64,{img_to_show_base64}">', )
        
        # Display screen info in a collapsible dropdown
        self.output_callback(
            f'<details>'
            f'  <summary><strong>Parsed Screen Elements (click to expand)</strong></summary>'
            f'  <div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-top: 5px;">'
            f'    <pre>{screen_info}</pre>'
            f'  </div>'
            f'</details>',
        )
        
        vlm_plan_str = ""
        for key, value in vlm_response_json.items():
            if key == "Reasoning":
                vlm_plan_str += f'{value}'
            else:
                vlm_plan_str += f'\n{key}: {value}'

        # construct the response so that anthropicExcutor can execute the tool
        response_content = [BetaTextBlock(text=vlm_plan_str, type='text')]
        if 'box_centroid_coordinate' in vlm_response_json:
            move_cursor_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}',
                                            input={'action': 'mouse_move', 'coordinate': vlm_response_json["box_centroid_coordinate"]},
                                            name='computer', type='tool_use')
            response_content.append(move_cursor_block)

        if vlm_response_json["Next Action"] == "None":
            print("Task paused/completed.")
        elif vlm_response_json["Next Action"] == "type":
            sim_content_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}',
                                        input={'action': vlm_response_json["Next Action"], 'text': vlm_response_json["value"]},
                                        name='computer', type='tool_use')
            response_content.append(sim_content_block)
        else:
            sim_content_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}',
                                            input={'action': vlm_response_json["Next Action"]},
                                            name='computer', type='tool_use')
            response_content.append(sim_content_block)
        response_message = BetaMessage(id=f'toolu_{uuid.uuid4()}', content=response_content, model='', role='assistant', type='message', stop_reason='tool_use', usage=BetaUsage(input_tokens=0, output_tokens=0))

        # save the intermediate step trajectory to the save folder
        step_trajectory = {
            "screenshot_path": f"{self.save_folder}/screenshot_{self.step_count}.png",
            "som_screenshot_path": f"{self.save_folder}/som_screenshot_{self.step_count}.png",
            "screen_info": screen_info,
            "latency_omniparser": latency_omniparser,
            "latency_vlm": latency_vlm,
            "vlm_response_json": vlm_response_json,
            'ledger': self.ledger,
        }
        with open(f"{self.save_folder}/trajectory.json", "a") as f:
            f.write(json.dumps(step_trajectory))
            f.write("\n")

        return response_message, vlm_response_json

    def _api_response_callback(self, response: APIResponse):
        self.api_response_callback(response)

    def _get_system_prompt(self, screen_info: str = ""):
        main_section = f"""
You are using a Windows device.
You are able to use a mouse and keyboard to interact with the computer based on the given task and screenshot.
You can only interact with the desktop GUI (no terminal or application menu access).

You may be given some history plan and actions, this is the response from the previous loop.
You should carefully consider your plan base on the task, screenshot, and history actions.

Here is the list of all detected bounding boxes by IDs on the screen and their description:{screen_info}

Your available "Next Action" only include:
- type: types a string of text.
- left_click: move mouse to box id and left clicks.
- right_click: move mouse to box id and right clicks.
- double_click: move mouse to box id and double clicks.
- hover: move mouse to box id.
- scroll_up: scrolls the screen up to view previous content.
- scroll_down: scrolls the screen down, when the desired button is not visible, or you need to see more content. 
- wait: waits for 1 second for the device to load or respond.

Based on the visual information from the screenshot image and the detected bounding boxes, please determine the next action, the Box ID you should operate on (if action is one of 'type', 'hover', 'scroll_up', 'scroll_down', 'wait', there should be no Box ID field), and the value (if the action is 'type') in order to complete the task.

Output format:
```json
{{
    "Reasoning": str, # describe what is in the current screen, taking into account the history, then describe your step-by-step thoughts on how to achieve the task, choose one action from available actions at a time.
    "Next Action": "action_type, action description" | "None" # one action at a time, describe it in short and precisely. 
    "Box ID": n,
    "value": "xxx" # only provide value field if the action is type, else don't include value key
}}
```

One Example:
```json
{{  
    "Reasoning": "The current screen shows google result of amazon, in previous action I have searched amazon on google. Then I need to click on the first search results to go to amazon.com.",
    "Next Action": "left_click",
    "Box ID": m
}}
```

Another Example:
```json
{{
    "Reasoning": "The current screen shows the front page of amazon. There is no previous action. Therefore I need to type "Apple watch" in the search bar.",
    "Next Action": "type",
    "Box ID": n,
    "value": "Apple watch"
}}
```

Another Example:
```json
{{
    "Reasoning": "The current screen does not show 'submit' button, I need to scroll down to see if the button is available.",
    "Next Action": "scroll_down",
}}
```

IMPORTANT NOTES:
1. You should only give a single action at a time.

"""
        thinking_model = "r1" in self.model
        if not thinking_model:
            main_section += """
2. You should give an analysis to the current screen, and reflect on what has been done by looking at the history, then describe your step-by-step thoughts on how to achieve the task.

"""
        else:
            main_section += """
2. In <think> XML tags give an analysis to the current screen, and reflect on what has been done by looking at the history, then describe your step-by-step thoughts on how to achieve the task. In <output> XML tags put the next action prediction JSON.

"""
        main_section += """
3. Attach the next action prediction in the "Next Action".
4. You should not include other actions, such as keyboard shortcuts.
5. When the task is completed, don't complete additional actions. You should say "Next Action": "None" in the json field.
6. The tasks involve buying multiple products or navigating through multiple pages. You should break it into subgoals and complete each subgoal one by one in the order of the instructions.
7. avoid choosing the same action/elements multiple times in a row, if it happens, reflect to yourself, what may have gone wrong, and predict a different action.
8. If you are prompted with login information page or captcha page, or you think it need user's permission to do the next action, you should say "Next Action": "None" in the json field.
9. On Windows, desktop icons and files in Windows Explorer require double_click to open, NOT left_click. Only use left_click for buttons, links, taskbar items, and menu items. ALWAYS use double_click when opening desktop icons or files.
"""

    def _initialize_task(self, messages: list):
        self._task = messages[0]["content"]
        # make a plan
        plan_prompt = self._get_plan_prompt(self._task)
        input_message = copy.deepcopy(messages)
        input_message.append({"role": "user", "content": plan_prompt})
        if self.is_ollama:
            vlm_response, token_usage = run_ollama_interleaved(
                messages=input_message,
                system="",
                model_name=self.model,
                base_url=self.ollama_base_url,
                max_tokens=self.max_tokens,
                temperature=0.1,
                supports_vision=self.ollama_supports_vision,
            )
        else:
            vlm_response, token_usage = run_oai_interleaved(
                messages=input_message,
                system="",
                model_name=self.model,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                provider_base_url="https://api.openai.com/v1",
                temperature=0,
            )
        plan = extract_data(vlm_response, "json")
        
        # Create a filename with timestamp
        plan_filename = f"plan.json"
        plan_path = os.path.join(self.save_folder, plan_filename)
        
        # Save the plan to a file
        try:
            with open(plan_path, "w") as f:
                f.write(plan)
            print(f"Plan successfully saved to {plan_path}")
        except Exception as e:
            print(f"Error saving plan to {plan_path}: {str(e)}")
        
        return plan

    def _update_ledger(self, messages):
        # tobe implemented
        # update the ledger with the current task and plan
        # return the updated ledger
        update_ledger_prompt = ORCHESTRATOR_LEDGER_PROMPT.format(task=self._task)
        input_message = copy.deepcopy(messages)
        input_message.append({"role": "user", "content": update_ledger_prompt})
        if self.is_ollama:
            vlm_response, token_usage = run_ollama_interleaved(
                messages=input_message,
                system="",
                model_name=self.model,
                base_url=self.ollama_base_url,
                max_tokens=self.max_tokens,
                temperature=0.1,
                supports_vision=self.ollama_supports_vision,
            )
        else:
            vlm_response, token_usage = run_oai_interleaved(
                messages=input_message,
                system="",
                model_name=self.model,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                provider_base_url="https://api.openai.com/v1",
                temperature=0,
            )
        updated_ledger = extract_data(vlm_response, "json")
        return updated_ledger
    
    def _get_plan_prompt(self, task):
        plan_prompt = f"""
        please devise a short bullet-point plan for addressing the original user task: {task}
        You should write your plan in a json dict, e.g:```json
{{
'step 1': xxx,
'step 2': xxxx,
...
}}```
        Now start your answer directly.
        """
        return plan_prompt

def _remove_som_images(messages):
    for msg in messages:
        msg_content = msg["content"]
        if isinstance(msg_content, list):
            msg["content"] = [
                cnt for cnt in msg_content 
                if not (isinstance(cnt, str) and 'som' in cnt and is_image_path(cnt))
            ]


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int = 10,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place
    """
    if images_to_keep is None:
        return messages

    total_images = 0
    for msg in messages:
        for cnt in msg.get("content", []):
            if isinstance(cnt, str) and is_image_path(cnt):
                total_images += 1
            elif isinstance(cnt, dict) and cnt.get("type") == "tool_result":
                for content in cnt.get("content", []):
                    if isinstance(content, dict) and content.get("type") == "image":
                        total_images += 1

    images_to_remove = total_images - images_to_keep
    
    for msg in messages:
        msg_content = msg["content"]
        if isinstance(msg_content, list):
            new_content = []
            for cnt in msg_content:
                # Remove images from SOM or screenshot as needed
                if isinstance(cnt, str) and is_image_path(cnt):
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                # VLM shouldn't use anthropic screenshot tool so shouldn't have these but in case it does, remove as needed
                elif isinstance(cnt, dict) and cnt.get("type") == "tool_result":
                    new_tool_result_content = []
                    for tool_result_entry in cnt.get("content", []):
                        if isinstance(tool_result_entry, dict) and tool_result_entry.get("type") == "image":
                            if images_to_remove > 0:
                                images_to_remove -= 1
                                continue
                        new_tool_result_content.append(tool_result_entry)
                    cnt["content"] = new_tool_result_content
                # Append fixed content to current message's content list
                new_content.append(cnt)
            msg["content"] = new_content
