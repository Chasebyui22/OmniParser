"""
python app.py --windows_host_url localhost:8006 --omniparser_server_url localhost:8000
"""

import os
from datetime import datetime
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import cast
import argparse
import gradio as gr
from anthropic import APIResponse
from anthropic.types import TextBlock
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock
from anthropic.types.tool_use_block import ToolUseBlock
from loop import (
    APIProvider,
    sampling_loop_sync,
)
from tools import ToolResult
import requests
from requests.exceptions import RequestException
import base64
from urllib.parse import urlparse

CONFIG_DIR = Path("~/.anthropic").expanduser()
API_KEY_FILE = CONFIG_DIR / "api_key"

INTRO_TEXT = '''
OmniParser lets you turn any vision-langauge model into an AI agent. We currently support **OpenAI (4o/o1/o3-mini), DeepSeek (R1), Qwen (2.5VL), Anthropic Computer Use (Sonnet), and Ollama (local models).**

Type a message and press submit to start OmniTool. Press stop to pause, and press the trash icon in the chat to clear the message history.
'''

def parse_arguments():

    parser = argparse.ArgumentParser(description="Gradio App")
    parser.add_argument("--windows_host_url", type=str, default='localhost:8006')
    parser.add_argument("--omniparser_server_url", type=str, default="localhost:8000")
    return parser.parse_args()
args = parse_arguments()


def normalize_ollama_ui_url(raw_url: str) -> str:
    value = (raw_url or "").strip()
    if not value:
        return "http://localhost:11434"
    if "://" not in value:
        value = f"http://{value}"
    parsed = urlparse(value)
    if parsed.hostname == "0.0.0.0":
        host = "localhost"
        port = parsed.port or 11434
        scheme = parsed.scheme or "http"
        return f"{scheme}://{host}:{port}"
    return value.rstrip("/")


def _windows_control_probe_url() -> str:
    """
    OmniBox exposes two endpoints:
    - NoVNC viewer on args.windows_host_url (default port 8006)
    - Control server on port 5000 (probe endpoint: /probe)

    When Gradio runs in WSL and OmniBox runs on Windows (Docker Desktop),
    'localhost' may not point at the same machine. Use the host from
    --windows_host_url and force port 5000 for the probe.
    """
    raw = (args.windows_host_url or "").strip()
    if "://" not in raw:
        raw = f"http://{raw}"
    parsed = urlparse(raw)
    host = parsed.hostname or "localhost"
    return f"http://{host}:5000/probe"


class Sender(StrEnum):
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"


def setup_state(state):
    if "messages" not in state:
        state["messages"] = []
    if "model" not in state:
        state["model"] = "omniparser + gpt-4o"
    if "provider" not in state:
        state["provider"] = "openai"
    if "openai_api_key" not in state:  # Fetch API keys from environment variables
        state["openai_api_key"] = os.getenv("OPENAI_API_KEY", "")
    if "anthropic_api_key" not in state:
        state["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY", "")
    if "api_key" not in state:
        state["api_key"] = ""
    if "auth_validated" not in state:
        state["auth_validated"] = False
    if "responses" not in state:
        state["responses"] = {}
    if "tools" not in state:
        state["tools"] = {}
    if "only_n_most_recent_images" not in state:
        state["only_n_most_recent_images"] = 2
    if 'chatbot_messages' not in state:
        state['chatbot_messages'] = []
    if 'stop' not in state:
        state['stop'] = False
    if 'ollama_model_name' not in state:
        state['ollama_model_name'] = ''
    if 'ollama_json_model_name' not in state:
        state['ollama_json_model_name'] = ''
    if 'ollama_base_url' not in state:
        state['ollama_base_url'] = 'http://localhost:11434'
    if 'ollama_supports_vision' not in state:
        state['ollama_supports_vision'] = True

async def main(state):
    """Render loop for Gradio"""
    setup_state(state)
    return "Setup completed"

def validate_auth(provider: APIProvider, api_key: str | None):
    if provider == APIProvider.ANTHROPIC:
        if not api_key:
            return "Enter your Anthropic API key to continue."
    if provider == APIProvider.BEDROCK:
        import boto3

        if not boto3.Session().get_credentials():
            return "You must have AWS credentials set up to use the Bedrock API."
    if provider == APIProvider.VERTEX:
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError

        if not os.environ.get("CLOUD_ML_REGION"):
            return "Set the CLOUD_ML_REGION environment variable to use the Vertex API."
        try:
            google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        except DefaultCredentialsError:
            return "Your google cloud credentials are not set up correctly."

def load_from_storage(filename: str) -> str | None:
    """Load data from a file in the storage directory."""
    try:
        file_path = CONFIG_DIR / filename
        if file_path.exists():
            data = file_path.read_text().strip()
            if data:
                return data
    except Exception as e:
        print(f"Debug: Error loading {filename}: {e}")
    return None

def save_to_storage(filename: str, data: str) -> None:
    """Save data to a file in the storage directory."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        file_path = CONFIG_DIR / filename
        file_path.write_text(data)
        # Ensure only user can read/write the file
        file_path.chmod(0o600)
    except Exception as e:
        print(f"Debug: Error saving {filename}: {e}")

def _api_response_callback(response: APIResponse[BetaMessage], response_state: dict):
    response_id = datetime.now().isoformat()
    response_state[response_id] = response

def _tool_output_callback(tool_output: ToolResult, tool_id: str, tool_state: dict):
    tool_state[tool_id] = tool_output

def chatbot_output_callback(message, chatbot_state, hide_images=False, sender="bot"):
    def _render_message(message: str | BetaTextBlock | BetaToolUseBlock | ToolResult, hide_images=False):
    
        print(f"_render_message: {str(message)[:100]}")
        
        if isinstance(message, str):
            return message
        
        is_tool_result = not isinstance(message, str) and (
            isinstance(message, ToolResult)
            or message.__class__.__name__ == "ToolResult"
        )
        if not message or (
            is_tool_result
            and hide_images
            and not hasattr(message, "error")
            and not hasattr(message, "output")
        ):  # return None if hide_images is True
            return
        # render tool result
        if is_tool_result:
            message = cast(ToolResult, message)
            if message.output:
                return message.output
            if message.error:
                return f"Error: {message.error}"
            if message.base64_image and not hide_images:
                # somehow can't display via gr.Image
                # image_data = base64.b64decode(message.base64_image)
                # return gr.Image(value=Image.open(io.BytesIO(image_data)))
                return f'<img src="data:image/png;base64,{message.base64_image}">'

        elif isinstance(message, BetaTextBlock) or isinstance(message, TextBlock):
            return f"Analysis: {message.text}"
        elif isinstance(message, BetaToolUseBlock) or isinstance(message, ToolUseBlock):
            # return f"Tool Use: {message.name}\nInput: {message.input}"
            return f"Next I will perform the following action: {message.input}"
        else:  
            return message

    def _truncate_string(s, max_length=500):
        """Truncate long strings for concise printing."""
        if isinstance(s, str) and len(s) > max_length:
            return s[:max_length] + "..."
        return s
    # processing Anthropic messages
    message = _render_message(message, hide_images)
    
    role = "assistant" if sender == "bot" else "user"
    chatbot_state.append({"role": role, "content": message})
    
    # Create a concise version of the chatbot state for printing
    concise_state = [(_truncate_string(str(m.get("content", "")))) for m in chatbot_state]
    # print(f"chatbot_output_callback chatbot_state: {concise_state} (truncated)")

def valid_params(user_input, state):
    """Validate all requirements and return a list of error messages."""
    errors = []
    
    for server_name, url in [
        ('Windows Host', _windows_control_probe_url()),
        ('OmniParser Server', f"http://{args.omniparser_server_url}/probe"),
    ]:
        try:
            response = requests.get(url, timeout=3)
            if response.status_code != 200:
                errors.append(f"{server_name} is not responding ({url})")
        except RequestException as e:
            errors.append(f"{server_name} is not responding ({url})")
    
    # Be tolerant to UI/state sync issues: if the model selection contains "ollama",
    # treat it as Ollama even if provider hasn't updated.
    is_ollama = (state.get("provider") == "ollama") or ("ollama" in str(state.get("model", "")).lower())
    
    if is_ollama:
        # Check Ollama connectivity instead of API key
        from agent.llm_utils.ollamaclient import check_ollama_connection
        ollama_url = state.get('ollama_base_url', 'http://localhost:11434')
        if not check_ollama_connection(ollama_url):
            errors.append(f"Cannot connect to Ollama at {ollama_url}. Is Ollama running?")
        if not state.get('ollama_model_name', '').strip():
            errors.append("Ollama model name is not set. Select or enter a model name.")
    else:
        if not state["api_key"].strip():
            errors.append("LLM API Key is not set")

    if not user_input:
        errors.append("no computer use request provided")
    
    return errors

def process_input(
    user_input,
    state,
    ollama_model_value=None,
    ollama_json_model_value=None,
    ollama_url_value=None,
    ollama_vision_value=None,
):
    # Reset the stop flag
    if state["stop"]:
        state["stop"] = False

    # Always sync current UI values into state before validation/execution.
    if ollama_model_value is not None:
        state["ollama_model_name"] = ollama_model_value
    if ollama_json_model_value is not None:
        state["ollama_json_model_name"] = ollama_json_model_value
    if ollama_url_value is not None:
        state["ollama_base_url"] = normalize_ollama_ui_url(ollama_url_value)
    if ollama_vision_value is not None:
        state["ollama_supports_vision"] = ollama_vision_value

    errors = valid_params(user_input, state)
    if errors:
        raise gr.Error("Validation errors: " + ", ".join(errors))
    
    # Append the user message to state["messages"]
    state["messages"].append(
        {
            "role": Sender.USER,
            "content": [TextBlock(type="text", text=user_input)],
        }
    )

    # Append the user's message to chatbot_messages
    state['chatbot_messages'].append({"role": "user", "content": user_input})
    yield state['chatbot_messages']  # Yield to update the chatbot UI with the user's message

    print("state")
    print(state)

    # Run sampling_loop_sync with the chatbot_output_callback
    for loop_msg in sampling_loop_sync(
        model=state["model"],
        provider=state["provider"],
        messages=state["messages"],
        output_callback=partial(chatbot_output_callback, chatbot_state=state['chatbot_messages'], hide_images=False),
        tool_output_callback=partial(_tool_output_callback, tool_state=state["tools"]),
        api_response_callback=partial(_api_response_callback, response_state=state["responses"]),
        api_key=state["api_key"],
        only_n_most_recent_images=state["only_n_most_recent_images"],
        max_tokens=16384,
        omniparser_url=args.omniparser_server_url,
        ollama_model_name=state.get("ollama_model_name", ""),
        ollama_json_model_name=state.get("ollama_json_model_name", ""),
        ollama_base_url=state.get("ollama_base_url", "http://localhost:11434"),
        ollama_supports_vision=state.get("ollama_supports_vision", True)
    ):  
        if loop_msg is None or state.get("stop"):
            yield state['chatbot_messages']
            print("End of task. Close the loop.")
            break
            
        yield state['chatbot_messages']  # Yield the updated chatbot_messages to update the chatbot UI

def stop_app(state):
    state["stop"] = True
    return "App stopped"

def get_header_image_base64():
    try:
        # Get the absolute path to the image relative to this script
        script_dir = Path(__file__).parent
        image_path = script_dir.parent.parent / "imgs" / "header_bar_thin.png"
        
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            return f'data:image/png;base64,{encoded_string}'
    except Exception as e:
        print(f"Failed to load header image: {e}")
        return None

with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.HTML("""
        <style>
        .no-padding {
            padding: 0 !important;
        }
        .no-padding > div {
            padding: 0 !important;
        }
        .markdown-text p {
            font-size: 18px;  /* Adjust the font size as needed */
        }
        </style>
    """)
    state = gr.State({})
    
    setup_state(state.value)
    
    header_image = get_header_image_base64()
    if header_image:
        gr.HTML(f'<img src="{header_image}" alt="OmniTool Header" width="100%">', elem_classes="no-padding")
        gr.HTML('<h1 style="text-align: center; font-weight: normal;">Omni<span style="font-weight: bold;">Tool</span></h1>')
    else:
        gr.Markdown("# OmniTool")

    if not os.getenv("HIDE_WARNING", False):
        gr.Markdown(INTRO_TEXT, elem_classes="markdown-text")


    with gr.Accordion("Settings", open=True): 
        with gr.Row():
            with gr.Column():
                model = gr.Dropdown(
                    label="Model",
                    choices=["omniparser + gpt-4o", "omniparser + o1", "omniparser + o3-mini", "omniparser + R1", "omniparser + qwen2.5vl", "omniparser + ollama", "claude-3-5-sonnet-20241022", "omniparser + gpt-4o-orchestrated", "omniparser + o1-orchestrated", "omniparser + o3-mini-orchestrated", "omniparser + R1-orchestrated", "omniparser + qwen2.5vl-orchestrated", "omniparser + ollama-orchestrated"],
                    value="omniparser + gpt-4o",
                    interactive=True,
                )
            with gr.Column():
                only_n_images = gr.Slider(
                    label="N most recent screenshots",
                    minimum=0,
                    maximum=10,
                    step=1,
                    value=2,
                    interactive=True
                )
        with gr.Row():
            with gr.Column(1):
                provider = gr.Dropdown(
                    label="API Provider",
                    choices=[option.value for option in APIProvider],
                    value="openai",
                    interactive=False,
                )
            with gr.Column(2):
                api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    value=state.value.get("api_key", ""),
                    placeholder="Paste your API key here",
                    interactive=True,
                )
        with gr.Row(visible=False) as ollama_settings_row:
            with gr.Column(scale=2):
                ollama_model_name = gr.Dropdown(
                    label="Ollama Model",
                    choices=[],
                    value="",
                    allow_custom_value=True,
                    interactive=True,
                    info="Select from running Ollama models or type a custom model name (e.g. llama3.2-vision:11b)"
                )
            with gr.Column(scale=2):
                ollama_json_model_name = gr.Dropdown(
                    label="Ollama JSON Model (Optional)",
                    choices=[],
                    value="",
                    allow_custom_value=True,
                    interactive=True,
                    info="Optional second model to convert vision output into strict action JSON (e.g. qwen3-coder:30b)"
                )
            with gr.Column(scale=1):
                ollama_base_url = gr.Textbox(
                    label="Ollama Base URL",
                    value="http://localhost:11434",
                    interactive=True,
                    info="URL where Ollama is running"
                )
            with gr.Column(scale=1):
                ollama_supports_vision = gr.Checkbox(
                    label="Vision Model",
                    value=True,
                    interactive=True,
                    info="Enable if model supports image input (e.g. llama3.2-vision, llava)"
                )
            with gr.Column(scale=1, min_width=120):
                ollama_refresh_btn = gr.Button(value="🔄 Fetch Models", variant="secondary")

    with gr.Row():
        with gr.Column(scale=8):
            chat_input = gr.Textbox(show_label=False, placeholder="Type a message to send to Omniparser + X ...", container=False)
        with gr.Column(scale=1, min_width=50):
            submit_button = gr.Button(value="Send", variant="primary")
        with gr.Column(scale=1, min_width=50):
            stop_button = gr.Button(value="Stop", variant="secondary")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chatbot History", autoscroll=True, height=580)
        with gr.Column(scale=3):
            iframe = gr.HTML(
                f'<iframe src="http://{args.windows_host_url}/vnc.html?view_only=1&autoconnect=1&resize=scale" width="100%" height="580" allow="fullscreen"></iframe>',
                container=False,
                elem_classes="no-padding"
            )

    def update_model(model_selection, state):
        state["model"] = model_selection
        print(f"Model updated to: {state['model']}")
        
        is_ollama = "ollama" in model_selection
        
        if model_selection == "claude-3-5-sonnet-20241022":
            provider_choices = [option.value for option in APIProvider if option.value not in ("openai", "ollama")]
        elif model_selection in set(["omniparser + gpt-4o", "omniparser + o1", "omniparser + o3-mini", "omniparser + gpt-4o-orchestrated", "omniparser + o1-orchestrated", "omniparser + o3-mini-orchestrated"]):
            provider_choices = ["openai"]
        elif model_selection == "omniparser + R1":
            provider_choices = ["groq"]
        elif model_selection == "omniparser + qwen2.5vl":
            provider_choices = ["dashscope"]
        elif is_ollama:
            provider_choices = ["ollama"]
        else:
            provider_choices = [option.value for option in APIProvider]
        default_provider_value = provider_choices[0]

        provider_interactive = len(provider_choices) > 1
        api_key_placeholder = f"{default_provider_value.title()} API Key"

        # Update state
        state["provider"] = default_provider_value
        state["api_key"] = state.get(f"{default_provider_value}_api_key", "")

        # Calls to update other components UI
        provider_update = gr.update(
            choices=provider_choices,
            value=default_provider_value,
            interactive=provider_interactive
        )
        
        if is_ollama:
            api_key_update = gr.update(
                placeholder="Not required for Ollama",
                value="",
                interactive=False
            )
        else:
            api_key_update = gr.update(
                placeholder=api_key_placeholder,
                value=state["api_key"],
                interactive=True
            )
        
        ollama_row_update = gr.update(visible=is_ollama)

        return provider_update, api_key_update, ollama_row_update

    def update_only_n_images(only_n_images_value, state):
        state["only_n_most_recent_images"] = only_n_images_value
   
    def update_provider(provider_value, state):
        # Update state
        state["provider"] = provider_value
        state["api_key"] = state.get(f"{provider_value}_api_key", "")
        
        # Calls to update other components UI
        api_key_update = gr.update(
            placeholder=f"{provider_value.title()} API Key",
            value=state["api_key"]
        )
        return api_key_update
                
    def update_api_key(api_key_value, state):
        state["api_key"] = api_key_value
        state[f'{state["provider"]}_api_key'] = api_key_value

    def clear_chat(state):
        # Reset message-related state
        state["messages"] = []
        state["responses"] = {}
        state["tools"] = {}
        state['chatbot_messages'] = []
        return state['chatbot_messages']

    model.change(fn=update_model, inputs=[model, state], outputs=[provider, api_key, ollama_settings_row])
    only_n_images.change(fn=update_only_n_images, inputs=[only_n_images, state], outputs=None)
    provider.change(fn=update_provider, inputs=[provider, state], outputs=api_key)
    api_key.change(fn=update_api_key, inputs=[api_key, state], outputs=None)
    chatbot.clear(fn=clear_chat, inputs=[state], outputs=[chatbot])

    def update_ollama_model_name(ollama_model_value, state):
        state["ollama_model_name"] = ollama_model_value

    def update_ollama_json_model_name(ollama_json_model_value, state):
        state["ollama_json_model_name"] = ollama_json_model_value
    
    def update_ollama_base_url(ollama_url_value, state):
        state["ollama_base_url"] = normalize_ollama_ui_url(ollama_url_value)
    
    def update_ollama_supports_vision(vision_value, state):
        state["ollama_supports_vision"] = vision_value
    
    def fetch_ollama_models(state, ollama_url_value):
        from agent.llm_utils.ollamaclient import get_ollama_models
        # Read latest textbox value directly so we don't depend on change-event timing.
        base_url = normalize_ollama_ui_url(ollama_url_value or state.get("ollama_base_url", "http://localhost:11434"))
        state["ollama_base_url"] = base_url
        models = get_ollama_models(base_url)
        if models:
            # Auto-detect vision models
            current = state.get("ollama_model_name", "")
            if not current:
                # Default to first vision model if available, else first model
                vision_keywords = ["vision", "llava", "bakllava", "moondream"]
                vision_models = [m for m in models if any(kw in m.lower() for kw in vision_keywords)]
                default_model = vision_models[0] if vision_models else models[0]
                state["ollama_model_name"] = default_model
                is_vision = any(kw in default_model.lower() for kw in vision_keywords)
                state["ollama_supports_vision"] = is_vision
                return (
                    gr.update(choices=models, value=default_model),
                    gr.update(choices=models, value=state.get("ollama_json_model_name", "")),
                    gr.update(value=is_vision),
                )
            return (
                gr.update(choices=models, value=current),
                gr.update(choices=models, value=state.get("ollama_json_model_name", "")),
                gr.update(),
            )
        else:
            gr.Warning(f"Could not fetch models from Ollama at {base_url}. Is Ollama running?")
            return gr.update(choices=[], value=""), gr.update(choices=[], value=""), gr.update()

    ollama_model_name.change(fn=update_ollama_model_name, inputs=[ollama_model_name, state], outputs=None)
    ollama_json_model_name.change(fn=update_ollama_json_model_name, inputs=[ollama_json_model_name, state], outputs=None)
    ollama_base_url.change(fn=update_ollama_base_url, inputs=[ollama_base_url, state], outputs=None)
    ollama_supports_vision.change(fn=update_ollama_supports_vision, inputs=[ollama_supports_vision, state], outputs=None)
    ollama_refresh_btn.click(
        fn=fetch_ollama_models,
        inputs=[state, ollama_base_url],
        outputs=[ollama_model_name, ollama_json_model_name, ollama_supports_vision],
    )

    submit_button.click(
        process_input,
        [chat_input, state, ollama_model_name, ollama_json_model_name, ollama_base_url, ollama_supports_vision],
        chatbot,
    )
    stop_button.click(stop_app, [state], None)
    
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7888)
