import httpx
import json
import sys
import yaml
import streamlit as st
from markdown2 import markdown
from pathlib import Path
from pydantic import BaseModel, HttpUrl, Field, validator
from typing import Optional, Dict, List, Union
from streamlit_float import *


class ChatResponse(BaseModel):
    """Response from chatbot including the answer and placeholder"""

    answer: str
    placeholder: object
    context: Optional[List[str]] = []
    sources: Optional[List[str]] = []

    class Config:
        arbitrary_types_allowed = True  # Required to allow Streamlit placeholder...


class ChatPair(BaseModel):
    """Represents a QA pair in the chat history"""

    question: str
    rag_response: str
    llm_response: str
    rag_context: Optional[List[str]] = []
    rag_sources: Optional[List[str]] = []


class AppConfig(BaseModel):
    """Application configuration model"""

    rag_url: HttpUrl
    llm_url: HttpUrl
    model: str

    @validator("rag_url", "llm_url", allow_reuse=True)
    def validate_urls(cls, v):
        return str(v)


class HTTPConfig(BaseModel):
    """HTTP configuration model"""

    host: str = Field(default="127.0.0.1")
    port: int

    @validator("host", allow_reuse=True)
    def validate_host(cls, v):
        return "127.0.0.1" if v == "0.0.0.0" else v


class LLMPluginConfig(BaseModel):
    """LLM plugin configuration model"""

    host: HttpUrl
    model: Optional[str] = None


class PluginsConfig(BaseModel):
    """Plugins configuration model"""

    llm: Dict[str, LLMPluginConfig]


class ConfigFile(BaseModel):
    """Complete configuration file model"""

    http: HTTPConfig
    plugins: PluginsConfig


class Config:
    """Configuration management class"""

    DEFAULT_CONFIG_PATH = Path("./config/configuration.yml")

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config = self._load_config()

    def _load_config(self) -> AppConfig:
        """Load and parse configuration file"""
        try:
            with open(self.config_path, "r") as file:
                config_data = yaml.safe_load(file)
                config_file = ConfigFile(**config_data)

            # HTTP URL configuration
            http_url = f"http://{config_file.http.host}:{config_file.http.port}"

            # Find the first valid setting in LLM settings
            llm_config = next(iter(config_file.plugins.llm.values()))

            return AppConfig(
                rag_url=http_url,
                llm_url=str(llm_config.host),
                model=llm_config.model or "default-model",
            )
        except (yaml.YAMLError, IOError) as e:
            st.error(f"Configuration error: {str(e)}")
            raise


class ChatMessage(BaseModel):
    """Chat message model for API requests"""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Chat API request model"""

    model: str
    messages: List[ChatMessage]
    stream: bool = True
    temperature: float = 0.7
    instruction: Optional[str] = None


class ChatBot:
    """Handles chat interactions with RAG and LLM models"""

    def __init__(self, baseurl: str, model: str):
        self.baseurl = baseurl
        self.model = model

    def _process_stream(self, response, bot_name: str) -> ChatResponse:
        """Process streaming response and update UI"""
        bot_label = f"**{bot_name}**\n\n"
        full_response = ""
        placeholder = st.empty()
        context_list = []
        sources_list = []

        for line in response.iter_lines():
            if line.startswith("data: "):
                chunk = self._process_line(line, bot_name)
                if chunk is None:
                    break
                if chunk:
                    if isinstance(chunk, dict):
                        if "context" in chunk:
                            context_list.append(chunk["context"])
                        elif "source" in chunk:
                            sources_list.append(chunk["source"])
                    else:
                        full_response += chunk
                        placeholder.markdown(bot_label + full_response)

        return ChatResponse(
            answer=full_response,
            placeholder=placeholder,
            context=context_list,
            sources=sources_list,
        )

    def _process_line(self, line: str, bot_name: str) -> Optional[Union[str, Dict]]:
        """Process a single line from the stream"""
        try:
            json_str = line[6:]
            if json_str.strip() == "[DONE]":
                return None

            json_obj = json.loads(json_str)

            # Handle RAG-specific information
            if json_obj.get("object") in ["rag.context", "rag.source"]:
                if "choices" in json_obj and json_obj["choices"]:
                    delta = json_obj["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        # Return different dict structure based on the object type
                        if json_obj["object"] == "rag.context":
                            return {"context": content}
                        else:  # rag.source
                            return {"source": content}
                return {}

            # Handle regular chat completion
            return json_obj.get("choices", [{}])[0].get("delta", {}).get("content", "")

        except json.JSONDecodeError:
            return ""

    def get_streaming_response(
        self,
        bot_name: str,
        question: str,
        temperature: float = 0.7,
        instruction: Optional[str] = None,
        timeout: float = 60.0,
    ) -> ChatResponse:
        """Get streaming response from the model with temperature and instruction"""
        messages = []
        if instruction:
            messages.append(ChatMessage(role="system", content=instruction))
        messages.append(ChatMessage(role="user", content=question))

        request = ChatRequest(
            model=self.model, messages=messages, temperature=temperature
        )

        with httpx.Client(timeout=timeout) as client:
            with client.stream(
                "POST",
                f"{self.baseurl}/v1/chat/completions".replace("//v1", "/v1"),
                json=request.dict(),
            ) as response:
                return self._process_stream(response, bot_name)


class ChatUI:
    """Handles UI components and styling"""

    def __init__(self):
        st.set_page_config(layout="wide")
        self._setup_styles()
        self._initialize_session_state()

        float_init()

        # Create two columns for main layout
        self.cols = st.columns([7, 3])  # 7:3 ratio for chat:settings

        # Keep original chat container in the main area
        self.chat_container = self.cols[0].container()
        # Settings will go in the right column
        self.settings_container = self.cols[1].container()
        self.settings_container.float("right: 2rem;")

    @staticmethod
    def _initialize_session_state():
        """Initialize session state for chat history and settings"""
        if "chat_pairs" not in st.session_state:
            st.session_state["chat_pairs"] = []
        if "instruction" not in st.session_state:
            st.session_state["instruction"] = (
                """You are a helpful AI assistant. Provide clear, accurate, and relevant responses."""
            )
        if "temperature" not in st.session_state:
            st.session_state["temperature"] = 0.7

    def render_chat_history(self):
        """Render chat history"""
        with self.chat_container:
            for pair in st.session_state["chat_pairs"]:
                self._render_chat_pair(pair)

    @staticmethod
    def _render_chat_pair(pair: ChatPair):
        """Render a single chat pair with button-style collapsible sections"""
        # Create context HTML if context exists
        context_html = ""
        if pair.rag_context:
            context_items = "".join([f"<li>{ctx}</li>" for ctx in pair.rag_context])
            context_html = f"""
                <details class="details-wrapper">
                    <summary class="details-summary">
                        <span class="summary-icon">▶</span>
                        <span class="summary-text">Related Contexts</span>
                    </summary>
                    <div class="context-section">
                        <ul class="context-list">
                            {context_items}
                        </ul>
                    </div>
                </details>
            """

        # Create sources HTML if sources exist
        sources_html = ""
        if pair.rag_sources:
            source_items = "".join([f"<li>{src}</li>" for src in pair.rag_sources])
            sources_html = f"""
                <details class="details-wrapper">
                    <summary class="details-summary">
                        <span class="summary-icon">▶</span>
                        <span class="summary-text">Sources</span>
                    </summary>
                    <div class="source-section">
                        <ul class="source-list">
                            {source_items}
                        </ul>
                    </div>
                </details>
            """

        st.html(
            f"""
            <div class="chat-group">
                <div class="chat-message user-message">
                    <b>You:</b> <div>{markdown(pair.question)}</div>
                </div>
                <div class="bot-responses">
                    <div class="chat-message rag-message">
                        <b>RAG:</b> 
                        <div>{markdown(pair.rag_response)}</div>
                        {sources_html}
                        {context_html}
                    </div>
                    <div class="chat-message llm-message">
                        <b>LLM:</b> <div>{markdown(pair.llm_response)}</div>
                    </div>
                </div>
            </div>
            """
        )

    @staticmethod
    def _setup_styles():
        """Setup CSS styles for the chat interface"""
        st.markdown(
            """
            <style>
            /* Main layout adjustments */
            .main > div:first-child {
                padding-right: 1rem;
            }

            /* Chat styles */
            .chat-group {
                margin-bottom: 2rem;
                color: #333;
            }
            .chat-message {
                padding: 1rem;
                margin: 0.5rem 0;
                border-radius: 0.5rem;
            }
            .chat-group ol, ul, dl { padding-left: 2rem; }
            .user-message { background-color: #e6f3ff; }
            .bot-responses {
                display: flex;
                gap: 1rem;
            }
            .rag-message {
                background-color: #e7f7e7;
                flex: 1;
            }
            .llm-message {
                background-color: #f7e7e7;
                flex: 1;
            }

            /* Details and Summary styling */
            .details-wrapper {
                margin-top: 0.8rem;
                border-radius: 0.3rem;
            }
            
            .details-summary {
                display: inline-flex;
                align-items: center;
                padding: 0.4rem 0.8rem;
                border-radius: 0.3rem;
                cursor: pointer;
                user-select: none;
                background-color: #ffffff;
                border: 1px solid #38a169;
                color: #38a169;
                font-size: 0.875rem;
                font-weight: 500;
                transition: all 0.2s ease;
            }
            
            .details-summary:hover {
                background-color: #38a169;
                color: white;
            }

            .summary-icon {
                font-size: 0.75rem;
                margin-right: 0.5rem;
                transition: transform 0.2s ease;
            }

            details[open] .summary-icon {
                transform: rotate(90deg);
            }

            .summary-text {
                position: relative;
                top: 0.5px;
            }

            /* Context styling */
            .context-section {
                margin-top: 0.5rem;
                padding: 0.75rem;
                background-color: rgba(255, 255, 255, 0.7);
                border-radius: 0.3rem;
                border: 1px solid #e2e8f0;
            }
            .context-list {
                margin: 0;
                padding-left: 1.5rem;
                font-size: 0.9em;
                color: #4a5568;
            }
            .context-list li {
                margin-bottom: 0.3rem;
            }

            /* Source styling */
            .source-section {
                margin-top: 0.5rem;
                padding: 0.75rem;
                background-color: rgba(255, 255, 255, 0.7);
                border-radius: 0.3rem;
                border: 1px solid #e2e8f0;
            }
            .source-list {
                margin: 0;
                padding-left: 1.5rem;
                font-size: 0.9em;
                color: #4a5568;
            }
            .source-list li {
                margin-bottom: 0.3rem;
            }

            /* Settings column styling */
            [data-testid="column"]:nth-child(2) {
                background-color: #f8f9fa;
                padding: 1rem;
                border-left: 1px solid #dee2e6;
            }

            .stChatInput [data-baseweb="textarea"] {
                border-color: #777;
                background-color: #f8f9fa;
            }

            .stChatInput textarea {
                color: #434343;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    def render_settings(self):
        """Render settings in the right column"""
        with self.settings_container:
            # LLM Settings section
            st.header("LLM Settings")

            # Temperature slider
            st.session_state["temperature"] = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state["temperature"],
                step=0.1,
                help="Higher values make output more random, lower values more deterministic",
            )

            # Instruction text area
            st.session_state["instruction"] = st.text_area(
                "System Instruction",
                value=st.session_state["instruction"],
                height=200,
                help="Set the behavior and context for the LLM",
            )


class ChatApp:
    """Main application class"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.ui = ChatUI()
        self.rag_bot = ChatBot(str(config.rag_url), config.model)
        self.llm_bot = ChatBot(str(config.llm_url), config.model)

    def run(self):
        """Run the chat application"""
        # Render settings in right column
        self.ui.render_settings()

        # Chat interface in main area (keeping original layout)
        question = st.chat_input("Your question:")
        if question:
            self._handle_question(question)

        # Render chat history in main container
        self.ui.render_chat_history()

    def _handle_question(self, question: str):
        """Handle user question and get responses"""
        try:
            col1, col2, col3 = st.columns([3.5, 3.5, 3], gap="medium")

            with col1:
                rag_response = self.rag_bot.get_streaming_response("RAG", question)

            with col2:
                llm_response = self.llm_bot.get_streaming_response(
                    "LLM",
                    question,
                    temperature=st.session_state["temperature"],
                    instruction=st.session_state["instruction"],
                )

            with col3:
                pass

            # Add to chat history with context and sources
            chat_pair = ChatPair(
                question=question,
                rag_response=rag_response.answer,
                llm_response=llm_response.answer,
                rag_context=(
                    rag_response.context if hasattr(rag_response, "context") else []
                ),
                rag_sources=(
                    rag_response.sources if hasattr(rag_response, "sources") else []
                ),
            )
            st.session_state["chat_pairs"].append(chat_pair)

            # Clear placeholders
            rag_response.placeholder.empty()
            llm_response.placeholder.empty()

        except Exception as e:
            st.error(f"Error processing question: {str(e)}")


def main():
    """Main entry point"""
    try:
        if "--app-config" in sys.argv:
            config_path = Path(sys.argv[sys.argv.index("--app-config") + 1])
        else:
            config_path = None

        config = Config(config_path).config
        app = ChatApp(config)
        app.run()

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
