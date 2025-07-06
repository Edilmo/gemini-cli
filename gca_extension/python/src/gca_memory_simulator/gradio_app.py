"""
This module contains a gradio App that allows you to test the GeminiMemory.
"""

import logging
import os
from datetime import datetime
from typing import Optional, List, Union, Dict, Any
import sys

import chromadb
from chromadb import EmbeddingFunction, Embeddings
from chromadb.api.types import Embeddable
from fastapi import FastAPI
from google import genai
from google.genai.types import Content, Part, File
import gradio as gr
import numpy as np
# from PIL import Image
from pydantic import BaseModel, Field, ConfigDict
# from pydantic_core import core_schema


# ------------------------------------------------------------------
# Configure logging to go to both stderr and stdout
# ------------------------------------------------------------------
root = logging.getLogger()  # root logger
root.setLevel(logging.INFO)

# Remove handlers that basicConfig() may have added
while root.handlers:
    root.removeHandler(root.handlers[0])

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)

stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.INFO)

# Optional: format messages
fmt = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
stdout_handler.setFormatter(fmt)
stderr_handler.setFormatter(fmt)

root.addHandler(stdout_handler)
root.addHandler(stderr_handler)
# ------------------------------------------------------------------


logger = logging.getLogger(__name__)


PartUnion = Union[File, Part, str]
ContentUnion = Union[Content, list[PartUnion], PartUnion]
ContentListUnion = Union[list[ContentUnion], ContentUnion]


class ContextWindow(BaseModel):
    """
    A class that represents a context window for the Gemini model call.
    """

    # class Config:
    #     arbitrary_types_allowed = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    system_instruction: Optional[ContentUnion] = Field(
        default=None, description="The system instruction for the context window."
    )
    contents: ContentListUnion = Field(default_factory=list, description="The contents of the context window.")


class ContextWindowInput(ContextWindow):
    """
    A class that represents a context window input for the Gemini memory.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_id: str = Field(description="The ID of the user that is interacting with the Gemini model.")


class MemoryEntry(BaseModel):
    """
    A class that represents a memory entry for the Gemini memory.
    """

    content: str = Field(description="The content of the memory.")
    user_id: str = Field(description="The ID of the user that is interacting with the Gemini model.")
    timestamp: str = Field(description="The creation timestamp of the memory entry.")
    section_id: str = Field(description="The ID of the memory section where the memory entry is stored.")

    @property
    def id(self) -> str:
        """
        Returns the ID of the memory entry.
        """
        return f"{self.user_id}_{self.timestamp}_{self.section_id}"


class GeminiMemory:
    """
    A class that represents a Gemini memory that works as context window manager for the Gemini model call.
    """

    def __init__(self, gemini_client: genai.Client, session_log_enabled: bool = False):
        """
        Initialize the GeminiMemory.

        Args:
            gemini_client: The Google GenAI client to use for any Gemini API calls.
        """
        self.gemini_client = gemini_client
        self.session_log_enabled = session_log_enabled
        self.session_log: List[ContextWindowInput] = []

    def put(self, input: ContextWindowInput) -> ContextWindow:
        """
        Put a new context window into the Gemini memory.

        Args:
            input: The input context window to add to the Gemini memory.

        Returns:
            The new system instruction and contents of the context window enhanced with the relevant memories.
        """
        if self.session_log_enabled:
            self.session_log.append(input)
        return self._put(input)

    def _put(self, input: ContextWindowInput) -> ContextWindow:
        """
        Put a new context window into the Gemini memory.

        Args:
            input: The input context window to add to the Gemini memory.

        Returns:
            The new system instruction and contents of the context window enhanced with the relevant memories.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def list(self) -> List[str]:
        """
        Returns all memories from the Gemini memory.

        Returns:
            A list of all memories from the Gemini memory.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def post(self, memory: MemoryEntry) -> None:
        """
        Add a new memory to the Gemini memory.

        Args:
            memory: The memory entry to add to the Gemini memory.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _get_last_user_message(self, input: ContextWindowInput) -> str:
        """
        Return the text of the most recent *user* message from the context window.
        Traverses the contents in reverse temporal order (last element first)
        and returns the first message whose role is "user".

        Args:
            input: The input context window to get the last user message from.

        Returns:
            The text of the most recent *user* message from the context window.
        """

        def transverse_contents(contents: List[ContentUnion]) -> Optional[ContentUnion]:
            first_non_content_item: Optional[Union[list[PartUnion], PartUnion]] = None
            for content in reversed(contents):
                if isinstance(content, Content):
                    if content.role == "user" and content.parts and any(part.text for part in content.parts):
                        return content
                elif first_non_content_item is None:
                    first_non_content_item = content
            return first_non_content_item

        last_user_message = ""
        parts: List[PartUnion] = []
        last_content: Optional[ContentUnion] = (
            transverse_contents(input.contents)
            if isinstance(input.contents, list)
            and all(isinstance(item, list) or isinstance(item, Content) for item in input.contents)
            else input.contents
        )
        if isinstance(last_content, Content):
            if last_content.parts and last_content.role == "user":
                parts = list(last_content.parts)
            else:
                parts = []
        elif isinstance(last_content, list):
            parts = last_content
        elif last_content is not None:
            parts = [last_content]

        if parts:
            parts_str = [
                part if isinstance(part, str) else part.text
                for part in parts
                if isinstance(part, Part) or isinstance(part, str)
            ]
            last_user_message = "\n".join([part for part in parts_str if isinstance(part, str)])
            logger.info("Last user message detected")

        if self.session_log_enabled:
            self.session_log.append(
                ContextWindowInput(
                    user_id=input.user_id,
                    system_instruction="",
                    contents=[last_user_message],
                )
            )
        return last_user_message

    def _get_new_system_instruction(
        self, user_id: str, system_instruction: Optional[ContentUnion], retrieved_memories: str
    ) -> ContentUnion:
        """
        Get a new system instruction from the retrieved memories.

        Args:
            user_id: The user ID of the user that is interacting with the Gemini model.
            system_instruction: The system instruction to enhance.
            retrieved_memories: The retrieved memories to enhance the system instruction with.

        Returns:
            The new system instruction.
        """
        if self.session_log_enabled:
            self.session_log.append(
                ContextWindowInput(
                    user_id=user_id,
                    system_instruction=f"Retrieved Memories\n{retrieved_memories}",
                    contents=[],
                )
            )
        if not system_instruction:
            system_instruction = retrieved_memories
        else:
            if isinstance(system_instruction, Content):
                if system_instruction.parts:
                    system_instruction.parts.append(Part(text=retrieved_memories))
                else:
                    system_instruction.parts = [Part(text=retrieved_memories)]
            elif isinstance(system_instruction, list):
                system_instruction.append(Part(text=retrieved_memories))
            else:
                system_instruction = f"{system_instruction}\n{retrieved_memories}"
        return system_instruction

    def set_session_log_enabled(self, value: bool) -> None:
        """
        Set the session log enabled flag.

        Args:
            value: The value to set the session log enabled flag to.
        """
        self.session_log_enabled = value


class GeminiEmbeddingFunction(EmbeddingFunction[Embeddable]):
    """
    A class that generates embeddings for text documents using the Gemini embedding model.
    """

    def __init__(self, client: genai.Client):
        """
        Initialize the GeminiEmbeddingFunction.

        Args:
            client: The Google GenAI client to use for embedding.
        """
        self.model = "gemini-embedding-exp-03-07"
        self.client = client

    def __call__(self, input: Embeddable) -> Embeddings:
        """Generate embeddings for the given documents using Gemini embedding model.

        Args:
            input: List of text documents to generate embeddings for.

        Returns:
            List of embeddings as numpy arrays.
        """
        # Gemini only works with text documents
        if not all(isinstance(item, str) for item in input):  # type: ignore
            raise ValueError("Gemini only supports text documents")

        # Generate embeddings using Gemini model
        contents: ContentListUnion = [text for text in input if isinstance(text, str)]
        result = self.client.models.embed_content(
            model=self.model,
            contents=contents,
        )
        assert result.embeddings is not None
        embeddings: Embeddings = [np.array(embedding.values, dtype=np.float32) for embedding in result.embeddings]
        return embeddings


class ChromaMemory(GeminiMemory):
    """
    A class that represents a Chroma memory that works as context window manager for the Gemini model call.
    """

    def __init__(self, gemini_client: genai.Client, collection_name: str = "gemini_cli_memory"):
        """
        Initialize the ChromaMemory.

        Args:
            gemini_client: The Google GenAI client to use for any Gemini API calls.
            collection_name: The name of the Chroma collection to use for storing the memories.
        """
        super().__init__(gemini_client)
        self.chroma_client = chromadb.PersistentClient()
        self.memory_collection = self.chroma_client.get_or_create_collection(
            name=collection_name, embedding_function=GeminiEmbeddingFunction(self.gemini_client)
        )

    def _put(self, input: ContextWindowInput) -> ContextWindow:
        """
        Put a new context window into the Gemini memory.

        Args:
            input: The input context window to add to the Gemini memory.

        Returns:
            The new system instruction and contents of the context window enhanced with the relevant memories.
        """
        if not input.contents:
            return ContextWindow(system_instruction=input.system_instruction, contents=input.contents)

        last_user_message = self._get_last_user_message(input)

        system_instruction = input.system_instruction

        if last_user_message:
            results = self.memory_collection.query(query_texts=[last_user_message], n_results=10)
            if results["documents"]:
                retrieved_memories = "\n".join(results["documents"][0])
                system_instruction = self._get_new_system_instruction(
                    input.user_id, system_instruction, retrieved_memories
                )

        return ContextWindow(system_instruction=system_instruction, contents=input.contents)

    def list(self) -> List[str]:
        """
        Returns all memories from the ChromaDB collection.

        Returns:
            A list of all memories from the ChromaDB collection.
        """
        memories = self.memory_collection.get()
        return memories["documents"] if memories["documents"] else []

    def post(self, memory: MemoryEntry) -> None:
        """
        Add a new memory to the ChromaDB collection.
        """
        self.memory_collection.add(
            documents=[memory.content],
            ids=[memory.id],
            metadatas=[{"user_id": memory.user_id, "timestamp": memory.timestamp, "section_id": memory.section_id}],
        )


# Initialize Google GenAI client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable is not set, using default key")
    # raise ValueError("GEMINI_API_KEY environment variable is not set")
    GEMINI_API_KEY = "AIzaSyBmJ92tvO010bIwyX1C5WMJYBAG9sE-zY0"
genai_client = genai.Client(api_key=GEMINI_API_KEY)
logger.info("Gemini Client initialized successfully")

# Initialize Gemini memory
memory = ChromaMemory(genai_client)


with gr.Blocks(fill_height=True, fill_width=True) as demo:
    gr.Markdown("# Gemini CLI Memory Extension Server")
    with gr.Tab("Put"):

        def put_memory(user_id: str, system_instruction: str, contents: str) -> ContextWindow:
            return memory.put(
                ContextWindowInput(
                    user_id=user_id,
                    system_instruction=system_instruction,
                    contents=contents,
                )
            )

        gr.Interface(
            fn=put_memory,
            inputs=[
                gr.Textbox(label="User ID", value="user_1", elem_id="user_id"),
                gr.Textbox(label="System Instruction", elem_id="system_instruction"),
                gr.Textbox(label="Contents", elem_id="contents"),
            ],
            outputs=gr.Json(label="Memory Augmented Context Window", height=900, max_height=2000),
            title="Memory Augmentation",
            description="Receives a context window, enhances it with memories, and stores the interaction.",
            api_name="put",
            flagging_mode="never",
        )
    with gr.Tab("List"):
        view_button = gr.Button("View All Memories")
        memory_display = gr.Textbox(label="Stored Memories", lines=45, interactive=False)
        view_button.click(fn=memory.list, inputs=None, outputs=memory_display)
    with gr.Tab("Post"):
        content_input = gr.Textbox(label="Content", lines=5)
        user_id_input = gr.Textbox(label="User ID", value="user_1")
        section_id_input = gr.Textbox(label="Section ID", value="user_profile")
        post_button = gr.Button("Post Memory")

        def post_memory(content: str, user_id: str, section_id: str) -> None:
            memory.post(
                MemoryEntry(
                    content=content, user_id=user_id, timestamp=datetime.now().isoformat(), section_id=section_id
                )
            )

        post_button.click(
            fn=post_memory,
            inputs=[content_input, user_id_input, section_id_input],
            outputs=None,
        ).success(fn=lambda: gr.Info("Memory posted successfully"), inputs=None, outputs=None).then(
            fn=lambda: gr.update(value=""),
            inputs=None,
            outputs=[content_input],
        )
    with gr.Tab("Session Log"):

        def render_session_log() -> Dict[str, Dict[str, Any]]:
            dict_list: Dict[str, Dict[str, Any]] = {}
            for i, item in enumerate(memory.session_log):
                dict_list[str(i)] = item.model_dump()
            return dict_list

        DEFAULT_ENABLED = True
        memory.session_log_enabled = DEFAULT_ENABLED
        checkbox = gr.Checkbox(label="Enable Session Log", value=DEFAULT_ENABLED)
        view_button = gr.Button("View Session Log")
        session_log_display = gr.JSON(label="Session Log", height=900, max_height=2000, show_indices=True)
        view_button.click(fn=render_session_log, inputs=None, outputs=session_log_display)
        checkbox.change(fn=memory.set_session_log_enabled, inputs=checkbox, outputs=None)


def main() -> None:
    demo.launch()


# Comment the following two lines to work with flask (fastapi)
# if __name__ == "__main__":
#     main()
# Comment the next lines and uncomment the previous two to work with gradio only
# Initialize FastAPI app
app = FastAPI()


@app.post("/api/put")
async def put_memory_endpoint(data: ContextWindowInput) -> ContextWindow:
    """
    Process a memory input and return the memory augmented context window.
    """

    logger.info(f"Put called with user_id: {data.user_id}")
    return memory.put(data)


app = gr.mount_gradio_app(app, demo, path="/ui")
