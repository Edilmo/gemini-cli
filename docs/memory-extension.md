# Memory Extension

The Gemini CLI supports a powerful extensibility point that allows for the dynamic modification of prompts sent to the language model. This is achieved through the `PromptEnhancer` interface, which enables the creation of extensions that can inspect, modify, and enhance the conversation history and system instructions before they are processed by the model.

This document provides an overview of the `PromptEnhancer` interface and walks through an example of a memory extension that uses a local vector database to provide the model with long-term memory.

## The `PromptEnhancer` Interface

The `PromptEnhancer` interface is defined in `packages/core/src/config/config.ts` and has the following structure:

```typescript
export interface PromptEnhancer {
  name: string;
  enhance(
    userId: string,
    systemInstruction: string | Part | undefined,
    contents: Content[],
  ): Promise<{
    systemInstruction: string | Part | undefined;
    contents: Content[];
  }>;
}
```

Any object that implements this interface can be loaded by the CLI and will have its `enhance` method called for every prompt. The method receives:

- `userId`: The user identifier from the session
- `systemInstruction`: The current system instruction
- `contents`: The conversation contents

The method is expected to return them, potentially modified.

## Example: A Memory Extension with Vector Database

We have created a memory extension that uses a local vector database (ChromaDB) and a FastAPI-based Python server to provide the CLI with long-term memory capabilities.

### Architecture

1. **The Extension:** The TypeScript extension (`gca_extension/index.ts`) intercepts prompts and sends them to a local Python server via FastAPI.
2. **The Server:** The Python server (`gca_extension/python/src/gca_memory_simulator/gradio_app.py`) receives the conversation, generates embeddings for the user's latest message, and queries a ChromaDB database for relevant past interactions.
3. **Enhancement:** Retrieved memories are prepended to the system instruction, providing the model with relevant context from past conversations.
4. **Storage:** The server stores current user messages and model responses in the database, continually expanding its memory.

### Setup Instructions

#### 1. Install Python Dependencies

Navigate to the Python server directory and install dependencies using `uv`:

```bash
cd gca_extension/python
uv sync
```

#### 2. Set Up Environment Variables

Ensure you have the `GEMINI_API_KEY` environment variable set:

```bash
export GEMINI_API_KEY=your_api_key_here
```

#### 3. Start the Python Server

Run the FastAPI server using uvicorn:

```bash
uv run uvicorn gca_memory_simulator.gradio_app:app --reload --port 7860
```

This will start:

- A FastAPI server at `http://127.0.0.1:7860` with the memory enhancement API
- A Gradio web interface at `http://127.0.0.1:7860/ui` for administration

#### 4. Install the Extension

For the CLI to load the extension, it needs to be in a `.gemini/extensions` directory in either your workspace or home directory.

**Option A: Workspace Installation**

```bash
mkdir -p .gemini/extensions/memory-extension
cp gca_extension/gemini-extension.json .gemini/extensions/memory-extension
cp gca_extension/index.ts .gemini/extensions/memory-extension
```

**Option B: Global Installation**

```bash
mkdir -p ~/.gemini/extensions/memory-extension
cp gca_extension/gemini-extension.json ~/.gemini/extensions/memory-extension
cp gca_extension/index.ts ~/.gemini/extensions/memory-extension
```

**Option C: Development Symlink**

```bash
mkdir -p .gemini/extensions
ln -s "$(pwd)/gca_extension" .gemini/extensions/memory-extension
```

#### 5. Build the Extension

Since the extension is written in TypeScript, you need to compile it to JavaScript:

```bash
cd .gemini/extensions/memory-extension
# Compile TypeScript to JavaScript
npx tsc index.ts --target es2022 --module es2022 --moduleResolution node
```

### Using the Extension

Once the server is running and the extension is installed, run the Gemini CLI:

```bash
gemini
```

You should see a message indicating the memory extension has been loaded:

```
Loading extension: simple-memory-extension (version: 1.0.0)
```

The extension will:

- Automatically enhance prompts with relevant memories from past conversations
- Store new interactions in the vector database
- Provide seamless long-term memory capabilities

### API Contract

The extension communicates with the Python server using the following API:

**Endpoint:** `POST /api/put`

**Request Body:**

```json
{
  "user_id": "string",
  "system_instruction": "string or structured content",
  "contents": [
    {
      "parts": [
        {
          "text": "user message"
        }
      ],
      "role": "user"
    }
  ]
}
```

**Response:**

```json
{
  "system_instruction": "enhanced system instruction with memories",
  "contents": [
    {
      "parts": [
        {
          "text": "user message"
        }
      ],
      "role": "user"
    }
  ]
}
```

### Administration Interface

The Gradio web interface at `http://127.0.0.1:7860/ui` provides:

- **Put**: Enhance context windows with memories and store interactions
- **Post**: Add new memories to specific sections and properties
- **List**: View all stored memories in the database
- **Search**: Search through memories using semantic search
- **Session Log**: View the log of calls to the put API endpoint

### Testing

An E2E test is available to verify the extension works correctly:

```bash
# Run the specific memory extension test
npm test -- integration-tests/memory-extension.test.js
```

The test will:

1. Start the Python server
2. Set up the extension in a test environment
3. Test memory storage and retrieval
4. Verify graceful handling of server downtime

### Error Handling

The extension includes robust error handling:

- If the Python server is unavailable, prompts are passed through unchanged
- Network errors are logged but don't interrupt the CLI operation
- Invalid responses from the server are handled gracefully

### Memory Persistence

Memories are stored in a ChromaDB database that persists across sessions. The database is created automatically and stored locally, ensuring your conversation history is maintained between CLI sessions.

This extension demonstrates how the `PromptEnhancer` interface can be used to create sophisticated extensions that augment the functionality of the Gemini CLI with long-term memory capabilities.
