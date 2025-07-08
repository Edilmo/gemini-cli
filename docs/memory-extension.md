# Memory Extension

The Gemini CLI supports a powerful extensibility point that allows for the dynamic modification of prompts sent to the language model. This is achieved through the `PromptEnhancer` interface, which enables the creation of extensions that can inspect, modify, and enhance the conversation history and system instructions before they are processed by the model.

This document provides an overview of the `PromptEnhancer` interface and walks through two implementations of memory extensions:

1. **Local Development Version**: Uses a local vector database (ChromaDB) and Python server
2. **Production Version**: Uses Google Cloud Assist Core API for enterprise-grade memory capabilities

## The `PromptEnhancer` Interface

The `PromptEnhancer` interface is defined in `packages/core/src/config/config.ts` and has the following structure:

```typescript
export interface PromptEnhancer {
  name: string;
  enhance(
    userId: string,
    systemInstruction: ContentUnion | undefined,
    contents: Content[],
  ): Promise<{
    systemInstruction: ContentUnion | undefined;
    contents: Content[];
  }>;
}
```

Any object that implements this interface can be loaded by the CLI and will have its `enhance` method called for every prompt. The method receives:

- `userId`: The user identifier from the session
- `systemInstruction`: The current system instruction (can be string, Part, or Content)
- `contents`: The conversation contents as Content objects

The method is expected to return them, potentially modified.

## Memory Extension Implementations

### 1. Local Development Version (Python Server)

This implementation uses a local vector database (ChromaDB) and a FastAPI-based Python server to provide the CLI with long-term memory capabilities.

#### Architecture

1. **The Extension:** The JavaScript extension (`gca_extension/index.js`) intercepts prompts and sends them to a local Python server via FastAPI.
2. **The Server:** The Python server (`gca_extension/python/src/gca_memory_simulator/gradio_app.py`) receives the conversation, generates embeddings for the user's latest message, and queries a ChromaDB database for relevant past interactions.
3. **Enhancement:** Retrieved memories are prepended to the system instruction, providing the model with relevant context from past conversations.
4. **Storage:** The server stores current user messages and model responses in the database, continually expanding its memory.

#### Setup Instructions

##### 1. Install Python Dependencies

Navigate to the Python server directory and install dependencies using `uv`:

```bash
cd gca_extension/python
uv sync
```

##### 2. Set Up Environment Variables

Ensure you have the `GEMINI_API_KEY` environment variable set:

```bash
export GEMINI_API_KEY=your_api_key_here
```

##### 3. Start the Python Server

Run the FastAPI server using uvicorn:

```bash
uv run uvicorn gca_memory_simulator.gradio_app:app --reload --port 7860
```

This will start:

- A FastAPI server at `http://127.0.0.1:7860` with the memory enhancement API
- A Gradio web interface at `http://127.0.0.1:7860/ui` for administration

##### 4. Use the Local Extension

**Option A: Direct Extension File (Recommended)**

```bash
gemini --extension-file gca_extension/gemini-extension.json
```

**Option B: Install to Extensions Directory**

For the CLI to automatically load the extension, install it to a `.gemini/extensions` directory:

_Workspace Installation:_

```bash
mkdir -p .gemini/extensions/memory-extension
cp gca_extension/gemini-extension.json .gemini/extensions/memory-extension/
cp gca_extension/index.js .gemini/extensions/memory-extension/
```

_Global Installation:_

```bash
mkdir -p ~/.gemini/extensions/memory-extension
cp gca_extension/gemini-extension.json ~/.gemini/extensions/memory-extension/
cp gca_extension/index.js ~/.gemini/extensions/memory-extension/
```

_Development Symlink:_

```bash
mkdir -p .gemini/extensions
ln -s "$(pwd)/gca_extension" .gemini/extensions/memory-extension
```

Then run the CLI normally:

```bash
gemini
```

### 2. Production Version (Google Cloud Platform)

This implementation uses Google Cloud Assist Core API for enterprise-grade memory capabilities with proper authentication and scalability.

#### Architecture

1. **The Extension:** The JavaScript extension (`gca_extension/index-gca.js`) implements the `PromptEnhancer` interface with proper type conversions.
2. **Authentication:** Uses Google Cloud Application Default Credentials for secure API access.
3. **Type Handling:** Converts between `ContentUnion`/`Content` types and the API's string format.
4. **API Integration:** Calls the Google Cloud Assist Core Memory Access API for enhanced context.

#### Setup Instructions

##### 1. Install Dependencies

Navigate to the extension directory and install Node.js dependencies:

```bash
cd gca_extension
npm install
```

##### 2. Configure Google Cloud Authentication

Set up Google Cloud credentials:

```bash
gcloud auth application-default login
```

Ensure your credentials are available at:

```
~/.config/gcloud/application_default_credentials.json
```

##### 3. Test the Extension

Run the test suite to verify functionality:

```bash
cd gca_extension
npm test
```

##### 4. Use the GCP Extension

**Option A: Direct Extension File (Recommended)**

```bash
gemini --extension-file gca_extension/gemini-extension-gca.json
```

**Option B: Install to Extensions Directory**

For the CLI to automatically load the extension, install it to a `.gemini/extensions` directory:

_Workspace Installation:_

```bash
mkdir -p .gemini/extensions/gca-memory-extension
cp gca_extension/gemini-extension-gca.json .gemini/extensions/gca-memory-extension/
cp gca_extension/index-gca.js .gemini/extensions/gca-memory-extension/
```

_Global Installation:_

```bash
mkdir -p ~/.gemini/extensions/gca-memory-extension
cp gca_extension/gemini-extension-gca.json ~/.gemini/extensions/gca-memory-extension/
cp gca_extension/index-gca.js ~/.gemini/extensions/gca-memory-extension/
```

_Development Symlink:_

```bash
mkdir -p .gemini/extensions
ln -s "$(pwd)/gca_extension" .gemini/extensions/gca-memory-extension
```

Then run the CLI normally:

```bash
gemini
```

You should see a message indicating the memory extension has been loaded:

```
Loading extension: gca-memory-extension (version: 1.0.0)
```

## Extension Configuration Files

### Local Development Configuration (`gemini-extension.json`)

```json
{
  "name": "simple-memory-extension",
  "version": "1.0.0",
  "description": "Memory extension using local Python server",
  "main": "index.js",
  "type": "prompt-enhancer"
}
```

### Production Configuration (`gemini-extension-gca.json`)

```json
{
  "name": "gca-memory-enhancer",
  "version": "1.0.0",
  "description": "Google Cloud Assist Memory Extension for enhanced conversations using GCP APIs",
  "main": "index-gca.js",
  "type": "prompt-enhancer",
  "capabilities": {
    "memory": true,
    "context_enhancement": true,
    "conversation_history": true
  },
  "configuration": {
    "api_endpoint": "https://cloudassistcore-pa.googleapis.com/v1/coretools:memoryaccess",
    "max_total_tokens": 32000,
    "authentication": "google_cloud_credentials"
  }
}
```

## API Contracts

### Local Python Server API

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

### Google Cloud Assist Core API

**Endpoint:** `POST /v1/coretools:memoryaccess`

**Request Body:**

```json
{
  "textRequest": {
    "prompt": "system instruction string",
    "messages": ["array", "of", "message", "strings"]
  },
  "inferenceContext": {
    "maxTotalTokens": "32000"
  }
}
```

**Response:**

```json
{
  "textRequest": {
    "prompt": "enhanced system instruction",
    "messages": ["enhanced", "message", "array"]
  },
  "memoryApplied": true
}
```

## Type Conversions (GCP Version)

The GCP extension handles complex type conversions:

### Input Processing

- `ContentUnion | undefined` → `Content` → `string`
- `Content[]` → `string[]`

### Output Processing

- API `string` responses → original `ContentUnion` types
- Preserves original structure (roles, parts, etc.)

### Key Functions

- **`toContent()`** - Converts ContentUnion to Content objects
- **`contentToString()`** - Extracts text from Content for API calls
- **`contentsToStrings()`** - Converts Content arrays to string arrays
- **`stringsToContents()`** - Converts API strings back to Content objects
- **`stringToContentUnion()`** - Restores original ContentUnion types

## Testing

### Local Version Testing

An E2E test is available to verify the extension works correctly:

```bash
# Run the specific memory extension test
npm test -- integration-tests/memory-extension.test.js
```

### GCP Version Testing

Test the GCP extension in isolation:

```bash
cd gca_extension
npm test
```

The test will:

1. Load the memory enhancer
2. Test with sample conversation data
3. Make API calls to Google Cloud Assist Core API
4. Display input/output for verification
5. Handle API failures gracefully

## Administration Interface (Local Version Only)

The Gradio web interface at `http://127.0.0.1:7860/ui` provides:

- **Put**: Enhance context windows with memories and store interactions
- **Post**: Add new memories to specific sections and properties
- **List**: View all stored memories in the database
- **Search**: Search through memories using semantic search
- **Session Log**: View the log of calls to the put API endpoint

## Error Handling

Both extensions include robust error handling:

- If the server/API is unavailable, prompts are passed through unchanged
- Network errors are logged but don't interrupt the CLI operation
- Invalid responses are handled gracefully
- Authentication failures (GCP version) fall back to original content

## Choosing Between Implementations

### Use Local Version When:

- Developing and testing locally
- Need full control over memory storage
- Want to customize memory retrieval logic
- Working in environments without GCP access

### Use GCP Version When:

- Deploying to production environments
- Need enterprise-grade scalability and reliability
- Want integrated Google Cloud authentication
- Require consistent performance across teams

## Security Considerations

### Local Version

- Memory data stored locally in ChromaDB
- No external data transmission (except to local server)
- API key required for embedding generation

### GCP Version

- Authenticated via Google Cloud credentials
- All communication over HTTPS
- No sensitive data logged in error messages
- Follows Google Cloud security standards

This extension demonstrates how the `PromptEnhancer` interface can be used to create sophisticated extensions that augment the functionality of the Gemini CLI with long-term memory capabilities, whether for local development or production deployment.
