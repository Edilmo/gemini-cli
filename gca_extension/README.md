# Google Cloud Assist (GCA) Memory Extension

This directory contains the Google Cloud Assist Memory Extension for the Gemini CLI, which provides memory-enhanced conversation capabilities using Google Cloud APIs.

## New Requirements not met with current implementation:

- FR: https://github.com/google-gemini/gemini-cli/issues/2779
- Taylor’s vision related to extensions TaylorGeminiCLIVision_3Month: Vision: The "VSCode of Terminals".... We will deliver a system where extensions can transform Gemini CLI's core "persona," allowing it to become a specialized tool for any vertical (i.e. data scientist, SRE, designer, …). This will be enabled by a robust CLI extension workflow (publish, install, config) that empowers a vibrant community to create, share, and manage these extensions, with the first community marketplaces beginning to emerge.
- Be extension first: details can be seen in this Epic.
- Provide easy (or direct) migration paths from Claude Code.
- Frame the contribution as Hooks support in the extension system.
- Facilitate Claude Code hooks migration to GeminiCLI.
  * Claude Code Repo bash_command_validator_example.py
  * Hook Output Reference
- Create an issue hierarchy in the OSS GeminiCLI public roadmap (just like the link above) that reflects everything above.
- Change our implementation accordingly.
- Split the implementation in reviewable small PRs.
- Goes through OSS PR process for each of the PRs.
- Ensure alignment with ADK Callback system
- Hooks to consider:
  * all on Claude code
  * One before calling the LLM -> memory
  * One before constructing the list of tools -> tool optimizer


## Files Overview

### Core Files

- **`index.js`** - Original memory enhancer for local testing with Python memory server
  - Connects to local memory server at `http://127.0.0.1:7860/api/put`
  - Used for development and testing with the Python Gradio app
  - Simpler implementation for local development

- **`index-gca.js`** - Production memory enhancer for Google Cloud Platform
  - Connects to Google Cloud Assist Core API at `https://cloudassistcore-pa.googleapis.com`
  - Implements the `PromptEnhancer` interface with proper type conversions
  - Handles authentication via Google Cloud credentials
  - Production-ready implementation

### Configuration Files

- **`gemini-extension.json`** - Extension configuration for local testing (uses `index.js`)
- **`gemini-extension-gca.json`** - Extension configuration for GCP API (uses `index-gca.js`)

### Testing and Build Files

- **`package.json`** - Node.js dependencies for testing
- **`test-memory-enhancer.js`** - Test script for isolated testing
- **`node_modules/`** - Dependencies (created after `npm install`)

## Building and Testing

### Prerequisites

1. **Node.js** (v20 or higher)
2. **Google Cloud credentials** configured at `~/.config/gcloud/application_default_credentials.json`
3. **Appropriate GCP permissions** for Cloud Assist Core API

### Setup

```bash
# Navigate to the extension directory
cd gca_extension

# Install dependencies
npm install
```

### Testing the GCP Memory Enhancer

```bash
# Run the test suite
npm test

# Or run directly
node test-memory-enhancer.js
```

The test will:
- Load the memory enhancer
- Test with sample conversation data
- Make API calls to Google Cloud Assist Core API
- Display input/output for verification
- Handle API failures gracefully

### Expected Test Output

```
Testing Memory Enhancer...
Input:
- userId: test-user-123
- systemInstruction: You are a helpful assistant.
- contents: [Array of Content objects]

Output:
- systemInstruction: [Enhanced or original instruction]
- contents: [Enhanced or original contents]

✅ Test completed successfully!
```

## Usage with Gemini CLI

### For Local Development (Python Server)

1. Start the Python memory server:
   ```bash
   cd python
   python -m gca_memory_simulator.gradio_app
   ```

2. Use the local extension configuration:
   ```bash
   gemini --extension-file gca_extension/gemini-extension.json
   ```

### For Production (GCP API)

1. Ensure Google Cloud credentials are configured:
   ```bash
   gcloud auth application-default login
   ```

2. Use the GCP extension configuration:
   ```bash
   gemini --extension-file gca_extension/gemini-extension-gca.json
   ```

## Architecture

### Type Conversions

The `index-gca.js` handles complex type conversions between:
- **Input**: `ContentUnion | undefined` (systemInstruction) and `Content[]` (contents)
- **API**: Plain strings for `prompt` and `messages[]`
- **Output**: Original types preserved with enhanced content

### Key Functions

- **`toContent()`** - Converts ContentUnion to Content objects
- **`contentToString()`** - Extracts text from Content for API calls
- **`contentsToStrings()`** - Converts Content arrays to string arrays
- **`stringsToContents()`** - Converts API strings back to Content objects
- **`stringToContentUnion()`** - Restores original ContentUnion types

### Error Handling

- **Authentication failures** - Graceful fallback to original content
- **API failures** - Returns original values unchanged
- **Network errors** - Comprehensive error logging
- **Type mismatches** - Robust type checking and conversion

## Configuration

### Memory Access Options

The extension uses the following default configuration:
- **`maxTotalTokens`**: 700,000 (configurable in `index-gca.js`)
- **API Endpoint**: `https://cloudassistcore-pa.googleapis.com/v1/coretools:memoryaccess`
- **Authentication**: Google Cloud Application Default Credentials

### Customization

To modify the memory access behavior:

1. Edit the `options` object in `index-gca.js`:
   ```javascript
   const options = {
     maxTotalTokens: 700000  // Adjust as needed
   };
   ```

2. Update the API endpoint if using a different environment:
   ```javascript
   const CLOUD_ASSIST_CORE_API_URL = "https://your-custom-endpoint.googleapis.com";
   ```

## Troubleshooting

### Common Issues

1. **Import errors** - Ensure Node.js supports ES modules (v20+)
2. **Authentication failures** - Check Google Cloud credentials
3. **API 404 errors** - Verify API endpoint availability
4. **Type errors** - Ensure proper Content object structure

### Debug Mode

Add console logging to track the conversion process:
```javascript
console.log('Original systemInstruction:', systemInstruction);
console.log('Converted to string:', systemInstructionString);
console.log('API response:', jsonResponse);
```

## Development

### Adding New Features

1. **Extend the API request** - Modify `requestBody` in `enhanceWithMemory()`
2. **Add new conversion types** - Create new conversion functions
3. **Update tests** - Add test cases in `test-memory-enhancer.js`
4. **Update documentation** - Keep this README and docs up to date

### Testing Changes

Always test both:
- **Unit tests** - `npm test`
- **Integration tests** - With actual Gemini CLI usage

## Security Considerations

- **Credentials** - Never commit Google Cloud credentials to version control
- **API Keys** - Use environment variables for sensitive configuration
- **Network** - All API calls use HTTPS
- **Error Handling** - Sensitive information is not logged in errors 
