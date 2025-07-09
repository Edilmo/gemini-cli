/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import * as path from 'path';
import * as os from 'os';
import {google} from 'googleapis';

/**
 * Google Auth client for authenticating with Google Cloud APIs.
 */
const auth = new google.auth.GoogleAuth({
  keyFile: path.join(os.homedir(), '.config', 'gcloud', 'application_default_credentials.json'),
  scopes: ['https://www.googleapis.com/auth/cloud-platform'],
});

/**
 * Base URL for the Cloud Assist Core API.
 */
// const CLOUD_ASSIST_CORE_API_URL = "https://cloudassistcore-pa.googleapis.com";
const CLOUD_ASSIST_CORE_API_URL = "https://cloudassistcore-pa-staging.googleapis.com";
// const CLOUD_ASSIST_CORE_API_URL = "https://cloudassistcore-pa.sandbox.googleapis.com";
const LOCAL_GCA_TOOLS_PROXY_URL = "http://localhost:8080";
const USE_LOCAL_GCA_TOOLS_PROXY = true;


/**
 * Maximum total tokens allowed for memory access requests.
 * This value is used to limit the size of the context window.
 */
const options = {
  maxTotalTokens: 700000
};

/**
 * Converts a ContentUnion to a Content object.
 * 
 * @param {any} content - The content to convert.
 * @returns {any} The converted Content object.
 */
function toContent(content) {
  if (!content) {
    return undefined;
  }
  
  if (Array.isArray(content)) {
    // it's a PartsUnion[]
    return {
      role: 'user',
      parts: content.map(part => typeof part === 'string' ? { text: part } : part),
    };
  }
  
  if (typeof content === 'string') {
    // it's a string
    return {
      role: 'user',
      parts: [{ text: content }],
    };
  }
  
  if ('parts' in content) {
    // it's already a Content
    return content;
  }
  
  // it's a Part
  return {
    role: 'user',
    parts: [content],
  };
}

/**
 * Extracts text content from a Content object for API calls.
 * 
 * @param {any} content - The Content object to extract text from.
 * @returns {string} The extracted text content.
 */
function contentToString(content) {
  if (!content || !content.parts) {
    return '';
  }
  
  return content.parts
    .map(part => part.text || '')
    .filter(text => text.length > 0)
    .join(' ');
}

/**
 * Converts an array of Content objects to an array of strings.
 * 
 * @param {any[]} contents - The array of Content objects.
 * @returns {string[]} The array of strings.
 */
function contentsToStrings(contents) {
  return contents.map(content => contentToString(content));
}

/**
 * Converts an array of strings back to Content objects, preserving the original structure.
 * 
 * @param {string[]} strings - The array of strings.
 * @param {any[]} originalContents - The original Content objects for structure reference.
 * @returns {any[]} The array of Content objects.
 */
function stringsToContents(strings, originalContents) {
  return strings.map((str, index) => {
    const originalContent = originalContents[index];
    if (originalContent && originalContent.parts) {
      // Preserve the original structure but update the text
      return {
        ...originalContent,
        parts: originalContent.parts.map(part => {
          if (part.text !== undefined) {
            return { ...part, text: str };
          }
          return part;
        })
      };
    }
    
    // Fallback to simple text content
    return {
      role: originalContent?.role || 'user',
      parts: [{ text: str }]
    };
  });
}

/**
 * Converts a string back to ContentUnion, preserving the original type.
 * 
 * @param {string} str - The string to convert.
 * @param {any} originalContent - The original ContentUnion for type reference.
 * @returns {any} The converted ContentUnion.
 */
function stringToContentUnion(str, originalContent) {
  if (!originalContent) {
    return undefined;
  }
  
  if (typeof originalContent === 'string') {
    return str;
  }
  
  if (Array.isArray(originalContent)) {
    // Return as array of parts
    return [{ text: str }];
  }
  
  if ('parts' in originalContent) {
    // Return as Content object, preserving structure
    return {
      ...originalContent,
      parts: originalContent.parts.map(part => {
        if (part.text !== undefined) {
          return { ...part, text: str };
        }
        return part;
      })
    };
  }
  
  // It was a Part, return as Part
  return { text: str };
}

/**
 * Enhances inference request with memory using Google Cloud Assist Core API.
 * 
 * @param {string} userId - The user ID for memory context.
 * @param {any} systemInstruction - The system instruction/prompt.
 * @param {any[]} contents - The conversation contents/messages.
 * @returns {Promise<{systemInstruction: any, contents: any[], memoryApplied?: boolean}>} Enhanced system instruction and contents.
 */
async function enhanceWithMemory(userId, systemInstruction, contents) {
  try {
    let accessToken = null;
    
    // Only get auth token if using cloud API
    if (!USE_LOCAL_GCA_TOOLS_PROXY) {
      const authClient = await auth.getClient();
      accessToken = await authClient.getAccessToken();
      
      if (!accessToken.token) {
        throw new Error('Failed to obtain access token');
      }
    }

    // Convert inputs to the format expected by the API
    const systemInstructionContent = toContent(systemInstruction);
    const systemInstructionString = systemInstructionContent ? contentToString(systemInstructionContent) : '';
    const contentsStrings = contentsToStrings(contents);

    // Prepare the request body according to the OpenAPI spec
    const requestBody = {
      textRequest: {
        prompt: systemInstructionString,
        messages: contentsStrings
      }
    };

    // Add inference context if maxTotalTokens is provided
    if (options.maxTotalTokens) {
      requestBody.inferenceContext = {
        maxTotalTokens: options.maxTotalTokens.toString()
      };
    }

    // Determine URL and headers based on proxy setting
    const apiUrl = USE_LOCAL_GCA_TOOLS_PROXY 
      ? `${LOCAL_GCA_TOOLS_PROXY_URL}/v1/coretools:memoryaccess`
      : `${CLOUD_ASSIST_CORE_API_URL}/v1/coretools:memoryaccess`;

    const headers = {
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    };

    // Add authorization header only for cloud API
    if (!USE_LOCAL_GCA_TOOLS_PROXY && accessToken) {
      headers['Authorization'] = `Bearer ${accessToken.token}`;
    }

    const response = await fetch(apiUrl, {
      method: "POST",
      headers,
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(
        `Cloud Assist Core API request failed with status: ${response.status}, error: ${errorText}`
      );
      
      // Return original values on API failure
      return {
        systemInstruction,
        contents,
        memoryApplied: false
      };
    }

    const jsonResponse = await response.json();
    
    // Extract the enhanced request from the response
    const enhancedTextRequest = jsonResponse.textRequest;
    const memoryApplied = jsonResponse.memoryApplied || false;

    // Convert the API response back to the original types
    const enhancedSystemInstruction = enhancedTextRequest?.prompt ? 
      stringToContentUnion(enhancedTextRequest.prompt, systemInstruction) : 
      systemInstruction;
    
    const enhancedContents = enhancedTextRequest?.messages ? 
      stringsToContents(enhancedTextRequest.messages, contents) : 
      contents;

    return {
      systemInstruction: enhancedSystemInstruction,
      contents: enhancedContents,
      memoryApplied
    };

  } catch (error) {
    console.error("Error contacting Cloud Assist Core API:", error);
    
    // Return original values on error
    return {
      systemInstruction,
      contents,
      memoryApplied: false
    };
  }
}

/**
 * Memory enhancer object that implements the PromptEnhancer interface.
 */
const memoryEnhancer = {
  name: "memory-enhancer",
  
  /**
   * Enhances the given system instruction and contents with memory.
   * 
   * @param {string} userId - The user ID for memory context.
   * @param {any} systemInstruction - The system instruction/prompt.
   * @param {any[]} contents - The conversation contents/messages.
   * @returns {Promise<{systemInstruction: any, contents: any[]}>} Enhanced system instruction and contents.
   */
  async enhance(userId, systemInstruction, contents) {
    const result = await enhanceWithMemory(userId, systemInstruction, contents);
    
    // Return only the required fields for the PromptEnhancer interface
    return {
      systemInstruction: result.systemInstruction,
      contents: result.contents
    };
  }
};

export default memoryEnhancer;
