/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
const memoryServerUrl = "http://127.0.0.1:7860/api/put";
async function enhanceWithMemory(userId, systemInstruction, contents) {
  try {
    const requestBody = {
      user_id: userId,
      system_instruction: systemInstruction,
      contents
    };
    const response = await fetch(memoryServerUrl, {
      method: "POST",
      headers: {
        accept: "application/json",
        "Content-Type": "application/json"
      },
      body: JSON.stringify(requestBody)
    });
    if (!response.ok) {
      console.error(
        `Memory server request failed with status: ${response.status}`
      );
      return {
        systemInstruction,
        contents
      };
    }
    const jsonResponse = await response.json();
    return {
      systemInstruction: jsonResponse.system_instruction ?? systemInstruction,
      contents: jsonResponse.contents ?? contents
    };
  } catch (error) {
    console.error("Error contacting memory server:", error);
    return {
      systemInstruction,
      contents
    };
  }
}
const memoryEnhancer = {
  name: "memory-enhancer",
  async enhance(userId, systemInstruction, contents) {
    return enhanceWithMemory(userId, systemInstruction, contents);
  }
};
var index_default = memoryEnhancer;
export {
  index_default as default
};
