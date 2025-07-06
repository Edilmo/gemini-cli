/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Content, ContentUnion } from '@google/genai';
import { PromptEnhancer } from '@google/gemini-cli-core';

const memoryServerUrl = 'http://127.0.0.1:7860/api/put';

interface ContextWindowInput {
  user_id: string;
  system_instruction?: ContentUnion;
  contents: Content[];
}

interface ContextWindowOutput {
  system_instruction?: ContentUnion;
  contents: Content[];
}

async function enhanceWithMemory(
  userId: string,
  systemInstruction: ContentUnion | undefined,
  contents: Content[],
): Promise<{
  systemInstruction: ContentUnion | undefined;
  contents: Content[];
}> {
  try {
    const requestBody: ContextWindowInput = {
      user_id: userId,
      system_instruction: systemInstruction,
      contents: contents,
    };

    const response = await fetch(memoryServerUrl, {
      method: 'POST',
      headers: {
        accept: 'application/json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      console.error(
        `Memory server request failed with status: ${response.status}`,
      );
      return {
        systemInstruction,
        contents,
      };
    }

    const jsonResponse = (await response.json()) as ContextWindowOutput;

    return {
      systemInstruction: jsonResponse.system_instruction ?? systemInstruction,
      contents: jsonResponse.contents ?? contents,
    };
  } catch (error) {
    console.error('Error contacting memory server:', error);
    return {
      systemInstruction,
      contents,
    };
  }
}

const memoryEnhancer: PromptEnhancer = {
  name: 'memory-enhancer',
  async enhance(
    userId: string,
    systemInstruction: ContentUnion | undefined,
    contents: Content[],
  ): Promise<{
    systemInstruction: ContentUnion | undefined;
    contents: Content[];
  }> {
    return enhanceWithMemory(userId, systemInstruction, contents);
  },
};

export default memoryEnhancer;
