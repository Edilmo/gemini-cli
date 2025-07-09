/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { vi, describe, it, expect, beforeEach, Mock } from 'vitest';
import { GeminiChat } from './geminiChat.js';
import { Config, PromptEnhancer } from '../config/config.js';
import { ContentGenerator } from './contentGenerator.js';
import { Content } from '@google/genai';

describe('GeminiChat with PromptEnhancer', () => {
  let mockConfig: Config;
  let mockContentGenerator: ContentGenerator;
  let chat: GeminiChat;

  beforeEach(() => {
    mockContentGenerator = {
      generateContent: vi.fn(),
      generateContentStream: vi.fn(),
      countTokens: vi.fn(),
      embedContent: vi.fn(),
    };

    const mockEnhancer: PromptEnhancer = {
      enhance: vi.fn(async (_userId, systemInstruction, contents) => {
        const newContents: Content[] = [
          ...contents,
          { role: 'user', parts: [{ text: 'enhanced' }] },
        ];
        return { systemInstruction, contents: newContents };
      }),
      name: 'mock-enhancer',
    };

    mockConfig = {
      getPromptEnhancers: vi.fn().mockReturnValue([mockEnhancer]),
      getModel: vi.fn().mockReturnValue('gemini-pro'),
      getSessionId: vi.fn().mockReturnValue('test-session-id'),
      getUsageStatisticsEnabled: vi.fn().mockReturnValue(false),
      getContentGeneratorConfig: vi
        .fn()
        .mockReturnValue({ authType: 'oauth-personal' }),
    } as unknown as Config;

    chat = new GeminiChat(mockConfig, mockContentGenerator);
  });

  it('should call the prompt enhancer and send the enhanced content', async () => {
    const response = {
      candidates: [
        { content: { role: 'model', parts: [{ text: 'response' }] } },
      ],
    };
    (mockContentGenerator.generateContent as unknown as Mock).mockResolvedValue(
      response,
    );

    await chat.sendMessage({ message: 'hello' });

    const enhancer = mockConfig.getPromptEnhancers()[0];
    expect(enhancer.enhance).toHaveBeenCalled();

    const generateContentCall = (mockContentGenerator.generateContent as Mock)
      .mock.calls[0][0];
    const sentContents = generateContentCall.contents;

    // The original history is empty, so it should have user message + enhanced message
    expect(sentContents).toHaveLength(2);
    expect(sentContents[0].parts[0].text).toBe('hello');
    expect(sentContents[1].parts[0].text).toBe('enhanced');
  });
});
