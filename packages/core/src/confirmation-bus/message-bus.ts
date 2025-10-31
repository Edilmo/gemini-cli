/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { randomUUID } from 'node:crypto';
import { EventEmitter } from 'node:events';
import type { PolicyEngine } from '../policy/policy-engine.js';
import { PolicyDecision } from '../policy/types.js';
import {
  MessageBusType,
  type Message,
  type HookExecutionRequest,
  type HookPolicyDecision,
} from './types.js';
import { safeJsonStringify } from '../utils/safeJsonStringify.js';

export class MessageBus extends EventEmitter {
  constructor(
    private readonly policyEngine: PolicyEngine,
    private readonly debug = false,
  ) {
    super();
    this.debug = debug;
  }

  private isValidMessage(message: Message): boolean {
    if (!message || !message.type) {
      return false;
    }

    if (
      message.type === MessageBusType.TOOL_CONFIRMATION_REQUEST &&
      !('correlationId' in message)
    ) {
      return false;
    }

    return true;
  }

  private emitMessage(message: Message): void {
    this.emit(message.type, message);
  }

  publish(message: Message): void {
    if (this.debug) {
      console.debug(`[MESSAGE_BUS] publish: ${safeJsonStringify(message)}`);
    }
    try {
      if (!this.isValidMessage(message)) {
        throw new Error(
          `Invalid message structure: ${safeJsonStringify(message)}`,
        );
      }

      if (message.type === MessageBusType.TOOL_CONFIRMATION_REQUEST) {
        const decision = this.policyEngine.check(message.toolCall);

        switch (decision) {
          case PolicyDecision.ALLOW:
            // Directly emit the response instead of recursive publish
            this.emitMessage({
              type: MessageBusType.TOOL_CONFIRMATION_RESPONSE,
              correlationId: message.correlationId,
              confirmed: true,
            });
            break;
          case PolicyDecision.DENY:
            // Emit both rejection and response messages
            this.emitMessage({
              type: MessageBusType.TOOL_POLICY_REJECTION,
              toolCall: message.toolCall,
            });
            this.emitMessage({
              type: MessageBusType.TOOL_CONFIRMATION_RESPONSE,
              correlationId: message.correlationId,
              confirmed: false,
            });
            break;
          case PolicyDecision.ASK_USER:
            // Pass through to UI for user confirmation
            this.emitMessage(message);
            break;
          default:
            throw new Error(`Unknown policy decision: ${decision}`);
        }
      } else if (message.type === MessageBusType.HOOK_EXECUTION_REQUEST) {
        // Handle hook execution requests through policy evaluation
        const hookRequest = message as HookExecutionRequest;
        const decision = this.policyEngine.checkHook(hookRequest);

        // Emit policy decision for observability
        this.emitMessage({
          type: MessageBusType.HOOK_POLICY_DECISION,
          eventName: hookRequest.eventName,
          hookSource:
            (hookRequest.input['hook_source'] as
              | 'project'
              | 'user'
              | 'system'
              | 'extension') || 'project',
          decision: decision === PolicyDecision.ALLOW ? 'allow' : 'deny',
          reason:
            decision === PolicyDecision.DENY
              ? 'Hook execution denied by policy'
              : undefined,
        } as HookPolicyDecision);

        // If allowed, emit the request for hook system to handle
        if (decision === PolicyDecision.ALLOW) {
          this.emitMessage(message);
        } else {
          // If denied, emit error response
          this.emitMessage({
            type: MessageBusType.HOOK_EXECUTION_RESPONSE,
            correlationId: hookRequest.correlationId,
            success: false,
            error: new Error('Hook execution denied by policy'),
          });
        }
      } else {
        // For all other message types, just emit them
        this.emitMessage(message);
      }
    } catch (error) {
      this.emit('error', error);
    }
  }

  subscribe<T extends Message>(
    type: T['type'],
    listener: (message: T) => void,
  ): void {
    this.on(type, listener);
  }

  unsubscribe<T extends Message>(
    type: T['type'],
    listener: (message: T) => void,
  ): void {
    this.off(type, listener);
  }

  /**
   * Request-response pattern: Publish a message and wait for a correlated response
   * This enables synchronous-style communication over the async MessageBus
   * The correlation ID is generated internally and added to the request
   */
  async request<TRequest extends Message, TResponse extends Message>(
    request: Omit<TRequest, 'correlationId'>,
    responseType: TResponse['type'],
    timeoutMs: number = 60000,
  ): Promise<TResponse> {
    const correlationId = randomUUID();

    return new Promise<TResponse>((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        cleanup();
        reject(new Error(`Request timed out waiting for ${responseType}`));
      }, timeoutMs);

      const cleanup = () => {
        clearTimeout(timeoutId);
        this.unsubscribe(responseType, responseHandler);
      };

      const responseHandler = (response: TResponse) => {
        // Check if this response matches our request
        if (
          'correlationId' in response &&
          response.correlationId === correlationId
        ) {
          cleanup();
          resolve(response);
        }
      };

      // Subscribe to responses
      this.subscribe<TResponse>(responseType, responseHandler);

      // Publish the request with correlation ID
      this.publish({ ...request, correlationId } as TRequest);
    });
  }
}
