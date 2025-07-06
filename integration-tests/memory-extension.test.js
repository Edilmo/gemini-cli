/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { test, describe, before, after } from 'node:test';
import { strict as assert } from 'node:assert';
import { TestRig } from './test-helper.js';
import { spawn } from 'child_process';
import { join } from 'path';
import { fileURLToPath } from 'url';
import { writeFileSync, mkdirSync, cpSync } from 'fs';
import esbuild from 'esbuild';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const projectRoot = join(__dirname, '..');

describe('memory-extension', () => {
  const rig = new TestRig();
  let pythonServer;

  before(async () => {
    // Start the Python server
    pythonServer = spawn(
      'uv',
      [
        'run',
        'uvicorn',
        'gca_memory_simulator.gradio_app:app',
        '--reload',
        '--port',
        '7860',
      ],
      {
        cwd: join(projectRoot, 'gca_extension', 'python'),
        stdio: ['pipe', 'pipe', 'pipe'],
        env: {
          ...process.env,
          GEMINI_API_KEY:
            process.env.GEMINI_API_KEY ??
            (() => {
              throw new Error('GEMINI_API_KEY not set');
            })(),
        },
      },
    );

    // Log server output for debugging
    pythonServer.stdout.on('data', (data) => {
      console.log(`Python server stdout: ${data}`);
    });

    pythonServer.stderr.on('data', (data) => {
      console.error(`Python server stderr: ${data}`);
    });

    // Wait for the server to be ready
    await new Promise((resolve) => setTimeout(resolve, 5000));
  });

  after(() => {
    if (pythonServer) {
      pythonServer.kill();
    }
  });

  test('should enhance prompt with memory', async (t) => {
    rig.setup(t.name);

    // Create extension directory structure
    const extensionsDir = join(rig.testDir, '.gemini', 'extensions');
    const extensionDir = join(extensionsDir, 'memory-extension');
    mkdirSync(extensionDir, { recursive: true });

    // Copy extension files
    cpSync(
      join(projectRoot, 'gca_extension', 'gemini-extension.json'),
      join(extensionDir, 'gemini-extension.json'),
    );

    // Compile the actual extension TypeScript file to JavaScript using esbuild
    const extensionTsPath = join(projectRoot, 'gca_extension', 'index.ts');
    const extensionJsPath = join(extensionDir, 'index.js');

    await esbuild.build({
      entryPoints: [extensionTsPath],
      outfile: extensionJsPath,
      platform: 'node',
      format: 'esm',
      target: 'node20',
      bundle: false, // Don't bundle dependencies, just compile TS to JS
    });

    // Create a simple test file to work with
    rig.createFile('test.txt', 'This is a test file.');

    // First, save some information to memory
    const saveResult = rig.run(
      'Remember that the best beaches are in Venezuela.',
    );
    console.log('Save result:', saveResult);

    // Then try to retrieve it
    const retrieveResult = rig.run('Where are the best beaches in the world?');
    console.log('Retrieve result:', retrieveResult);

    // The response should contain information about TypeScript preference
    assert.ok(
      retrieveResult.toLowerCase().includes('venezuela'),
      'Response should contain memory about Venezuela',
    );
  });

  test('should work without memory server', async (t) => {
    rig.setup(t.name);

    // Create extension directory structure but don't start the server
    const extensionsDir = join(rig.testDir, '.gemini', 'extensions');
    const extensionDir = join(extensionsDir, 'memory-extension');
    mkdirSync(extensionDir, { recursive: true });

    // Copy extension files
    cpSync(
      join(projectRoot, 'gca_extension', 'gemini-extension.json'),
      join(extensionDir, 'gemini-extension.json'),
    );

    // Create a minimal extension that simulates server being down
    const extensionContent = `/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

const memoryEnhancer = {
  name: 'memory-enhancer',
  async enhance(userId, systemInstruction, contents) {
    // Simulate server being down - just return unchanged
    return {
      systemInstruction,
      contents,
    };
  },
};

export default memoryEnhancer;
`;

    writeFileSync(join(extensionDir, 'index.js'), extensionContent);

    // This should work even without the memory server
    const result = rig.run('What is 2 + 2?');
    assert.ok(result.includes('4'), 'Should work without memory server');
  });
});
