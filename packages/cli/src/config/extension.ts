/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { MCPServerConfig, PromptEnhancer } from '@google/gemini-cli-core';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { pathToFileURL } from 'url';
// Alias to avoid conflicts with the banner that esbuild injects
import { createRequire as createRequireForExt } from 'module';

export const EXTENSIONS_DIRECTORY_NAME = path.join('.gemini', 'extensions');
export const EXTENSIONS_CONFIG_FILENAME = 'gemini-extension.json';

export interface Extension {
  config: ExtensionConfig;
  contextFiles: string[];
  promptEnhancer?: PromptEnhancer;
}

export interface ExtensionConfig {
  name: string;
  version: string;
  mcpServers?: Record<string, MCPServerConfig>;
  contextFileName?: string | string[];
  excludeTools?: string[];
  promptEnhancerPath?: string;
}

const require = createRequireForExt(import.meta.url); // initialize once

export async function loadExtensions(
  workspaceDir: string,
): Promise<Extension[]> {
  const allExtensions = [
    ...(await loadExtensionsFromDir(workspaceDir)),
    ...(await loadExtensionsFromDir(os.homedir())),
  ];

  const uniqueExtensions: Extension[] = [];
  const seenNames = new Set<string>();
  for (const extension of allExtensions) {
    if (!seenNames.has(extension.config.name)) {
      console.log(
        `Loading extension: ${extension.config.name} (version: ${extension.config.version})`,
      );
      uniqueExtensions.push(extension);
      seenNames.add(extension.config.name);
    }
  }

  return uniqueExtensions;
}

async function loadExtensionsFromDir(dir: string): Promise<Extension[]> {
  const extensionsDir = path.join(dir, EXTENSIONS_DIRECTORY_NAME);
  if (!fs.existsSync(extensionsDir)) {
    return [];
  }

  const extensions: Extension[] = [];
  for (const subdir of fs.readdirSync(extensionsDir)) {
    const extensionDir = path.join(extensionsDir, subdir);

    const extension = await loadExtension(extensionDir);
    if (extension != null) {
      extensions.push(extension);
    }
  }
  return extensions;
}

async function loadExtension(extensionDir: string): Promise<Extension | null> {
  if (!fs.statSync(extensionDir).isDirectory()) {
    console.error(
      `Warning: unexpected file ${extensionDir} in extensions directory.`,
    );
    return null;
  }

  const configFilePath = path.join(extensionDir, EXTENSIONS_CONFIG_FILENAME);
  if (!fs.existsSync(configFilePath)) {
    console.error(
      `Warning: extension directory ${extensionDir} does not contain a config file ${configFilePath}.`,
    );
    return null;
  }

  try {
    const configContent = fs.readFileSync(configFilePath, 'utf-8');
    const config = JSON.parse(configContent) as ExtensionConfig;
    if (!config.name || !config.version) {
      console.error(
        `Invalid extension config in ${configFilePath}: missing name or version.`,
      );
      return null;
    }

    const contextFiles = getContextFileNames(config)
      .map((contextFileName) => path.join(extensionDir, contextFileName))
      .filter((contextFilePath) => fs.existsSync(contextFilePath));

    const extension: Extension = {
      config,
      contextFiles,
    };

    if (config.promptEnhancerPath) {
      const modifierPath = path.join(extensionDir, config.promptEnhancerPath);
      if (fs.existsSync(modifierPath)) {
        // 1. resolve any symlinks (/var -> /private/var on macOS)
        const realPath = fs.realpathSync(modifierPath);

        let mod: any;
        if (process.env.VITEST) {
          // ⬅️  Bypass Vite: load with CJS require
          mod = require(realPath);
        } else {
          // normal runtime – native ESM import
          mod = await import(/* @vite-ignore */ pathToFileURL(realPath).href);
        }

        if (
          typeof mod.default === 'object' &&
          mod.default.enhance &&
          mod.default.name
        ) {
          extension.promptEnhancer = mod.default;
        } else {
          console.error(
            `Warning: prompt enhancer ${modifierPath} does not have a default export that follows the PromptEnhancer interface. Here's the object: ${JSON.stringify(mod.default)}`,
          );
        }
      } else {
        console.error(
          `Warning: prompt enhancer file not found at ${modifierPath}.`,
        );
      }
    }

    return extension;
  } catch (e) {
    console.error(
      `Warning: error parsing extension config in ${configFilePath}: ${e}`,
    );
    return null;
  }
}

function getContextFileNames(config: ExtensionConfig): string[] {
  if (!config.contextFileName) {
    return ['GEMINI.md'];
  } else if (!Array.isArray(config.contextFileName)) {
    return [config.contextFileName];
  }
  return config.contextFileName;
}
