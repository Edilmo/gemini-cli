/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  MCPServerConfig,
  GeminiCLIExtension,
  Storage,
  PromptEnhancer,
} from '@google/gemini-cli-core';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { pathToFileURL } from 'url';
// Alias to avoid conflicts with the banner that esbuild injects
import { createRequire as createRequireForExt } from 'module';

export const EXTENSIONS_CONFIG_FILENAME = 'gemini-extension.json';

export interface Extension {
  path: string;
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

  const uniqueExtensions = new Map<string, Extension>();
  for (const extension of allExtensions) {
    if (!uniqueExtensions.has(extension.config.name)) {
      uniqueExtensions.set(extension.config.name, extension);
    }
  }

  return Array.from(uniqueExtensions.values());
}

async function loadExtensionsFromDir(dir: string): Promise<Extension[]> {
  const storage = new Storage(dir);
  const extensionsDir = storage.getExtensionsDir();
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

    return {
      path: extensionDir,
      config,
      contextFiles,
    };

    if (config.promptEnhancerPath) {
      const modifierPath = path.join(extensionDir, config.promptEnhancerPath);
      if (fs.existsSync(modifierPath)) {
        // 1. resolve any symlinks (/var -> /private/var on macOS)
        const realPath = fs.realpathSync(modifierPath);

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        let mod: any;
        if (process.env.VITEST) {
          // ⬅️  Bypass Vite: load with CJS require
          // eslint-disable-next-line no-restricted-syntax
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

export function annotateActiveExtensions(
  extensions: Extension[],
  enabledExtensionNames: string[],
): GeminiCLIExtension[] {
  const annotatedExtensions: GeminiCLIExtension[] = [];

  if (enabledExtensionNames.length === 0) {
    return extensions.map((extension) => ({
      name: extension.config.name,
      version: extension.config.version,
      isActive: true,
      path: extension.path,
    }));
  }

  const lowerCaseEnabledExtensions = new Set(
    enabledExtensionNames.map((e) => e.trim().toLowerCase()),
  );

  if (
    lowerCaseEnabledExtensions.size === 1 &&
    lowerCaseEnabledExtensions.has('none')
  ) {
    return extensions.map((extension) => ({
      name: extension.config.name,
      version: extension.config.version,
      isActive: false,
      path: extension.path,
    }));
  }

  const notFoundNames = new Set(lowerCaseEnabledExtensions);

  for (const extension of extensions) {
    const lowerCaseName = extension.config.name.toLowerCase();
    const isActive = lowerCaseEnabledExtensions.has(lowerCaseName);

    if (isActive) {
      notFoundNames.delete(lowerCaseName);
    }

    annotatedExtensions.push({
      name: extension.config.name,
      version: extension.config.version,
      isActive,
      path: extension.path,
    });
  }

  for (const requestedName of notFoundNames) {
    console.error(`Extension not found: ${requestedName}`);
  }

  return annotatedExtensions;
}
