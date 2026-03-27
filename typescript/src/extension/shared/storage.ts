import { type ExtensionConfig, DEFAULT_CONFIG } from './types';

const STORAGE_KEY = 'robutler_config';

export async function loadConfig(): Promise<ExtensionConfig> {
  try {
    const result = await chrome.storage.local.get(STORAGE_KEY);
    if (result[STORAGE_KEY]) {
      return { ...DEFAULT_CONFIG, ...result[STORAGE_KEY] };
    }
  } catch {
    // Outside extension context (tests, etc.)
  }
  return { ...DEFAULT_CONFIG };
}

export async function saveConfig(
  config: Partial<ExtensionConfig>,
): Promise<ExtensionConfig> {
  const current = await loadConfig();
  const merged = { ...current, ...config };
  try {
    await chrome.storage.local.set({ [STORAGE_KEY]: merged });
  } catch {
    // Outside extension context
  }
  return merged;
}

export async function clearConfig(): Promise<void> {
  try {
    await chrome.storage.local.remove(STORAGE_KEY);
  } catch {
    // Outside extension context
  }
}
