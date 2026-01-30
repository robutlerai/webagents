/**
 * E2E Test Server
 * 
 * Serves the test HTML page for WebGPU/GPU testing in Playwright.
 */

import { createServer } from 'http';
import { readFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PORT = 3456;

// MIME types
const mimeTypes: Record<string, string> = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.mjs': 'application/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.wasm': 'application/wasm',
};

const server = createServer((req, res) => {
  const url = new URL(req.url || '/', `http://localhost:${PORT}`);
  let filePath = join(__dirname, url.pathname === '/' ? 'tests-index.html' : url.pathname);

  // CORS headers for WebGPU and SharedArrayBuffer
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'credentialless'); // Less strict for CDN imports
  res.setHeader('Cross-Origin-Resource-Policy', 'cross-origin');

  // Check if file exists
  if (!existsSync(filePath)) {
    // Try serving from node_modules for transformers.js
    if (url.pathname.startsWith('/node_modules/')) {
      filePath = join(__dirname, '../../', url.pathname);
    }
    
    if (!existsSync(filePath)) {
      res.writeHead(404);
      res.end('Not found');
      return;
    }
  }

  // Determine content type
  const ext = filePath.substring(filePath.lastIndexOf('.'));
  const contentType = mimeTypes[ext] || 'application/octet-stream';

  try {
    const content = readFileSync(filePath);
    res.writeHead(200, { 'Content-Type': contentType });
    res.end(content);
  } catch (err) {
    res.writeHead(500);
    res.end('Server error');
  }
});

server.on('error', (err: NodeJS.ErrnoException) => {
  if (err.code === 'EADDRINUSE') {
    console.log(`✅ Server already running at http://localhost:${PORT}`);
    process.exit(0);
  } else {
    console.error('Server error:', err);
    process.exit(1);
  }
});

server.listen(PORT, () => {
  console.log(`🚀 E2E test server running at http://localhost:${PORT}`);
});
