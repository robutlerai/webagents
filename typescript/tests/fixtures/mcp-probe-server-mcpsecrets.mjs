// The MCP server the secrets tests start (S-292, 2026-09-26;
// python/tests/fixtures/mcp_tool/probe_server_mcpsecrets.json holds its tools,
// probe_server_mcpsecrets.py is the same server in Python). Over stdio it
// answers what its own environment holds, over Streamable HTTP which
// Authorization header the request carried, so a test can prove what a server
// actually received without any of it being printed. Over stdio by default;
// with `--http <port>`, stateless Streamable HTTP at /mcp on 127.0.0.1.

import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { WebStandardStreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(
  readFileSync(path.resolve(HERE, '../../../python/tests/fixtures/mcp_tool/probe_server_mcpsecrets.json'), 'utf8'),
);

let lastAuthorization = FIXTURE.none;

function makeServer() {
  const server = new Server({ name: 'probe', version: '0.0.1' }, { capabilities: { tools: {} } });
  server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: FIXTURE.tools }));
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args = {} } = request.params;
    if (name === 'env') return { content: [{ type: 'text', text: process.env[String(args.name)] ?? FIXTURE.unset }] };
    if (name === 'authorization') return { content: [{ type: 'text', text: lastAuthorization }] };
    return { content: [{ type: 'text', text: `unknown tool ${name}` }], isError: true };
  });
  return server;
}

const httpAt = process.argv.indexOf('--http');
if (httpAt < 0) {
  const server = makeServer();
  await server.connect(new StdioServerTransport());
} else {
  const port = Number(process.argv[httpAt + 1]);
  const httpServer = createServer(async (req, res) => {
    if (!req.url?.startsWith(FIXTURE.http_path)) {
      res.writeHead(404).end();
      return;
    }
    lastAuthorization = typeof req.headers.authorization === 'string' ? req.headers.authorization : FIXTURE.none;
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const body = Buffer.concat(chunks);
    const headers = new Headers();
    for (const [key, value] of Object.entries(req.headers)) {
      if (typeof value === 'string') headers.set(key, value);
      else if (Array.isArray(value)) headers.set(key, value.join(', '));
    }
    const request = new Request(`http://127.0.0.1:${port}${req.url}`, {
      method: req.method,
      headers,
      body: req.method === 'GET' || req.method === 'HEAD' ? undefined : body,
    });
    const transport = new WebStandardStreamableHTTPServerTransport({ sessionIdGenerator: undefined, enableJsonResponse: true });
    const server = makeServer();
    await server.connect(transport);
    const response = await transport.handleRequest(request);
    res.writeHead(response.status, Object.fromEntries(response.headers.entries()));
    res.end(Buffer.from(await response.arrayBuffer()));
    await server.close();
  });
  httpServer.listen(port, '127.0.0.1', () => {
    process.stderr.write(`probe server on http://127.0.0.1:${port}${FIXTURE.http_path}\n`);
  });
}
