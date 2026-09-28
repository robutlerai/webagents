// The small MCP server the agent-file mcp client is tested against
// (python/tests/fixtures/mcp_tool/echo_server.json holds its tools and answers;
// echo_server.py is the same server in Python). Over stdio by default; with
// `--http <port>`, stateless Streamable HTTP at /mcp on 127.0.0.1.
//
// Kept to the SDK's low-level Server so the tool schemas are served exactly as
// the fixture writes them.

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
  readFileSync(path.resolve(HERE, '../../../python/tests/fixtures/mcp_tool/echo_server.json'), 'utf8'),
);

function makeServer() {
  const server = new Server({ name: 'echo', version: '0.0.1' }, { capabilities: { tools: {} } });
  server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: FIXTURE.tools }));
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args = {} } = request.params;
    if (name === 'echo') return { content: [{ type: 'text', text: `echo: ${args.message}` }] };
    if (name === 'add') return { content: [{ type: 'text', text: String(Number(args.a) + Number(args.b)) }] };
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
    process.stderr.write(`echo server on http://127.0.0.1:${port}${FIXTURE.http_path}\n`);
  });
}
