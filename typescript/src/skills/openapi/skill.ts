import { Skill } from '../../core/skill';
import type { Context, Tool } from '../../core/types';

const MAX_SPEC_SIZE = 5 * 1024 * 1024; // 5 MB
const SPEC_CACHE_TTL = 10 * 60 * 1000; // 10 minutes
const MAX_RESPONSE_SIZE = 100 * 1024; // 100 KB

interface OpenAPIServerConfig {
  specUrl: string;
  baseUrl?: string;
  auth?: { type: string; token: string };
  operations?: string[]; // allowlist of operationIds
}

export interface OpenAPISkillConfig {
  servers: Record<string, OpenAPIServerConfig>;
  [key: string]: unknown;
}

interface ParsedOperation {
  operationId: string;
  method: string;
  path: string;
  summary?: string;
  description?: string;
  parameters?: Array<{ name: string; in: string; required?: boolean; schema?: Record<string, unknown> }>;
  requestBody?: Record<string, unknown>;
}

interface CachedSpec {
  operations: ParsedOperation[];
  baseUrl: string;
  cachedAt: number;
}

const specCache = new Map<string, CachedSpec>();

export class OpenAPISkill extends Skill {
  name = 'openapi';
  description = 'Provides tools from OpenAPI specifications';
  enabled = true;

  private config: OpenAPISkillConfig;
  private registeredTools = new Map<string, { serverName: string; operation: ParsedOperation; serverConfig: OpenAPIServerConfig; baseUrl: string }>();

  constructor(config: OpenAPISkillConfig) {
    super();
    this.config = config;
  }

  async initialize(): Promise<void> {
    for (const [serverName, serverConfig] of Object.entries(this.config.servers)) {
      try {
        const operations = await this._fetchAndParseSpec(serverName, serverConfig);
        const allowlist = serverConfig.operations ? new Set(serverConfig.operations) : null;
        
        for (const op of operations.operations) {
          if (allowlist && !allowlist.has(op.operationId)) continue;
          
          const qualifiedName = `${serverName}__${op.operationId}`;
          this.registeredTools.set(qualifiedName, {
            serverName,
            operation: op,
            serverConfig,
            baseUrl: operations.baseUrl,
          });
          this._registerDynamicTool(qualifiedName, op, serverConfig, operations.baseUrl);
        }
      } catch (err) {
        console.error(`[OpenAPISkill] Failed to initialize server "${serverName}":`, err);
      }
    }
  }

  private async _fetchAndParseSpec(serverName: string, config: OpenAPIServerConfig): Promise<CachedSpec> {
    const cached = specCache.get(config.specUrl);
    if (cached && Date.now() - cached.cachedAt < SPEC_CACHE_TTL) {
      return cached;
    }

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30_000);
    
    try {
      const res = await fetch(config.specUrl, {
        signal: controller.signal,
        headers: { Accept: 'application/json, application/yaml' },
      });
      
      if (!res.ok) throw new Error(`Failed to fetch spec: ${res.status}`);
      
      const text = await res.text();
      if (text.length > MAX_SPEC_SIZE) {
        throw new Error(`Spec exceeds ${MAX_SPEC_SIZE / 1024 / 1024}MB limit`);
      }
      
      let spec: Record<string, unknown>;
      try {
        const SwaggerParser = (await import('@apidevtools/swagger-parser')).default;
        spec = (await SwaggerParser.dereference(JSON.parse(text))) as Record<string, unknown>;
      } catch {
        spec = JSON.parse(text);
      }
      const operations = this._discoverOperations(spec);
      const baseUrl = config.baseUrl || (spec.servers as Array<{ url?: string }>)?.[0]?.url || new URL(config.specUrl).origin;
      
      const result: CachedSpec = { operations, baseUrl, cachedAt: Date.now() };
      specCache.set(config.specUrl, result);
      return result;
    } finally {
      clearTimeout(timeout);
    }
  }

  private _discoverOperations(spec: Record<string, unknown>): ParsedOperation[] {
    const operations: ParsedOperation[] = [];
    const paths = spec.paths as Record<string, Record<string, unknown>> | undefined;
    if (!paths) return operations;

    for (const [path, methods] of Object.entries(paths)) {
      for (const [method, def] of Object.entries(methods)) {
        if (['get', 'post', 'put', 'patch', 'delete'].indexOf(method.toLowerCase()) === -1) continue;
        const opDef = def as Record<string, unknown>;
        const operationId = (opDef.operationId as string) ||
          `${method.toLowerCase()}_${path.replace(/[{}]/g, '').replace(/[^a-zA-Z0-9]/g, '_').replace(/_+/g, '_').replace(/^_|_$/g, '')}`;
        
        operations.push({
          operationId,
          method: method.toUpperCase(),
          path,
          summary: opDef.summary as string | undefined,
          description: opDef.description as string | undefined,
          parameters: opDef.parameters as ParsedOperation['parameters'],
          requestBody: opDef.requestBody as Record<string, unknown> | undefined,
        });
      }
    }
    return operations;
  }

  private _registerDynamicTool(
    qualifiedName: string,
    op: ParsedOperation,
    serverConfig: OpenAPIServerConfig,
    baseUrl: string,
  ): void {
    const pathParams = op.parameters?.filter(p => p.in === 'path') ?? [];
    const queryParams = op.parameters?.filter(p => p.in === 'query') ?? [];
    const hasBody = !!op.requestBody;
    
    const properties: Record<string, unknown> = {};
    const required: string[] = [];
    
    for (const p of [...pathParams, ...queryParams]) {
      properties[p.name] = p.schema || { type: 'string' };
      if (p.required) required.push(p.name);
    }
    
    if (hasBody) {
      properties['body'] = { type: 'object', description: 'Request body (JSON)' };
    }

    const description = [
      `${op.method} ${op.path}`,
      op.summary,
      op.description,
    ].filter(Boolean).join(' — ');

    const self = this;
    (this as any)['_tool_' + qualifiedName] = {
      name: qualifiedName,
      description,
      parameters: { type: 'object', properties, required },
      handler: async (args: Record<string, unknown>) => {
        return JSON.stringify(await self._executeOperation(qualifiedName, args));
      },
    };
  }

  get tools(): Tool[] {
    const dynamicTools: Tool[] = [];
    for (const [name] of this.registeredTools) {
      const toolDef = (this as any)['_tool_' + name];
      if (toolDef) dynamicTools.push(toolDef);
    }

    dynamicTools.push({
      name: 'list_openapi_endpoints',
      description: 'List all available OpenAPI endpoints across connected servers',
      parameters: { type: 'object', properties: {}, required: [] },
      handler: async () => {
        const endpoints: Record<string, unknown[]> = {};
        for (const [name, info] of this.registeredTools) {
          const serverEndpoints = endpoints[info.serverName] || [];
          serverEndpoints.push({
            tool: name,
            method: info.operation.method,
            path: info.operation.path,
            summary: info.operation.summary,
          });
          endpoints[info.serverName] = serverEndpoints;
        }
        return JSON.stringify(endpoints);
      },
    });

    return dynamicTools;
  }

  private async _executeOperation(qualifiedName: string, args: Record<string, unknown>): Promise<unknown> {
    const info = this.registeredTools.get(qualifiedName);
    if (!info) return { error: `Unknown operation: ${qualifiedName}` };

    const { operation, serverConfig, baseUrl } = info;
    
    let url = `${baseUrl}${operation.path}`;
    
    // Substitute path params
    const pathParams = operation.parameters?.filter(p => p.in === 'path') ?? [];
    for (const p of pathParams) {
      const val = args[p.name];
      if (val !== undefined) url = url.replace(`{${p.name}}`, encodeURIComponent(String(val)));
    }
    
    // Add query params
    const queryParams = operation.parameters?.filter(p => p.in === 'query') ?? [];
    const searchParams = new URLSearchParams();
    for (const p of queryParams) {
      const val = args[p.name];
      if (val !== undefined) searchParams.set(p.name, String(val));
    }
    const qs = searchParams.toString();
    if (qs) url += `?${qs}`;

    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (serverConfig.auth?.token) {
      const prefix = serverConfig.auth.type === 'api_key' ? 'Bearer' : 'Bearer';
      headers['Authorization'] = `${prefix} ${serverConfig.auth.token}`;
    }

    const fetchOpts: RequestInit = {
      method: operation.method,
      headers,
      redirect: 'manual',
    };
    
    if (args.body && ['POST', 'PUT', 'PATCH'].includes(operation.method)) {
      fetchOpts.body = JSON.stringify(args.body);
    }

    try {
      let res = await fetch(url, fetchOpts);
      
      // Handle redirects safely — strip auth on cross-origin
      if ([301, 302, 307, 308].includes(res.status)) {
        const location = res.headers.get('location');
        if (location) {
          const redirectUrl = new URL(location, url);
          const originalOrigin = new URL(url).origin;
          const redirectHeaders = { ...headers };
          if (redirectUrl.origin !== originalOrigin) {
            delete redirectHeaders['Authorization'];
          }
          res = await fetch(redirectUrl.toString(), { ...fetchOpts, headers: redirectHeaders, redirect: 'manual' });
        }
      }
      
      const text = await res.text();
      const truncated = text.length > MAX_RESPONSE_SIZE
        ? text.slice(0, MAX_RESPONSE_SIZE) + `\n...(truncated, ${text.length} bytes total)`
        : text;
      
      try {
        return { status: res.status, data: JSON.parse(truncated) };
      } catch {
        return { status: res.status, data: truncated };
      }
    } catch (err) {
      return { error: `Request failed: ${(err as Error).message}` };
    }
  }
}
