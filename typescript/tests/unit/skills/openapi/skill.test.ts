import { describe, it, expect, vi, beforeEach } from 'vitest';
import { OpenAPISkill } from '../../../../src/skills/openapi/skill';

describe('OpenAPISkill', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('creates an instance with config', () => {
    const skill = new OpenAPISkill({
      servers: {
        test: { specUrl: 'http://example.com/openapi.json' },
      },
    });
    expect(skill.name).toBe('openapi');
    expect(skill.enabled).toBe(true);
  });

  it('has list_openapi_endpoints tool', () => {
    const skill = new OpenAPISkill({
      servers: {
        test: { specUrl: 'http://example.com/openapi.json' },
      },
    });
    const tools = skill.tools;
    const listTool = tools.find(t => t.name === 'list_openapi_endpoints');
    expect(listTool).toBeDefined();
  });

  it('_discoverOperations parses paths correctly', () => {
    const skill = new OpenAPISkill({ servers: {} });
    const operations = (skill as any)._discoverOperations({
      paths: {
        '/users': {
          get: { operationId: 'listUsers', summary: 'List users' },
          post: { operationId: 'createUser', summary: 'Create user' },
        },
        '/users/{id}': {
          get: { summary: 'Get user by ID' },
        },
      },
    });
    expect(operations).toHaveLength(3);
    expect(operations[0].operationId).toBe('listUsers');
    expect(operations[1].operationId).toBe('createUser');
    expect(operations[2].operationId).toBe('get_users_id');
  });

  it('generates operationId fallback when missing', () => {
    const skill = new OpenAPISkill({ servers: {} });
    const operations = (skill as any)._discoverOperations({
      paths: {
        '/api/v1/items/{itemId}/details': {
          get: {},
        },
      },
    });
    expect(operations[0].operationId).toBe('get_api_v1_items_itemId_details');
  });
});
