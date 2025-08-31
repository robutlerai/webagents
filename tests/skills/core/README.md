# MCP Skill Tests

Comprehensive test suite for the MCPSkill (Model Context Protocol) implementation.

## Running Tests

### Basic Usage

```bash
# Run all MCP skill tests
pytest tests/skills/core/test_mcp_skill.py -v

# Run specific test class
pytest tests/skills/core/test_mcp_skill.py::TestMCPSkillInitialization -v

# Run specific test
pytest tests/skills/core/test_mcp_skill.py::TestMCPSkillInitialization::test_mcp_skill_init_default_config -v
```

### Test Categories

#### Basic Tests
```bash
# Initialization and configuration tests
pytest tests/skills/core/test_mcp_skill.py -k "test_mcp_skill_init or test_mcp_server_config or test_mcp_transport" -v
```

#### Connection Tests
```bash
# Server registration and connection tests
pytest tests/skills/core/test_mcp_skill.py -k "test_register_mcp_server or test_connect_to" -v
```

#### Tool Tests
```bash
# Dynamic tool discovery and execution tests
pytest tests/skills/core/test_mcp_skill.py -k "test_execute_mcp_tool or test_register_dynamic_tool" -v
```

#### Background Task Tests
```bash
# Monitoring and health check tests
pytest tests/skills/core/test_mcp_skill.py -k "test_monitor_connections or test_refresh_capabilities" -v
```

#### Skill Tool Tests
```bash
# Built-in skill tools tests
pytest tests/skills/core/test_mcp_skill.py -k "test_list_mcp_servers or test_show_mcp_history or test_add_mcp_server" -v
```

#### Error Handling Tests
```bash
# Edge cases and error handling
pytest tests/skills/core/test_mcp_skill.py -k "error or exception or failure" -v
```

### Test Options

#### With Coverage
```bash
pytest tests/skills/core/test_mcp_skill.py --cov=robutler.agents.skills.core.mcp --cov-report=html -v
```

#### Parallel Execution
```bash
pytest tests/skills/core/test_mcp_skill.py -n auto -v
```

#### Only Failed Tests
```bash
pytest tests/skills/core/test_mcp_skill.py --lf -v
```

#### Stop on First Failure
```bash
pytest tests/skills/core/test_mcp_skill.py -x -v
```

#### With Test Duration
```bash
pytest tests/skills/core/test_mcp_skill.py --durations=10 -v
```

## Test Structure

The test suite is organized into these main test classes:

- **TestMCPSkillInitialization** - Skill initialization and configuration
- **TestMCPServerConfig** - Server configuration data structures
- **TestMCPTransport** - Transport enum and protocol handling
- **TestMCPExecution** - Execution tracking data structures
- **TestMCPServerRegistration** - Server registration and management
- **TestMCPServerConnection** - Connection lifecycle and management
- **TestMCPCapabilityDiscovery** - Tool/resource/prompt discovery
- **TestMCPDynamicToolRegistration** - Dynamic tool system
- **TestMCPToolExecution** - Tool execution and result handling
- **TestMCPBackgroundTasks** - Monitoring and maintenance tasks
- **TestMCPSkillTools** - Built-in skill tools (list_mcp_servers, etc.)
- **TestMCPSkillStatistics** - Statistics and metrics collection
- **TestMCPSkillCleanup** - Resource cleanup and lifecycle management
- **TestMCPSkillEdgeCases** - Edge cases and error conditions

## Test Configuration

Tests use configuration from:
- `mcp_test_config.json` - Sample server configurations and test data
- `conftest.py` - Pytest fixtures and utilities

## Fixtures Available

Key fixtures for testing:
- `mock_agent` - Mock BaseAgent for skill testing
- `mock_logger` - Mock logger with assertion helpers
- `mock_mcp_server_config` - Pre-configured server configurations
- `mock_mcp_tool` - Sample MCP tool definitions
- `mock_mcp_session` - Mock MCP session with capabilities
- `sample_mcp_executions` - Sample execution history
- `mcp_skill_with_servers` - Pre-configured skill with mock servers

## Markers

Tests are marked with:
- `@pytest.mark.asyncio` - For async test functions
- `mcp` - MCP-specific tests
- `integration` - Integration tests
- `unit` - Unit tests

Use markers to filter tests:
```bash
# Run only MCP tests
pytest -m mcp -v

# Run only unit tests  
pytest -m unit -v

# Skip integration tests
pytest -m "not integration" -v
```

## Requirements

Tests require:
- pytest
- pytest-asyncio
- pytest-mock
- pytest-cov (for coverage)
- unittest.mock (standard library)

## Notes

- All async tests properly handle event loops
- Mock objects simulate real MCP SDK behavior
- Tests cover both success and failure scenarios
- Fixtures provide reusable test data and utilities 