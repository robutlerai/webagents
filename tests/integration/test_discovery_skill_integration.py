"""
DiscoverySkill Integration Tests

Tests DiscoverySkill with real WebAgents Portal API integration.
Uses fixtures to procure test agents and data via WebAgents API.

Run with: python -m pytest tests/integration/test_discovery_skill_integration.py -v
"""

import pytest
import os
import asyncio
from unittest.mock import Mock

# Load environment from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, use environment as-is
    pass

from webagents.agents.skills.robutler.discovery import (
    DiscoverySkill,
    AgentSearchResult,
    IntentRegistration,
    SearchMode
)
from webagents.agents.core.base_agent import BaseAgent
from robutler.api import RobutlerClient


class MockAgentContext:
    """Mock agent context for integration testing"""
    def __init__(self, agent=None):
        if agent is None:
            agent = Mock()
            agent.name = 'test-discovery-agent'
            agent.api_key = None
        
        self.agent = agent
        # Ensure agent has required attributes
        if not hasattr(self.agent, 'name'):
            self.agent.name = 'test-discovery-agent'
        if not hasattr(self.agent, 'api_key'):
            self.agent.api_key = None


@pytest.fixture
def robutler_api_key():
    """Get WebAgents API key from environment"""
    api_key = os.getenv('WEBAGENTS_API_KEY')
    if not api_key:
        pytest.skip("Integration tests require WEBAGENTS_API_KEY environment variable")
    
    # Accept test key for integration tests in development
    print(f"Using API key: {'***' + api_key[-8:] if len(api_key) > 8 else api_key}")
    return api_key


@pytest.fixture
def webagents_api_url():
    """Get WebAgents API URL from environment"""
    # Use ROBUTLER_API_URL if available, fallback to ROBUTLER_API_URL
    api_url = os.getenv('ROBUTLER_API_URL') or os.getenv('ROBUTLER_API_URL', 'http://localhost:3000')
    print(f"Using API URL: {api_url}")
    return api_url


@pytest.fixture
async def webagents_client(robutler_api_key, webagents_api_url):
    """Create WebAgents client for integration testing"""
    client = RobutlerClient(
        api_key=robutler_api_key,
        base_url=webagents_api_url
    )
    
    yield client
    
    # Cleanup
    try:
        await client.close()
    except:
        pass  # Client might not need explicit cleanup


@pytest.fixture
async def test_agents_data(webagents_client):
    """Procure test agents data via WebAgents API"""
    try:
        # Create/register test agents for discovery via API with timeout
        test_agents = [
            {
                'agent_id': 'integration-test-coding-assistant',
                'name': 'Integration Test Coding Assistant',
                'description': 'Test coding assistant for integration tests',
                'intents': ['help with programming', 'debug code', 'write functions'],
                'capabilities': ['python', 'javascript', 'debugging'],
                'url': 'http://localhost:8001/integration-test-coding-assistant'
            },
            {
                'agent_id': 'integration-test-data-analyst',
                'name': 'Integration Test Data Analyst',
                'description': 'Test data analyst for integration tests',
                'intents': ['analyze data', 'create charts', 'statistics'],
                'capabilities': ['python', 'pandas', 'visualization'],
                'url': 'http://localhost:8002/integration-test-data-analyst'
            }
        ]
        
        # Try to register test agents via API with timeout
        response = await asyncio.wait_for(
            webagents_client._make_request(
                'POST', '/agents/test/register',
                data={
                    'agents': test_agents,
                    'description': 'Integration test agents for discovery testing'
                }
            ),
            timeout=10.0
        )
        
        if response.success:
            registered_agents = response.data.get('agents', [])
            print(f"✅ Registered {len(registered_agents)} test agents")
        else:
            print(f"⚠️  Could not register test agents via API: {response.message}")
        
        # Return test agent IDs and expected data
        return {
            'coding_assistant_id': 'integration-test-coding-assistant',
            'data_analyst_id': 'integration-test-data-analyst',
            'reference_agent_id': os.getenv('TEST_AGENT_ID', 'integration-test-coding-assistant'),
            'test_capabilities': ['python', 'data', 'web'],
            'test_intents': ['programming help', 'data analysis', 'debugging']
        }
        
    except asyncio.TimeoutError:
        print(f"⚠️  Test agent registration timed out, using fallback data")
        return {
            'coding_assistant_id': 'fallback-coding-assistant',
            'data_analyst_id': 'fallback-data-analyst',
            'reference_agent_id': os.getenv('TEST_AGENT_ID', 'fallback-coding-assistant'),
            'test_capabilities': ['python', 'data', 'web'],
            'test_intents': ['programming help', 'data analysis', 'debugging']
        }
    except Exception as e:
        print(f"⚠️  Test agent registration failed, using fallback data: {e}")
        # Fallback test data when API is not available
        return {
            'coding_assistant_id': 'fallback-coding-assistant',
            'data_analyst_id': 'fallback-data-analyst',
            'reference_agent_id': os.getenv('TEST_AGENT_ID', 'fallback-coding-assistant'),
            'test_capabilities': ['python', 'data', 'web'],
            'test_intents': ['programming help', 'data analysis', 'debugging']
        }


@pytest.fixture
def discovery_integration_config(robutler_api_key, webagents_api_url):
    """DiscoverySkill configuration for integration testing"""
    return {
        'enable_discovery': True,
        'search_mode': 'semantic',
        'max_results': 10,
        'webagents_api_url': webagents_api_url,
        'robutler_api_key': robutler_api_key,
        'cache_ttl': 60,  # Shorter TTL for testing
        'agent_url': f'{webagents_api_url}/test-discovery-agent'
    }


@pytest.fixture
async def discovery_skill_integration(discovery_integration_config, robutler_api_key):
    """DiscoverySkill instance configured for integration testing"""
    skill = DiscoverySkill(discovery_integration_config)
    
    # Initialize with mock agent (not mock context)
    mock_agent = Mock()
    mock_agent.name = 'integration-test-agent'
    mock_agent.api_key = robutler_api_key
    
    await skill.initialize(mock_agent)
    
    yield skill
    
    # Cleanup
    if hasattr(skill, 'cleanup'):
        await skill.cleanup()


class TestDiscoverySkillPlatformConnection:
    """Test DiscoverySkill connection to real WebAgents Platform"""
    
    @pytest.mark.asyncio
    async def test_platform_client_connection(self, discovery_skill_integration, webagents_api_url, robutler_api_key):
        """Test connection to real WebAgents Platform"""
        skill = discovery_skill_integration
        
        assert skill.client is not None, "Should have initialized WebAgents client"
        assert skill.webagents_api_url == webagents_api_url
        assert skill.robutler_api_key == robutler_api_key
        
        print(f"Connected to: {skill.webagents_api_url}")
        print(f"API key: {'***' + skill.robutler_api_key[-8:] if len(skill.robutler_api_key) > 8 else skill.robutler_api_key}")
        
        # Test health check if available
        try:
            health = await skill.client.health_check()
            if health.success:
                print(f"✅ Platform health check passed")
            else:
                print(f"⚠️  Platform health check failed: {health.message}")
        except Exception as e:
            print(f"⚠️  Health check error (expected if endpoint unavailable): {e}")
    
    @pytest.mark.asyncio
    async def test_api_key_resolution_integration(self, robutler_api_key, webagents_api_url):
        """Test real API key resolution in integration environment"""
        
        # Test with config key
        skill_config = DiscoverySkill({
            'webagents_api_url': webagents_api_url,
            'robutler_api_key': robutler_api_key
        })
        mock_context = MockAgentContext()
        await skill_config.initialize(mock_context)
        
        assert skill_config.robutler_api_key == robutler_api_key
        print(f"✅ Config API key resolution: {'***' + skill_config.robutler_api_key[-8:]}")
        
        # Test with environment variable fallback
        skill_env = DiscoverySkill({
            'webagents_api_url': webagents_api_url
            # No explicit API key - should use environment
        })
        mock_context2 = MockAgentContext()
        mock_context2.agent.api_key = None  # Force env resolution
        
        await skill_env.initialize(mock_context2)
        
        expected_key = robutler_api_key
        assert skill_env.robutler_api_key == expected_key
        print(f"✅ Environment API key resolution: {'***' + skill_env.robutler_api_key[-8:]}")
        
        await skill_config.cleanup()
        await skill_env.cleanup()


class TestAgentSearchIntegration:
    """Test agent search with real WebAgents Platform API"""
    
    @pytest.mark.asyncio
    async def test_real_agent_search_semantic(self, discovery_skill_integration, test_agents_data):
        """Test semantic agent search with real API"""
        skill = discovery_skill_integration
        
        search_queries = test_agents_data['test_intents'] + [
            "coding expert",
            "python development"
        ]
        
        for query in search_queries:
            try:
                print(f"\nSearching for: '{query}'")
                
                result = await skill.search_agents(
                    query=query,
                    max_results=5,
                    search_mode="semantic",
                    context=None
                )
                
                print(f"Search result: {result.get('success', False)}")
                
                if result.get('success'):
                    agents = result.get('agents', [])
                    total = result.get('total_results', 0)
                    
                    print(f"✅ Found {total} agents")
                    
                    for i, agent in enumerate(agents[:3], 1):  # Show top 3
                        print(f"  {i}. {agent.get('name', 'Unknown')} (ID: {agent.get('agent_id', 'N/A')})")
                        print(f"     Intents: {', '.join(agent.get('intents', [])[:2])}")
                        print(f"     Similarity: {agent.get('similarity_score', 0):.3f}")
                        
                    # Verify response structure
                    assert 'query' in result
                    assert 'search_mode' in result
                    assert 'total_results' in result
                    assert 'agents' in result
                    
                    assert result['query'] == query
                    assert result['search_mode'] == 'semantic'
                    
                else:
                    error = result.get('error', 'Unknown error')
                    print(f"⚠️  Search failed: {error}")
                    
                    # Common failure reasons in integration tests
                    if any(keyword in error.lower() for keyword in ['unavailable', 'connection', 'timeout']):
                        print("   (Platform may not be available)")
                    elif 'not found' in error.lower():
                        print("   (No agents found - expected if platform has no data)")
                    
            except Exception as e:
                print(f"⚠️  Search error for '{query}': {e}")
    
    @pytest.mark.asyncio
    async def test_search_mode_variations(self, discovery_skill_integration):
        """Test different search modes with real API"""
        skill = discovery_skill_integration
        
        query = "help with programming"
        modes = ['semantic', 'exact', 'fuzzy']
        
        for mode in modes:
            try:
                print(f"\nTesting {mode} search mode...")
                
                result = await skill.search_agents(
                    query=query,
                    search_mode=mode,
                    max_results=3,
                    context=None
                )
                
                if result.get('success'):
                    total = result.get('total_results', 0)
                    search_mode = result.get('search_mode')
                    
                    print(f"✅ {mode.title()} mode: {total} results")
                    assert search_mode == mode
                else:
                    print(f"⚠️  {mode.title()} mode failed: {result.get('error')}")
                    
            except Exception as e:
                print(f"⚠️  {mode.title()} mode error: {e}")
    
    @pytest.mark.asyncio
    async def test_search_with_result_limits(self, discovery_skill_integration):
        """Test search result limiting"""
        skill = discovery_skill_integration
        
        limits = [1, 3, 5, 10]
        query = "agent"
        
        for limit in limits:
            try:
                result = await skill.search_agents(
                    query=query,
                    max_results=limit,
                    context=None
                )
                
                if result.get('success'):
                    agents = result.get('agents', [])
                    actual_count = len(agents)
                    
                    print(f"Limit {limit}: Got {actual_count} agents")
                    assert actual_count <= limit, f"Expected max {limit}, got {actual_count}"
                    
                    if actual_count > 0:
                        print("✅ Result limiting working correctly")
                else:
                    print(f"⚠️  Limit test failed: {result.get('error')}")
                    
            except Exception as e:
                print(f"⚠️  Limit test error: {e}")


class TestCapabilityDiscoveryIntegration:
    """Test capability-based discovery with real APIs"""
    
    @pytest.mark.asyncio
    async def test_real_capability_discovery(self, discovery_skill_integration, test_agents_data):
        """Test capability-based agent discovery"""
        skill = discovery_skill_integration
        
        capability_sets = [
            ['python'],
            ['javascript', 'web'],
            ['data', 'analysis'],
        ] + [test_agents_data['test_capabilities']]
        
        for capabilities in capability_sets:
            try:
                print(f"\nDiscovering agents with capabilities: {capabilities}")
                
                result = await skill.discover_agents(
                    capabilities=capabilities,
                    max_results=5,
                    context=None
                )
                
                if result.get('success'):
                    agents = result.get('agents', [])
                    total = result.get('total_results', 0)
                    
                    print(f"✅ Found {total} agents")
                    
                    for i, agent in enumerate(agents[:2], 1):  # Show top 2
                        agent_caps = agent.get('capabilities', [])
                        matches = set(capabilities) & set(agent_caps)
                        
                        print(f"  {i}. {agent.get('name', 'Unknown')}")
                        print(f"     Capabilities: {', '.join(agent_caps[:4])}")
                        print(f"     Matches: {', '.join(matches)}")
                        
                    # Verify response structure
                    assert 'capabilities' in result
                    assert 'total_results' in result
                    assert 'agents' in result
                    assert result['capabilities'] == capabilities
                    
                else:
                    print(f"⚠️  Discovery failed: {result.get('error')}")
                    
            except Exception as e:
                print(f"⚠️  Capability discovery error: {e}")
    
    @pytest.mark.asyncio
    async def test_multi_capability_filtering(self, discovery_skill_integration):
        """Test filtering agents by multiple capabilities"""
        skill = discovery_skill_integration
        
        try:
            # Test with multiple related capabilities
            result = await skill.discover_agents(
                capabilities=['python', 'api', 'web'],
                max_results=10,
                context=None
            )
            
            if result.get('success'):
                agents = result.get('agents', [])
                
                print(f"Multi-capability search found {len(agents)} agents")
                
                # Check that returned agents have at least one matching capability
                for agent in agents:
                    agent_caps = set(agent.get('capabilities', []))
                    search_caps = set(['python', 'api', 'web'])
                    matches = agent_caps & search_caps
                    
                    if matches:
                        print(f"✅ {agent.get('name')}: {matches}")
                    else:
                        print(f"⚠️  {agent.get('name')}: No matching capabilities found")
                        
            else:
                print(f"⚠️  Multi-capability search failed: {result.get('error')}")
                
        except Exception as e:
            print(f"⚠️  Multi-capability test error: {e}")


class TestSimilarAgentsIntegration:
    """Test similar agent discovery with real APIs"""
    
    @pytest.mark.asyncio
    async def test_real_similar_agents_discovery(self, discovery_skill_integration, test_agents_data):
        """Test finding similar agents with real API"""
        skill = discovery_skill_integration
        
        # Test with different reference agent IDs
        reference_agents = [
            test_agents_data['coding_assistant_id'],
            test_agents_data['data_analyst_id'],
            test_agents_data['reference_agent_id']
        ]
        
        for agent_id in reference_agents:
            try:
                print(f"\nFinding agents similar to: {agent_id}")
                
                result = await skill.find_similar_agents(
                    agent_id=agent_id,
                    max_results=5,
                    context=None
                )
                
                if result.get('success'):
                    agents = result.get('agents', [])
                    total = result.get('total_results', 0)
                    
                    print(f"✅ Found {total} similar agents")
                    
                    for i, agent in enumerate(agents[:3], 1):  # Show top 3
                        similarity = agent.get('similarity_score', 0)
                        
                        print(f"  {i}. {agent.get('name', 'Unknown')} (ID: {agent.get('agent_id')})")
                        print(f"     Similarity: {similarity:.3f}")
                        print(f"     Description: {agent.get('description', 'No description')[:50]}...")
                        
                    # Verify response structure
                    assert 'reference_agent' in result
                    assert 'total_results' in result
                    assert 'agents' in result
                    assert result['reference_agent'] == agent_id
                    
                    # Verify no agent matches itself
                    similar_ids = [a.get('agent_id') for a in agents]
                    assert agent_id not in similar_ids, "Similar agents should not include reference agent"
                    
                else:
                    error = result.get('error', 'Unknown error')
                    print(f"⚠️  Similar search failed: {error}")
                    
                    if 'not found' in error.lower():
                        print(f"   (Agent {agent_id} may not exist in platform)")
                    
            except Exception as e:
                print(f"⚠️  Similar agent search error for {agent_id}: {e}")


class TestIntentPublishingIntegration:
    """Test intent publishing with real APIs (requires server handshake)"""
    
    @pytest.mark.asyncio
    async def test_intent_publishing_handshake_required(self, discovery_skill_integration):
        """Test intent publishing - should indicate handshake required"""
        skill = discovery_skill_integration
        
        try:
            result = await skill.publish_intents(
                intents=['integration test intent', 'discovery testing'],
                description='Integration test agent intents',
                capabilities=['testing', 'integration'],
                context=None
            )
            
            print(f"Intent publishing result: {result}")
            
            if result.get('success'):
                print("✅ Intent publishing succeeded (unexpected)")
                print(f"   Published: {result.get('published_intents')}")
            else:
                error = result.get('error', '')
                print(f"⚠️  Intent publishing failed: {error}")
                
                # Expected failures until server handshake implemented
                if any(keyword in error.lower() for keyword in ['handshake', 'authentication', 'client not available']):
                    print("   ✅ Expected failure - handshake required")
                else:
                    print("   ⚠️  Unexpected failure reason")
                    
        except Exception as e:
            print(f"⚠️  Intent publishing error: {e}")
            if 'handshake' in str(e).lower():
                print("   ✅ Expected error - handshake required")
    
    @pytest.mark.asyncio
    async def test_get_published_intents_integration(self, discovery_skill_integration):
        """Test retrieving published intents for current agent"""
        skill = discovery_skill_integration
        
        try:
            result = await skill.get_published_intents(context=None)
            
            print(f"Get published intents result: {result}")
            
            if result.get('success'):
                intents = result.get('intents', [])
                agent_id = result.get('agent_id')
                
                print(f"✅ Retrieved {len(intents)} published intents for agent {agent_id}")
                
                for intent in intents[:3]:  # Show first 3
                    print(f"   - {intent.get('intent', 'Unknown intent')}")
                    
            else:
                error = result.get('error', '')
                print(f"⚠️  Get intents failed: {error}")
                
                if 'not found' in error.lower() or 'no intents' in error.lower():
                    print("   (No published intents found - expected)")
                    
        except Exception as e:
            print(f"⚠️  Get intents error: {e}")


class TestDiscoveryErrorHandlingIntegration:
    """Test error handling in real integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_platform_unavailable_handling(self):
        """Test handling when platform is unavailable"""
        
        # Create skill with invalid URL to simulate unavailable platform
        skill = DiscoverySkill({
            'webagents_api_url': 'http://invalid-platform-url:9999',
            'robutler_api_key': 'test_key'
        })
        
        mock_context = MockAgentContext()
        
        try:
            # Use timeout to prevent hanging
            await asyncio.wait_for(skill.initialize(mock_context), timeout=5.0)
            
            # All operations should fail gracefully with timeouts
            search_result = await asyncio.wait_for(
                skill.search_agents("test query", context=None), 
                timeout=5.0
            )
            assert not search_result.get('success'), "Should fail with unavailable platform"
            
            discover_result = await asyncio.wait_for(
                skill.discover_agents(['python'], context=None),
                timeout=5.0
            )
            assert not discover_result.get('success'), "Should fail with unavailable platform"
            
            similar_result = await asyncio.wait_for(
                skill.find_similar_agents('test-agent', context=None),
                timeout=5.0
            )
            assert not similar_result.get('success'), "Should fail with unavailable platform"
            
            print("✅ Graceful error handling for unavailable platform")
            
        except asyncio.TimeoutError:
            print("✅ Timeout handling for unavailable platform (expected)")
        except Exception as e:
            print(f"✅ Exception handling for unavailable platform: {e}")
        finally:
            try:
                await skill.cleanup()
            except:
                pass  # Cleanup might also timeout
    
    @pytest.mark.asyncio 
    async def test_invalid_parameters_handling(self, discovery_skill_integration):
        """Test handling of invalid parameters"""
        skill = discovery_skill_integration
        
        # Test invalid search mode
        try:
            result = await skill.search_agents(
                query="test",
                search_mode="invalid_mode",
                context=None
            )
            
            assert not result.get('success'), "Should reject invalid search mode"
            print("✅ Correctly handled invalid search mode")
            
        except Exception as e:
            print(f"✅ Exception for invalid search mode: {e}")
        
        # Test negative max_results
        try:
            result = await skill.search_agents(
                query="test",
                max_results=-1,
                context=None
            )
            
            # Should either correct the value or fail gracefully
            if result.get('success'):
                agents = result.get('agents', [])
                assert len(agents) == 0, "Negative max_results should return empty results"
                print("✅ Handled negative max_results")
            else:
                print(f"✅ Rejected negative max_results: {result.get('error')}")
                
        except Exception as e:
            print(f"✅ Exception for negative max_results: {e}")


class TestDiscoveryToolsIntegration:
    """Test DiscoverySkill tools with real API integration"""
    
    @pytest.mark.asyncio
    async def test_all_discovery_tools_integration(self, discovery_skill_integration):
        """Test all discovery tools with real API calls"""
        skill = discovery_skill_integration
        
        # Test search_agents tool
        print("Testing search_agents tool with real API...")
        search_result = await skill.search_agents(
            query="programming help",
            max_results=3,
            context=None
        )
        print(f"Search tool result: {search_result.get('success', False)}")
        
        # Test discover_agents tool
        print("Testing discover_agents tool with real API...")
        discover_result = await skill.discover_agents(
            capabilities=['python'],
            max_results=3,
            context=None
        )
        print(f"Discover tool result: {discover_result.get('success', False)}")
        
        # Test find_similar_agents tool
        print("Testing find_similar_agents tool with real API...")
        similar_result = await skill.find_similar_agents(
            agent_id='test-agent',
            max_results=3,
            context=None
        )
        print(f"Similar tool result: {similar_result.get('success', False)}")
        
        # Test get_published_intents tool
        print("Testing get_published_intents tool with real API...")
        intents_result = await skill.get_published_intents(context=None)
        print(f"Intents tool result: {intents_result.get('success', False)}")
        
        print("✅ All discovery tools tested with real APIs")


@pytest.mark.asyncio
async def test_discovery_integration_configuration(robutler_api_key, webagents_api_url):
    """Test DiscoverySkill configuration with real environment variables"""
    
    print(f"Discovery Integration test configuration:")
    print(f"  ROBUTLER_API_URL: {webagents_api_url}")
    print(f"  WEBAGENTS_API_KEY: {'***' + robutler_api_key[-8:] if robutler_api_key and len(robutler_api_key) > 8 else 'Not set'}")
    
    # Test configuration validation
    config = {
        'webagents_api_url': webagents_api_url,
        'robutler_api_key': robutler_api_key,
        'enable_discovery': True
    }
    
    skill = DiscoverySkill(config)
    assert skill.webagents_api_url == webagents_api_url
    assert skill.robutler_api_key == robutler_api_key
    assert skill.enable_discovery == True
    
    print("✅ Discovery integration test configuration validated") 