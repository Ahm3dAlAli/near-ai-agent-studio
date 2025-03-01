"""
Tests for Market Analyzer Agent
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from agents.market_analyzer import MarketAnalyzerPlugin
from near_swarm.plugins.base import PluginConfig
from near_swarm.core.agent import AgentConfig

@pytest.fixture
def mock_config():
    """Create mock agent configuration."""
    agent_config = AgentConfig(
        name="test-agent",
        environment="test",
        log_level="INFO",
        llm={
            "provider": "mock",
            "api_key": "test-key",
            "model": "test-model",
            "temperature": 0.5,
            "max_tokens": 1000,
            "api_url": "https://test-api.com",
            "system_prompt": "You are a test agent."
        },
        near={
            "network": "testnet",
            "account_id": "test.testnet",
            "private_key": "ed25519:test-key",
            "rpc_url": "https://test-rpc.near.org",
            "use_backup_rpc": False
        }
    )
    
    plugin_config = PluginConfig(
        name="market-analyzer",
        role="market_analyzer",
        capabilities=["market_analysis", "trend_detection"],
        system_prompt="You are a test market analyzer.",
        custom_settings={
            "min_confidence_threshold": 0.7,
            "update_interval": 300,
            "alert_threshold": 0.05
        }
    )
    
    return agent_config, plugin_config

@pytest.mark.asyncio
async def test_market_analyzer_initialization(mock_config):
    """Test market analyzer initialization."""
    agent_config, plugin_config = mock_config
    
    with patch('near_swarm.core.llm_provider.create_llm_provider', return_value=AsyncMock()), \
         patch('near_swarm.core.market_data.MarketDataManager', return_value=AsyncMock()):
        
        plugin = MarketAnalyzerPlugin(agent_config, plugin_config)
        await plugin.initialize()
        
        assert plugin.llm_provider is not None
        assert hasattr(plugin, 'market_data')

@pytest.mark.asyncio
async def test_market_analyzer_evaluate(mock_config):
    """Test market analyzer evaluation."""
    agent_config, plugin_config = mock_config
    
    # Mock response from LLM provider
    mock_analysis = {
        "trends": ["Bullish momentum for NEAR", "Increasing DeFi TVL"],
        "sentiment_score": 75,
        "opportunities": ["Yield farming on Ref Finance", "Staking on Burrow"],
        "confidence": 0.85
    }
    
    # Mock Market Data
    mock_market_data = {
        "price": 4.25,
        "price_change_24h": 3.5,
        "volume_24h": 5000000,
        "market_cap": 420000000
    }
    
    with patch('near_swarm.core.llm_provider.create_llm_provider') as mock_llm_provider, \
         patch('near_swarm.core.market_data.MarketDataManager') as mock_market_manager:
        
        # Configure mocks
        mock_llm = AsyncMock()
        mock_llm.query.return_value = mock_analysis
        mock_llm_provider.return_value = mock_llm
        
        mock_market = AsyncMock()
        mock_market.get_token_price.return_value = mock_market_data
        mock_market_manager.return_value = mock_market
        
        # Initialize and test plugin
        plugin = MarketAnalyzerPlugin(agent_config, plugin_config)
        await plugin.initialize()
        
        context = {"timestamp": "2025-03-01T12:00:00Z"}
        result = await plugin.evaluate(context)
        
        # Verify results
        assert result == mock_analysis
        assert mock_llm.query.called
        assert mock_market.get_token_price.called
