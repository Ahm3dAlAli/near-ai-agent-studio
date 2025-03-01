"""
Tests for Risk Manager Agent
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from decimal import Decimal

from agents.risk_manager import RiskManagerPlugin
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
            "temperature": 0.4,
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
        name="risk-manager",
        role="risk_manager",
        capabilities=["risk_assessment", "portfolio_analysis"],
        system_prompt="You are a test risk manager.",
        custom_settings={
            "min_confidence_threshold": 0.8,
            "max_concentration": 0.25,
            "min_diversification": 3,
            "max_leverage": 2.0,
            "max_exposure": 0.75,
            "protocol_risk_threshold": 0.7
        }
    )
    
    return agent_config, plugin_config

@pytest.mark.asyncio
async def test_risk_manager_initialization(mock_config):
    """Test risk manager initialization."""
    agent_config, plugin_config = mock_config
    
    with patch('near_swarm.core.llm_provider.create_llm_provider', return_value=AsyncMock()), \
         patch('near_swarm.core.near_integration.NEARConnection.from_config', return_value=AsyncMock()):
        
        plugin = RiskManagerPlugin(agent_config, plugin_config)
        await plugin.initialize()
        
        assert plugin.llm_provider is not None
        assert plugin.near is not None
        assert plugin.risk_thresholds['max_concentration'] == Decimal('0.25')

@pytest.mark.asyncio
async def test_risk_manager_portfolio_metrics(mock_config):
    """Test portfolio metrics calculation."""
    agent_config, plugin_config = mock_config
    
    with patch('near_swarm.core.llm_provider.create_llm_provider', return_value=AsyncMock()), \
         patch('near_swarm.core.near_integration.NEARConnection.from_config', return_value=AsyncMock()):
        
        plugin = RiskManagerPlugin(agent_config, plugin_config)
        await plugin.initialize()
        
        # Test portfolio with multiple assets
        portfolio = {
            "assets": [
                {"symbol": "NEAR", "value_usd": 5000},
                {"symbol": "USDC", "value_usd": 3000},
                {"symbol": "wETH", "value_usd": 2000}
            ]
        }
        
        metrics = plugin.calculate_portfolio_metrics(portfolio)
        
        assert metrics['total_value'] == 10000
        assert metrics['asset_count'] == 3
        assert metrics['diversification_score'] == 6  # 3 assets * 2 = 6
        assert metrics['concentration_risk'] == 5  # 5000/10000 = 0.5, 0.5*10 = 5
        
        # Test empty portfolio
        empty_portfolio = {"assets": []}
        empty_metrics = plugin.calculate_portfolio_metrics(empty_portfolio)
        
        assert empty_metrics['total_value'] == 0
        assert empty_metrics['asset_count'] == 0
        assert empty_metrics['diversification_score'] == 0
        assert empty_metrics['concentration_risk'] == 10

