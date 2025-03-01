"""
Market Analyzer Agent for NEAR DeFAI Asset Guardian
Analyzes market conditions and identifies opportunities/threats
"""

from typing import Dict, Any, Optional
from near_swarm.plugins.base import AgentPlugin
from near_swarm.core.llm_provider import create_llm_provider, LLMConfig
from near_swarm.core.market_data import MarketDataManager
import logging

logger = logging.getLogger(__name__)

class MarketAnalyzerPlugin(AgentPlugin):
    """Market analysis agent for DeFAI Asset Guardian"""
    
    async def initialize(self) -> None:
        """Initialize plugin resources"""
        # Initialize LLM provider for advanced market analysis
        llm_config = LLMConfig(
            provider=self.agent_config.llm.provider,
            api_key=self.agent_config.llm.api_key,
            model=self.agent_config.llm.model,
            temperature=0.5,  # Lower temperature for more precise analysis
            max_tokens=1500,
            api_url=self.agent_config.llm.api_url,
            system_prompt="""You are a sophisticated market analysis agent specializing in NEAR ecosystem and DeFi.
            Your role is to analyze market conditions, identify trends, and detect patterns that indicate opportunities or threats.
            
            Focus on:
            1. Price trends and correlations
            2. Market sentiment analysis
            3. Protocol-specific metrics (TVL, volume, unique users)
            4. Cross-chain patterns relevant to NEAR
            
            Always provide quantitative reasoning and confidence levels in your analysis.
            """
        )
        self.llm_provider = create_llm_provider(llm_config)
        
        # Initialize market data manager for real-time data
        self.market_data = MarketDataManager()
        
        logger.info("Market Analyzer initialized successfully")
    
    async def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate market conditions with detailed analysis"""
        if not self.llm_provider:
            raise RuntimeError("Plugin not initialized")
            
        # Get current market data
        near_data = await self.market_data.get_token_price("near")
        market_context = await self.get_market_context()
        
        # Create comprehensive prompt for the LLM
        prompt = f"""Analyze the current NEAR ecosystem market conditions:

Price Data:
- NEAR Price: ${near_data['price']:.2f}
- 24h Change: {near_data.get('price_change_24h', 0):.2f}%
- Market Cap: ${near_data.get('market_cap', 0):,.0f}
- 24h Volume: ${near_data.get('volume_24h', 0):,.0f}

DeFi Metrics:
- Total TVL: ${market_context.get('tvl', 0):,.0f}
- DEX 24h Volume: ${market_context.get('dex_volume', 0):,.0f}
- Protocol Activity: {market_context.get('protocol_activity', 'moderate')}

On-Chain Activity:
- Active Wallets: {market_context.get('active_wallets', 0):,}
- Transaction Volume: {market_context.get('tx_volume', 0):,}

External Factors:
- BTC Correlation: {market_context.get('btc_correlation', 0)}
- Overall Market Trend: {market_context.get('market_trend', 'neutral')}

Provide a comprehensive analysis with:
1. Main market trends and patterns
2. Key risk factors to monitor
3. Potential opportunities (short and medium-term)
4. Overall market sentiment score (0-100)
5. Confidence level in your analysis (0-1)

Format your response as JSON with these fields.
"""
        
        # Get detailed market analysis from LLM
        try:
            analysis = await self.llm_provider.query(prompt, expect_json=True)
            
            # Add raw market data to the analysis
            analysis['raw_data'] = {
                'near_price': near_data['price'],
                'price_change_24h': near_data.get('price_change_24h', 0),
                'market_cap': near_data.get('market_cap', 0),
                'tvl': market_context.get('tvl', 0)
            }
            
            # Add timestamp
            from datetime import datetime
            analysis['timestamp'] = datetime.now().isoformat()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error during market analysis: {e}")
            return {
                "error": str(e),
                "trends": [],
                "risk_factors": ["Unable to complete analysis"],
                "opportunities": [],
                "sentiment_score": 50,
                "confidence": 0
            }
    
    async def get_market_context(self) -> Dict[str, Any]:
        """Get comprehensive market context data"""
        try:
            # Get DeFi protocol data
            defi_data = {
                'tvl': 250000000,  # Example TVL in USD
                'dex_volume': 15000000,  # Example 24h volume in USD
                'protocol_activity': 'high'
            }
            
            # Get on-chain metrics
            onchain_data = {
                'active_wallets': 12500,
                'tx_volume': 85000
            }
            
            # Get correlation with BTC
            btc_data = await self.market_data.get_token_price("btc")
            btc_correlation = 0.75  # Example correlation coefficient
            
            # Overall market trend based on major assets
            market_trend = "bullish"  # Example trend
            
            return {
                **defi_data,
                **onchain_data,
                'btc_correlation': btc_correlation,
                'market_trend': market_trend
            }
        except Exception as e:
            logger.error(f"Error getting market context: {e}")
            return {
                'tvl': 0,
                'dex_volume': 0,
                'protocol_activity': 'unknown',
                'active_wallets': 0,
                'tx_volume': 0,
                'btc_correlation': 0,
                'market_trend': 'neutral'
            }
    
    async def execute(self, operation: Optional[str] = None, **kwargs) -> Any:
        """Execute plugin operation"""
        if operation == "analyze_market":
            return await self.evaluate(kwargs)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources"""
        if self.llm_provider:
            await self.llm_provider.close()
        if hasattr(self, 'market_data'):
            await self.market_data.close()

# Register the plugin
from near_swarm.plugins import register_plugin
register_plugin("market-analyzer", MarketAnalyzerPlugin)