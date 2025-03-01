"""
Risk Manager Agent for NEAR DeFAI Asset Guardian
Assesses portfolio risk, smart contract vulnerabilities, and market exposure
"""

from typing import Dict, Any, Optional, List
from near_swarm.plugins.base import AgentPlugin
from near_swarm.core.llm_provider import create_llm_provider, LLMConfig
from near_swarm.core.near_integration import NEARConnection, NEARConfig
import logging
from decimal import Decimal

logger = logging.getLogger(__name__)

class RiskManagerPlugin(AgentPlugin):
    """Risk management agent for DeFAI Asset Guardian"""
    
    async def initialize(self) -> None:
        """Initialize plugin resources"""
        # Initialize LLM provider for risk assessment
        llm_config = LLMConfig(
            provider=self.agent_config.llm.provider,
            api_key=self.agent_config.llm.api_key,
            model=self.agent_config.llm.model,
            temperature=0.4,  # Lower temperature for conservative risk assessment
            max_tokens=1500,
            api_url=self.agent_config.llm.api_url,
            system_prompt="""You are a sophisticated risk assessment agent for the NEAR ecosystem.
            Your role is to analyze portfolio risk, detect smart contract vulnerabilities, and monitor market exposure.
            
            Focus on:
            1. Portfolio diversification and concentration risk
            2. Smart contract security analysis
            3. Protocol-specific risk factors
            4. Liquidation risk and leverage assessment
            
            Always prioritize capital preservation and provide quantitative risk scores.
            Be conservative in your assessment - it's better to overestimate than underestimate risk.
            """
        )
        self.llm_provider = create_llm_provider(llm_config)
        
        # Initialize NEAR connection
        near_config = NEARConfig(
            network=self.agent_config.near.network,
            account_id=self.agent_config.near.account_id,
            private_key=self.agent_config.near.private_key,
            node_url=self.agent_config.near.rpc_url,
            use_backup=self.agent_config.near.use_backup_rpc
        )
        self.near = await NEARConnection.from_config(near_config)
        
        # Risk thresholds based on configured risk_tolerance (conservative by default)
        self.risk_thresholds = {
            'max_concentration': Decimal('0.25'),  # Max 25% in single asset
            'min_diversification': 3,  # At least 3 different assets
            'max_leverage': Decimal('2.0'),  # Max 2x leverage
            'max_exposure': Decimal('0.75'),  # Max 75% of portfolio exposed
            'protocol_risk_threshold': Decimal('0.7')  # Protocol risk score threshold
        }
        
        logger.info("Risk Manager initialized successfully")
    
    async def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate portfolio risk with detailed analysis"""
        if not self.llm_provider:
            raise RuntimeError("Plugin not initialized")
            
        # Extract portfolio and market context
        portfolio = context.get('portfolio', {})
        market_analysis = context.get('market_analysis', {})
        
        # Get protocol risk data
        protocol_risks = await self.get_protocol_risks()
        
        # Calculate portfolio metrics
        portfolio_metrics = self.calculate_portfolio_metrics(portfolio)
        
        # Create comprehensive prompt for the LLM
        prompt = f"""Evaluate the risk profile of this NEAR ecosystem portfolio:

Portfolio Composition:
{self.format_portfolio(portfolio)}

Portfolio Metrics:
- Diversification Score: {portfolio_metrics.get('diversification_score', 0)}/10
- Concentration Risk: {portfolio_metrics.get('concentration_risk', 0)}/10
- Total Value: ${portfolio_metrics.get('total_value', 0):,.2f}
- Assets Count: {portfolio_metrics.get('asset_count', 0)}

Protocol Risk Factors:
{self.format_protocol_risks(protocol_risks)}

Market Context:
- NEAR Price Volatility: {market_analysis.get('volatility', 'medium')}
- Market Sentiment: {market_analysis.get('sentiment_score', 50)}/100
- Market Trend: {market_analysis.get('market_trend', 'neutral')}

Perform a comprehensive risk assessment with:
1. Overall risk score (0-100, higher = riskier)
2. Key risk factors (list the top 3-5 concerns)
3. Exposure analysis (what market conditions would most impact this portfolio)
4. Recommended risk mitigation steps
5. Maximum recommended position size for new investments (as % of portfolio)
6. Confidence level in your assessment (0-1)

Format your response as JSON with these fields.
"""
        
        # Get risk assessment from LLM
        try:
            assessment = await self.llm_provider.query(prompt, expect_json=True)
            
            # Add raw metrics to the assessment
            assessment['raw_metrics'] = portfolio_metrics
            
            # Add timestamp
            from datetime import datetime
            assessment['timestamp'] = datetime.now().isoformat()
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error during risk assessment: {e}")
            return {
                "error": str(e),
                "risk_score": 50,  # Default to medium risk
                "risk_factors": ["Unable to complete risk assessment"],
                "exposure_analysis": "Undetermined",
                "risk_mitigation": ["Reduce portfolio exposure until assessment is available"],
                "max_position_size": 0.05,  # Conservative 5% max position
                "confidence": 0
            }
    
    def calculate_portfolio_metrics(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate portfolio risk metrics"""
        try:
            if not portfolio or not portfolio.get('assets'):
                return {
                    'diversification_score': 0,
                    'concentration_risk': 10,  # Maximum risk
                    'total_value': 0,
                    'asset_count': 0
                }
            
            assets = portfolio.get('assets', [])
            total_value = sum(asset.get('value_usd', 0) for asset in assets)
            
            if total_value == 0:
                return {
                    'diversification_score': 0,
                    'concentration_risk': 10,
                    'total_value': 0,
                    'asset_count': len(assets)
                }
                
            # Calculate asset concentrations
            concentrations = [asset.get('value_usd', 0) / total_value for asset in assets if asset.get('value_usd', 0) > 0]
            
            # Sort concentrations (highest first)
            concentrations.sort(reverse=True)
            
            # Diversification score
            asset_count = len(concentrations)
            diversification_score = min(10, asset_count * 2)  # 5 assets = max score
            
            # Concentration risk (based on highest concentration)
            max_concentration = concentrations[0] if concentrations else 1.0
            concentration_risk = min(10, int(max_concentration * 10))
            
            return {
                'diversification_score': diversification_score,
                'concentration_risk': concentration_risk,
                'total_value': total_value,
                'asset_count': asset_count,
                'concentrations': concentrations
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {
                'diversification_score': 0,
                'concentration_risk': 10,
                'total_value': 0,
                'asset_count': 0,
                'error': str(e)
            }
    
    async def get_protocol_risks(self) -> List[Dict[str, Any]]:
        """Get risk scores for major NEAR ecosystem protocols"""
        # In a production version, this would fetch real protocol risk data
        # For the hackathon, we'll use example data
        return [
            {
                'name': 'Ref Finance',
                'risk_score': 3.2,  # Scale 1-10
                'tvl': 65000000,
                'audit_status': 'Audited - Dec 2024',
                'risk_factors': ['Complex AMM logic', 'High TVL exposure']
            },
            {
                'name': 'Burrow',
                'risk_score': 4.5,
                'tvl': 45000000,
                'audit_status': 'Audited - Aug 2024',
                'risk_factors': ['Liquidation mechanisms', 'Oracle dependencies']
            },
            {
                'name': 'Jumbo Exchange',
                'risk_score': 3.8,
                'tvl': 25000000,
                'audit_status': 'Audited - Oct 2024',
                'risk_factors': ['New codebase', 'Concentrated liquidity model']
            }
        ]
    
    def format_portfolio(self, portfolio: Dict[str, Any]) -> str:
        """Format portfolio data for LLM prompt"""
        if not portfolio or not portfolio.get('assets'):
            return "No portfolio data available"
        
        assets = portfolio.get('assets', [])
        result = ""
        for i, asset in enumerate(assets):
            result += f"- Asset {i+1}: {asset.get('symbol', 'Unknown')}\n"
            result += f"  Value: ${asset.get('value_usd', 0):,.2f}\n"
            result += f"  Protocol: {asset.get('protocol', 'Unknown')}\n"
            result += f"  Type: {asset.get('type', 'Token')}\n"
            
        return result
    
    def format_protocol_risks(self, protocol_risks: List[Dict[str, Any]]) -> str:
        """Format protocol risk data for LLM prompt"""
        if not protocol_risks:
            return "No protocol risk data available"
        
        result = ""
        for protocol in protocol_risks:
            result += f"- {protocol.get('name', 'Unknown Protocol')}:\n"
            result += f"  Risk Score: {protocol.get('risk_score', 0)}/10\n"
            result += f"  TVL: ${protocol.get('tvl', 0):,.0f}\n"
            result += f"  Audit Status: {protocol.get('audit_status', 'Unknown')}\n"
            
            risk_factors = protocol.get('risk_factors', [])
            if risk_factors:
                result += f"  Risk Factors: {', '.join(risk_factors)}\n"
            
        return result
    
    async def execute(self, operation: Optional[str] = None, **kwargs) -> Any:
        """Execute plugin operation"""
        if operation == "assess_risk":
            return await self.evaluate(kwargs)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources"""
        if self.llm_provider:
            await self.llm_provider.close()
        if hasattr(self, 'near'):
            await self.near.close()

# Register the plugin
from near_swarm.plugins import register_plugin
register_plugin("risk-manager", RiskManagerPlugin)