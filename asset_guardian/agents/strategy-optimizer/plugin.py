"""
Strategy Optimizer Agent for NEAR DeFAI Asset Guardian
Designs and executes optimal trading strategies based on market data and risk assessment
"""

from typing import Dict, Any, Optional, List, Tuple
from near_swarm.plugins.base import AgentPlugin
from near_swarm.core.llm_provider import create_llm_provider, LLMConfig
from near_swarm.core.near_integration import NEARConnection, NEARConfig
import logging
import json
from decimal import Decimal

logger = logging.getLogger(__name__)

class StrategyOptimizerPlugin(AgentPlugin):
    """Strategy optimization agent for DeFAI Asset Guardian"""
    
    async def initialize(self) -> None:
        """Initialize plugin resources"""
        # Initialize LLM provider for strategy optimization
        llm_config = LLMConfig(
            provider=self.agent_config.llm.provider,
            api_key=self.agent_config.llm.api_key,
            model=self.agent_config.llm.model,
            temperature=0.5,
            max_tokens=2000,
            api_url=self.agent_config.llm.api_url,
            system_prompt="""You are a sophisticated strategy optimization agent for the NEAR ecosystem.
            Your role is to design and execute optimal trading strategies based on market analysis and risk assessment.
            
            Focus on:
            1. Creating risk-adjusted trading strategies
            2. Optimizing entry and exit points
            3. Balancing portfolio allocation
            4. Maximizing returns while respecting risk constraints
            
            Always provide detailed reasoning and quantitative metrics for your strategies.
            Consider gas costs, slippage, and market impact in your recommendations.
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
        
        # Strategy parameters
        self.strategy_params = {
            'min_confidence_threshold': Decimal('0.75'),  # Minimum confidence for execution
            'max_slippage': Decimal('0.01'),  # Maximum allowed slippage (1%)
            'min_profit_threshold': Decimal('0.005'),  # Minimum expected profit (0.5%)
            'max_position_size': Decimal('0.1'),  # Maximum position size (10% of portfolio)
            'rebalance_threshold': Decimal('0.05'),  # Trigger rebalance when allocation drifts by 5%
            'dynamic_gas_multiplier': Decimal('1.2')  # Multiply estimated gas by 1.2x
        }
        
        logger.info("Strategy Optimizer initialized successfully")
    
    async def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate and generate optimal trading strategies"""
        if not self.llm_provider:
            raise RuntimeError("Plugin not initialized")
            
        # Extract context data
        market_analysis = context.get('market_analysis', {})
        risk_assessment = context.get('risk_assessment', {})
        portfolio = context.get('portfolio', {})
        operation_type = context.get('operation_type', 'rebalance')
        
        # Get available protocols and their metrics
        protocols = await self.get_protocol_metrics()
        
        # Get recent trading history
        trading_history = await self.get_trading_history()
        
        # Create comprehensive prompt for the LLM based on operation type
        if operation_type == 'rebalance':
            prompt = self._create_rebalance_prompt(portfolio, market_analysis, risk_assessment, protocols)
        elif operation_type == 'entry':
            prompt = self._create_entry_prompt(portfolio, market_analysis, risk_assessment, protocols)
        elif operation_type == 'exit':
            prompt = self._create_exit_prompt(portfolio, market_analysis, risk_assessment, protocols)
        else:
            prompt = self._create_general_prompt(portfolio, market_analysis, risk_assessment, protocols)
        
        # Get strategy recommendation from LLM
        try:
            strategy = await self.llm_provider.query(prompt, expect_json=True)
            
            # Validate and augment strategy
            strategy = self._validate_strategy(strategy, portfolio)
            
            # Add execution details
            strategy['execution_details'] = self._generate_execution_details(strategy, protocols)
            
            # Add timestamp
            from datetime import datetime
            strategy['timestamp'] = datetime.now().isoformat()
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating strategy: {e}")
            return {
                "error": str(e),
                "actions": [],
                "expected_outcome": "Unable to generate strategy",
                "confidence": 0,
                "reasoning": ["Error occurred during strategy generation"]
            }
    
    def _create_rebalance_prompt(self, portfolio: Dict[str, Any], market_analysis: Dict[str, Any], 
                                 risk_assessment: Dict[str, Any], protocols: List[Dict[str, Any]]) -> str:
        """Create prompt for portfolio rebalancing strategy"""
        return f"""Generate an optimal portfolio rebalancing strategy based on the following:

Portfolio Composition:
{self._format_portfolio(portfolio)}

Market Analysis:
- Market Trends: {market_analysis.get('trends', ['Unknown'])}
- Sentiment Score: {market_analysis.get('sentiment_score', 50)}/100
- Key Opportunities: {market_analysis.get('opportunities', ['Unknown'])}
- Confidence: {market_analysis.get('confidence', 0)}

Risk Assessment:
- Overall Risk Score: {risk_assessment.get('risk_score', 50)}/100
- Key Risk Factors: {risk_assessment.get('risk_factors', ['Unknown'])}
- Max Recommended Position: {risk_assessment.get('max_position_size', 0.05) * 100}% of portfolio

Available Protocols:
{self._format_protocols(protocols)}

Generate a rebalancing strategy with:
1. List of specific actions (buy/sell/swap specific assets with amounts as % of portfolio)
2. Expected outcome (improved portfolio metrics, risk reduction, etc.)
3. Reasoning behind each recommendation
4. Execution sequence (order of operations)
5. Confidence level in the strategy (0-1)

Format your response as JSON with these fields.
"""
    
    def _create_entry_prompt(self, portfolio: Dict[str, Any], market_analysis: Dict[str, Any], 
                            risk_assessment: Dict[str, Any], protocols: List[Dict[str, Any]]) -> str:
        """Create prompt for new position entry strategy"""
        return f"""Generate an optimal entry strategy for new positions based on the following:

Portfolio Composition:
{self._format_portfolio(portfolio)}

Market Analysis:
- Market Trends: {market_analysis.get('trends', ['Unknown'])}
- Sentiment Score: {market_analysis.get('sentiment_score', 50)}/100
- Key Opportunities: {market_analysis.get('opportunities', ['Unknown'])}
- Confidence: {market_analysis.get('confidence', 0)}

Risk Assessment:
- Overall Risk Score: {risk_assessment.get('risk_score', 50)}/100
- Key Risk Factors: {risk_assessment.get('risk_factors', ['Unknown'])}
- Max Recommended Position: {risk_assessment.get('max_position_size', 0.05) * 100}% of portfolio

Available Protocols:
{self._format_protocols(protocols)}

Generate an entry strategy with:
1. List of specific assets to acquire (with allocation as % of portfolio)
2. Entry method (market buy, limit order, DCA, etc.)
3. Expected outcome (target metrics, expected returns)
4. Risk mitigation steps
5. Target protocols to use
6. Confidence level in the strategy (0-1)

Format your response as JSON with these fields.
"""
    
    def _create_exit_prompt(self, portfolio: Dict[str, Any], market_analysis: Dict[str, Any], 
                           risk_assessment: Dict[str, Any], protocols: List[Dict[str, Any]]) -> str:
        """Create prompt for position exit strategy"""
        return f"""Generate an optimal exit strategy for current positions based on the following:

Portfolio Composition:
{self._format_portfolio(portfolio)}

Market Analysis:
- Market Trends: {market_analysis.get('trends', ['Unknown'])}
- Sentiment Score: {market_analysis.get('sentiment_score', 50)}/100
- Key Risk Factors: {market_analysis.get('risk_factors', ['Unknown'])}
- Confidence: {market_analysis.get('confidence', 0)}

Risk Assessment:
- Overall Risk Score: {risk_assessment.get('risk_score', 50)}/100
- Key Risk Factors: {risk_assessment.get('risk_factors', ['Unknown'])}
- Exposure Analysis: {risk_assessment.get('exposure_analysis', 'Unknown')}

Available Protocols:
{self._format_protocols(protocols)}

Generate an exit strategy with:
1. List of specific positions to exit (full or partial with %)
2. Exit method (market sell, limit order, staged exit)
3. Reasoning for each exit recommendation
4. Execution timing and sequence
5. Expected outcome after exits
6. Confidence level in the strategy (0-1)

Format your response as JSON with these fields.
"""
    
    def _create_general_prompt(self, portfolio: Dict[str, Any], market_analysis: Dict[str, Any], 
                              risk_assessment: Dict[str, Any], protocols: List[Dict[str, Any]]) -> str:
        """Create general strategy optimization prompt"""
        return f"""Generate the optimal trading strategy based on the following:

Portfolio Composition:
{self._format_portfolio(portfolio)}

Market Analysis:
- Market Trends: {market_analysis.get('trends', ['Unknown'])}
- Sentiment Score: {market_analysis.get('sentiment_score', 50)}/100
- Key Opportunities: {market_analysis.get('opportunities', ['Unknown'])}
- Confidence: {market_analysis.get('confidence', 0)}

Risk Assessment:
- Overall Risk Score: {risk_assessment.get('risk_score', 50)}/100
- Key Risk Factors: {risk_assessment.get('risk_factors', ['Unknown'])}
- Max Recommended Position: {risk_assessment.get('max_position_size', 0.05) * 100}% of portfolio

Available Protocols:
{self._format_protocols(protocols)}

Generate a comprehensive strategy with:
1. List of specific actions (entries, exits, rebalancing)
2. Expected outcome and target metrics
3. Detailed reasoning for each action
4. Risk mitigation steps
5. Execution sequence and timing
6. Confidence level in the strategy (0-1)

Format your response as JSON with these fields.
"""
    
    def _validate_strategy(self, strategy: Dict[str, Any], portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and augment strategy with additional metrics"""
        if not strategy or not isinstance(strategy, dict):
            return {
                "error": "Invalid strategy format",
                "actions": [],
                "confidence": 0,
                "reasoning": ["Strategy validation failed"]
            }
        
        # Ensure required fields exist
        if 'actions' not in strategy:
            strategy['actions'] = []
        
        if 'confidence' not in strategy:
            strategy['confidence'] = 0.5
        
        # Cap confidence based on min threshold
        min_confidence = float(self.strategy_params['min_confidence_threshold'])
        if strategy['confidence'] < min_confidence:
            strategy['warning'] = f"Strategy confidence ({strategy['confidence']}) below minimum threshold ({min_confidence})"
            
        # Validate actions against portfolio
        valid_actions = []
        for action in strategy.get('actions', []):
            if self._validate_action(action, portfolio):
                valid_actions.append(action)
                
        strategy['actions'] = valid_actions
        strategy['validated'] = True
        
        return strategy
    
    def _validate_action(self, action: Dict[str, Any], portfolio: Dict[str, Any]) -> bool:
        """Validate individual action against portfolio constraints"""
        try:
            # Validate basic structure
            if not action or not isinstance(action, dict):
                return False
            
            if 'type' not in action or 'asset' not in action:
                return False
                
            action_type = action.get('type', '').lower()
            
            # Exit actions need to target existing assets
            if action_type == 'sell' or action_type == 'exit':
                asset = action.get('asset', '')
                portfolio_assets = [a.get('symbol', '') for a in portfolio.get('assets', [])]
                if asset not in portfolio_assets:
                    return False
            
            # Validate amounts
            if 'percentage' in action:
                pct = float(action['percentage'])
                if pct <= 0 or pct > 100:
                    return False
                    
                # For buys, check against max position size
                if action_type == 'buy' and pct > float(self.strategy_params['max_position_size']) * 100:
                    action['warning'] = f"Amount exceeds maximum position size of {float(self.strategy_params['max_position_size'])*100}%"
                    action['percentage'] = float(self.strategy_params['max_position_size']) * 100
            
            return True
            
        except Exception as e:
            logger.error(f"Action validation error: {e}")
            return False
    
    def _generate_execution_details(self, strategy: Dict[str, Any], protocols: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate detailed execution plan for the strategy"""
        execution = {
            "steps": [],
            "estimated_gas": 0,
            "best_protocols": {},
            "execution_sequence": []
        }
        
        try:
            # Find best protocol for each action
            for i, action in enumerate(strategy.get('actions', [])):
                action_type = action.get('type', '').lower()
                asset = action.get('asset', '')
                
                # Find optimal protocol for this action
                best_protocol, protocol_details = self._find_best_protocol(action, protocols)
                
                step = {
                    "order": i + 1,
                    "action": action,
                    "protocol": best_protocol,
                    "estimated_gas": protocol_details.get('estimated_gas', 0),
                    "estimated_slippage": protocol_details.get('estimated_slippage', 0)
                }
                
                execution["steps"].append(step)
                execution["estimated_gas"] += step["estimated_gas"]
                
                # Track best protocols by asset
                if asset not in execution["best_protocols"]:
                    execution["best_protocols"][asset] = best_protocol
                
                # Add to execution sequence
                execution["execution_sequence"].append({
                    "step": i + 1,
                    "description": f"{action_type.capitalize()} {asset} on {best_protocol}"
                })
                
            # Apply gas multiplier for safety
            execution["estimated_gas"] *= float(self.strategy_params['dynamic_gas_multiplier'])
                
            return execution
            
        except Exception as e:
            logger.error(f"Error generating execution details: {e}")
            return {
                "steps": [],
                "estimated_gas": 0,
                "error": str(e)
            }
    
    def _find_best_protocol(self, action: Dict[str, Any], protocols: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """Find the best protocol for a given action based on fees, liquidity, etc."""
        try:
            action_type = action.get('type', '').lower()
            asset = action.get('asset', '')
            
            # Filter protocols that support this asset
            viable_protocols = []
            for protocol in protocols:
                if asset in protocol.get('supported_assets', []):
                    viable_protocols.append(protocol)
            
            if not viable_protocols:
                return "Unknown", {"estimated_gas": 0, "estimated_slippage": 0.01}
                
            # Find protocol with lowest fees for this action type
            if action_type in ['buy', 'sell', 'swap']:
                viable_protocols.sort(key=lambda p: p.get('swap_fee', 1.0))
            elif action_type in ['deposit', 'stake']:
                viable_protocols.sort(key=lambda p: p.get('deposit_fee', 1.0))
            elif action_type in ['withdraw', 'unstake']:
                viable_protocols.sort(key=lambda p: p.get('withdrawal_fee', 1.0))
            
            # Take the best option
            best_protocol = viable_protocols[0]
            
            return best_protocol['name'], {
                "estimated_gas": best_protocol.get(f'{action_type}_gas', 30_000_000_000_000),
                "estimated_slippage": best_protocol.get('typical_slippage', 0.005)
            }
            
        except Exception as e:
            logger.error(f"Error finding best protocol: {e}")
            return "Ref Finance", {"estimated_gas": 30_000_000_000_000, "estimated_slippage": 0.005}
    
    async def get_protocol_metrics(self) -> List[Dict[str, Any]]:
        """Get metrics for major NEAR ecosystem protocols"""
        # In a production version, this would fetch real protocol metrics
        # For the hackathon, we'll use example data
        return [
            {
                'name': 'Ref Finance',
                'type': 'DEX',
                'swap_fee': 0.003,  # 0.3%
                'typical_slippage': 0.002,  # 0.2%
                'buy_gas': 25_000_000_000_000,
                'sell_gas': 25_000_000_000_000,
                'swap_gas': 30_000_000_000_000,
                'supported_assets': ['NEAR', 'USDC', 'USDT', 'wBTC', 'wETH', 'REF'],
                'tvl': 65000000,
                'volume_24h': 3500000
            },
            {
                'name': 'Jumbo Exchange',
                'type': 'DEX',
                'swap_fee': 0.0025,  # 0.25%
                'typical_slippage': 0.003,  # 0.3%
                'buy_gas': 28_000_000_000_000,
                'sell_gas': 28_000_000_000_000,
                'swap_gas': 35_000_000_000_000,
                'supported_assets': ['NEAR', 'USDC', 'USDT', 'wBTC', 'wETH', 'JUMBO'],
                'tvl': 25000000,
                'volume_24h': 2000000
            },
            {
                'name': 'Burrow',
                'type': 'Lending',
                'deposit_fee': 0.0,
                'withdrawal_fee': 0.0,
                'borrow_fee': 0.001,  # 0.1%
                'deposit_gas': 15_000_000_000_000,
                'withdraw_gas': 20_000_000_000_000,
                'supported_assets': ['NEAR', 'USDC', 'USDT', 'wBTC', 'wETH'],
                'tvl': 45000000,
                'total_borrowed': 20000000
            }
        ]
    
    async def get_trading_history(self) -> List[Dict[str, Any]]:
        """Get recent trading history for the account"""
        # In a production version, this would fetch real trading history
        # For the hackathon, we'll use example data
        return [
            {
                'timestamp': '2025-02-20T14:30:00Z',
                'type': 'buy',
                'asset': 'NEAR',
                'amount': 100,
                'price': 3.25,
                'tx_hash': '8FtKbCYrYvPg92SV8Gj9JGpATGKiWYLm4WbBQvA4rCTF'
            },
            {
                'timestamp': '2025-02-15T09:45:00Z',
                'type': 'sell',
                'asset': 'REF',
                'amount': 250,
                'price': 0.45,
                'tx_hash': 'D6ESCyK7Z9Pn92M2oJ1FWgJVXJV8rDsrHMnQZKdJkNFk'
            },
            {
                'timestamp': '2025-02-10T16:20:00Z',
                'type': 'swap',
                'from_asset': 'USDC',
                'to_asset': 'wETH',
                'from_amount': 1000,
                'to_amount': 0.385,
                'tx_hash': 'CEuTw4Ch8pDGcpKhphyLtMeuYaK1APvHtTGQMCU2QUSF'
            }
        ]
    
    def _format_portfolio(self, portfolio: Dict[str, Any]) -> str:
        """Format portfolio data for LLM prompt"""
        if not portfolio or not portfolio.get('assets'):
            return "No portfolio data available"
        
        assets = portfolio.get('assets', [])
        result = ""
        for i, asset in enumerate(assets):
            result += f"- Asset {i+1}: {asset.get('symbol', 'Unknown')}\n"
            result += f"  Value: ${asset.get('value_usd', 0):,.2f}\n"
            result += f"  Allocation: {asset.get('allocation', 0):.2f}%\n"
            result += f"  Type: {asset.get('type', 'Token')}\n"
            
        return result
    
    def _format_protocols(self, protocols: List[Dict[str, Any]]) -> str:
        """Format protocol data for LLM prompt"""
        if not protocols:
            return "No protocol data available"
        
        result = ""
        for protocol in protocols:
            result += f"- {protocol.get('name', 'Unknown Protocol')}:\n"
            result += f"  Type: {protocol.get('type', 'Unknown')}\n"
            result += f"  Fees: {protocol.get('swap_fee', 0) * 100:.2f}%\n"
            result += f"  TVL: ${protocol.get('tvl', 0):,.0f}\n"
            
            assets = protocol.get('supported_assets', [])
            if assets:
                result += f"  Supported Assets: {', '.join(assets)}\n"
            
        return result
    
    async def execute(self, operation: Optional[str] = None, **kwargs) -> Any:
        """Execute plugin operation"""
        if operation == "optimize_strategy":
            return await self.evaluate(kwargs)
        elif operation == "execute_transaction":
            return await self.execute_transaction(kwargs)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    async def execute_transaction(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a transaction on the blockchain based on strategy"""
        try:
            action = params.get('action', {})
            protocol = params.get('protocol', 'Ref Finance')
            
            if not action or 'type' not in action or 'asset' not in action:
                raise ValueError("Invalid action parameters")
                
            action_type = action.get('type', '').lower()
            asset = action.get('asset', '')
            percentage = action.get('percentage', 100)
            
            logger.info(f"Executing {action_type} for {asset} at {percentage}% on {protocol}")
            
            # In a production version, this would execute real transactions
            # For the hackathon, we'll simulate success
            
            # Simulate blockchain delay
            import asyncio
            await asyncio.sleep(1)
            
            return {
                "status": "success",
                "transaction_hash": "FQVGjuTNXZ5PVQwxUNxkHjYzm8asvS43agxBXjTcKWjn",
                "block_height": 76584321,
                "gas_used": 28500000000000,
                "executed_at": self._get_current_time()
            }
            
        except Exception as e:
            logger.error(f"Transaction execution error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": self._get_current_time()
            }
    
    def _get_current_time(self) -> str:
        """Get current time in ISO format"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources"""
        if self.llm_provider:
            await self.llm_provider.close()
        if hasattr(self, 'near'):
            await self.near.close()

# Register the plugin
from near_swarm.plugins import register_plugin
register_plugin("strategy-optimizer", StrategyOptimizerPlugin)