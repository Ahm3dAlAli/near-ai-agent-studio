"""
DeFAI Asset Guardian - Intelligent Portfolio Management on NEAR
Main service that orchestrates the specialized agents and provides a unified interface
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
import json
import os
from datetime import datetime
from dotenv import load_dotenv

from near_swarm.core.agent import AgentConfig
from near_swarm.core.swarm_agent import SwarmAgent, SwarmConfig
from near_swarm.plugins import PluginLoader
from near_swarm.core.near_integration import NEARConnection, NEARConfig
from near_swarm.core.market_data import MarketDataManager
from near_swarm.core.memory_manager import MemoryManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeFAIAssetGuardian:
    """
    DeFAI Asset Guardian - Intelligent Portfolio Management System
    
    Uses swarm intelligence from specialized agents to provide:
    1. Real-time portfolio monitoring
    2. Risk assessment and mitigation
    3. Automated strategy execution
    4. Market insights and predictions
    """
    
    def __init__(self):
        """Initialize DeFAI Asset Guardian"""
        # Load environment variables
        load_dotenv()
        
        self.config = self._load_config()
        self.agents = {}
        self.near = None
        self.market_data = None
        self.memory = None
        self.plugin_loader = PluginLoader()
        
        # Asset Guardian state
        self.portfolio = {}
        self.market_analysis = {}
        self.risk_assessment = {}
        self.active_strategies = []
        self.transaction_history = []
        
        # System settings
        self.settings = {
            'auto_execution': False,  # Start in manual mode for safety
            'simulation_mode': True,  # Start in simulation mode
            'risk_tolerance': 'medium',  # Default risk tolerance
            'notification_level': 'high',  # Default to high notifications
            'update_interval': 300,  # 5 minutes between updates
            'emergency_stop_loss': True  # Enable emergency stop loss
        }
        
        logger.info("DeFAI Asset Guardian initialized")
    
    def _load_config(self) -> AgentConfig:
        """Load configuration from environment variables"""
        # Validate required environment variables
        required_vars = ['NEAR_ACCOUNT_ID', 'NEAR_PRIVATE_KEY', 'LLM_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return AgentConfig(
            name="defai-asset-guardian",
            environment=os.getenv('ENVIRONMENT', 'development'),
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            llm={
                'provider': os.getenv('LLM_PROVIDER', 'hyperbolic'),
                'api_key': os.getenv('LLM_API_KEY'),
                'model': os.getenv('LLM_MODEL', 'meta-llama/Llama-3.3-70B-Instruct'),
                'temperature': float(os.getenv('LLM_TEMPERATURE', '0.7')),
                'max_tokens': int(os.getenv('LLM_MAX_TOKENS', '2000')),
                'api_url': os.getenv('LLM_API_URL', 'https://api.hyperbolic.xyz/v1')
            },
            near={
                'network': os.getenv('NEAR_NETWORK', 'testnet'),
                'account_id': os.getenv('NEAR_ACCOUNT_ID'),
                'private_key': os.getenv('NEAR_PRIVATE_KEY'),
                'rpc_url': os.getenv('NEAR_RPC_URL', ''),
                'use_backup_rpc': os.getenv('NEAR_USE_BACKUP_RPC', 'true').lower() == 'true'
            }
        )
    
    async def initialize(self) -> None:
        """Initialize the DeFAI Asset Guardian system"""
        try:
            logger.info("Starting DeFAI Asset Guardian initialization")
            
            # Initialize core services
            self.market_data = MarketDataManager()
            self.memory = MemoryManager()
            
            # Initialize NEAR connection
            near_config = NEARConfig(
                network=self.config.near.network,
                account_id=self.config.near.account_id,
                private_key=self.config.near.private_key,
                node_url=self.config.near.rpc_url,
                use_backup=self.config.near.use_backup_rpc
            )
            self.near = await NEARConnection.from_config(near_config)
            
            # Load and initialize specialized agents
            await self._initialize_agents()
            
            # Get initial portfolio data
            await self._update_portfolio()
            
            # Initial analyses
            self.market_analysis = await self._perform_market_analysis()
            self.risk_assessment = await self._assess_portfolio_risk()
            
            logger.info("DeFAI Asset Guardian initialization complete")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            await self.cleanup()
            raise
    
    async def _initialize_agents(self) -> None:
        """Load and initialize all specialized agents"""
        try:
            agent_configs = [
                {
                    'name': 'market-analyzer',
                    'role': 'market_analyzer',
                    'min_confidence': 0.7
                },
                {
                    'name': 'risk-manager',
                    'role': 'risk_manager',
                    'min_confidence': 0.8  # Higher threshold for risk assessment
                },
                {
                    'name': 'strategy-optimizer',
                    'role': 'strategy_optimizer',
                    'min_confidence': 0.75
                }
            ]
            
            # Load and initialize each agent
            for cfg in agent_configs:
                logger.info(f"Loading agent: {cfg['name']}")
                
                # Create SwarmConfig for this agent
                swarm_config = SwarmConfig(
                    role=cfg['role'],
                    min_confidence=cfg['min_confidence'],
                    min_votes=2,
                    timeout=60.0
                )
                
                # Load agent plugin
                plugin = await self.plugin_loader.load_plugin(cfg['name'])
                
                if not plugin:
                    raise ValueError(f"Failed to load agent plugin: {cfg['name']}")
                
                # Create swarm agent
                agent = SwarmAgent(
                    config=self.config,
                    swarm_config=swarm_config
                )
                
                # Initialize agent
                await agent.initialize()
                
                # Store agent
                self.agents[cfg['name']] = agent
                logger.info(f"Agent {cfg['name']} initialized")
            
            # Form swarm network between agents
            market_analyzer = self.agents.get('market-analyzer')
            risk_manager = self.agents.get('risk-manager')
            strategy_optimizer = self.agents.get('strategy-optimizer')
            
            if market_analyzer and risk_manager and strategy_optimizer:
                await market_analyzer.join_swarm([risk_manager, strategy_optimizer])
                logger.info("Swarm network established between agents")
            
        except Exception as e:
            logger.error(f"Agent initialization error: {e}")
            raise
    
    async def update(self) -> Dict[str, Any]:
        """Update all data and analyses"""
        try:
            # Update portfolio
            await self._update_portfolio()
            
            # Update market analysis
            self.market_analysis = await self._perform_market_analysis()
            
            # Update risk assessment
            self.risk_assessment = await self._assess_portfolio_risk()
            
            # Store history in memory
            await self.memory.store(
                category="updates",
                data={
                    "portfolio": self.portfolio,
                    "market_analysis": self.market_analysis,
                    "risk_assessment": self.risk_assessment
                },
                context={
                    "timestamp": datetime.now().isoformat(),
                    "account_id": self.config.near.account_id
                }
            )
            
            # Check for automated actions
            await self._check_automated_actions()
            
            return {
                "portfolio": self.portfolio,
                "market_analysis": self.market_analysis,
                "risk_assessment": self.risk_assessment,
                "updated_at": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Update error: {e}")
            return {
                "error": str(e),
                "updated_at": datetime.now().isoformat()
            }
    
    async def _update_portfolio(self) -> None:
        """Update portfolio data"""
        try:
            # Get account balance
            balance = await self.near.get_account_balance()
            
            # Get token prices
            near_price = (await self.market_data.get_token_price("near"))["price"]
            
            # In a production version, this would fetch all user tokens and positions
            # For the hackathon, we'll use example data plus actual NEAR balance
            
            # Convert NEAR balance to float
            near_balance = float(balance["available"]) / 1e24
            near_value = near_balance * near_price
            
            self.portfolio = {
                "total_value_usd": near_value + 2500,  # Example additional value from other tokens
                "assets": [
                    {
                        "symbol": "NEAR",
                        "balance": near_balance,
                        "value_usd": near_value,
                        "allocation": (near_value / (near_value + 2500)) * 100,
                        "price_usd": near_price,
                        "type": "Token"
                    },
                    {
                        "symbol": "USDC",
                        "balance": 1000,
                        "value_usd": 1000,
                        "allocation": (1000 / (near_value + 2500)) * 100,
                        "price_usd": 1,
                        "type": "Stablecoin"
                    },
                    {
                        "symbol": "wETH",
                        "balance": 0.5,
                        "value_usd": 1000,
                        "allocation": (1000 / (near_value + 2500)) * 100,
                        "price_usd": 2000,
                        "type": "Token"
                    },
                    {
                        "symbol": "REF",
                        "balance": 1000,
                        "value_usd": 500,
                        "allocation": (500 / (near_value + 2500)) * 100,
                        "price_usd": 0.5,
                        "type": "Token"
                    }
                ],
                "updated_at": datetime.now().isoformat()
            }
            
            logger.info(f"Portfolio updated. Total value: ${self.portfolio['total_value_usd']:.2f}")
            
        except Exception as e:
            logger.error(f"Portfolio update error: {e}")
            raise
    
    async def _perform_market_analysis(self) -> Dict[str, Any]:
        """Perform market analysis using the market analyzer agent"""
        try:
            agent = self.agents.get('market-analyzer')
            if not agent:
                raise ValueError("Market analyzer agent not found")
                
            # Create context for analysis
            context = {
                "timestamp": datetime.now().isoformat(),
                "account_id": self.config.near.account_id
            }
            
            # Perform analysis
            response = await agent.evaluate(context)
            
            # Log analysis
            logger.info(f"Market analysis completed. Confidence: {response.get('confidence', 0):.2f}")
            
            return response
            
        except Exception as e:
            logger.error(f"Market analysis error: {e}")
            return {
                "error": str(e),
                "confidence": 0,
                "trends": ["Error performing market analysis"],
                "timestamp": datetime.now().isoformat()
            }
    
    async def _assess_portfolio_risk(self) -> Dict[str, Any]:
        """Assess portfolio risk using the risk manager agent"""
        try:
            agent = self.agents.get('risk-manager')
            if not agent:
                raise ValueError("Risk manager agent not found")
                
            # Create context for risk assessment
            context = {
                "portfolio": self.portfolio,
                "market_analysis": self.market_analysis,
                "timestamp": datetime.now().isoformat(),
                "account_id": self.config.near.account_id
            }
            
            # Perform risk assessment
            response = await agent.evaluate(context)
            
            # Log assessment
            logger.info(f"Risk assessment completed. Risk score: {response.get('risk_score', 0)}")
            
            return response
            
        except Exception as e:
            logger.error(f"Risk assessment error: {e}")
            return {
                "error": str(e),
                "risk_score": 50,  # Default to medium risk
                "risk_factors": ["Error performing risk assessment"],
                "timestamp": datetime.now().isoformat()
            }
    
    async def generate_strategy(self, operation_type: str = "rebalance") -> Dict[str, Any]:
        """Generate a portfolio strategy using the strategy optimizer agent"""
        try:
            agent = self.agents.get('strategy-optimizer')
            if not agent:
                raise ValueError("Strategy optimizer agent not found")
                
            # Create context for strategy generation
            context = {
                "portfolio": self.portfolio,
                "market_analysis": self.market_analysis,
                "risk_assessment": self.risk_assessment,
                "operation_type": operation_type,
                "timestamp": datetime.now().isoformat(),
                "account_id": self.config.near.account_id
            }
            
            # Generate strategy
            response = await agent.evaluate(context)
            
            # Log strategy
            logger.info(f"Strategy generated. Confidence: {response.get('confidence', 0):.2f}")
            
            # Store strategy
            await self.memory.store(
                category="strategies",
                data=response,
                context={
                    "timestamp": datetime.now().isoformat(),
                    "account_id": self.config.near.account_id,
                    "operation_type": operation_type
                }
            )
            
            # Add to active strategies if confidence is high enough
            if response.get('confidence', 0) >= 0.75:
                self.active_strategies.append({
                    "strategy": response,
                    "status": "pending",
                    "created_at": datetime.now().isoformat()
                })
            
            return response
            
        except Exception as e:
            logger.error(f"Strategy generation error: {e}")
            return {
                "error": str(e),
                "confidence": 0,
                "actions": [],
                "reasoning": ["Error generating strategy"],
                "timestamp": datetime.now().isoformat()
            }
    
    async def execute_strategy(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute a generated strategy
        
        Args:
            strategy_id: Optional ID of a specific strategy to execute.
                         If not provided, will use the latest active strategy.
        
        Returns:
            Dict with execution results and status
        """
        try:
            # If strategy_id is provided, find that specific strategy
            # Otherwise, use the latest active strategy
            target_strategy = None
            
            if strategy_id:
                for strategy in self.active_strategies:
                    if strategy.get('strategy', {}).get('id') == strategy_id:
                        target_strategy = strategy
                        break
            else:
                if self.active_strategies:
                    target_strategy = self.active_strategies[-1]
            
            if not target_strategy:
                raise ValueError("No active strategy found for execution")
                
            strategy_data = target_strategy.get('strategy', {})
            
            # Check simulation mode
            if self.settings['simulation_mode']:
                logger.info("Running in simulation mode - no real transactions will be executed")
                
                # Update strategy status
                target_strategy['status'] = 'simulated'
                target_strategy['executed_at'] = datetime.now().isoformat()
                
                return {
                    "status": "simulated",
                    "strategy": strategy_data,
                    "message": "Strategy execution simulated - no real transactions were performed",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Real execution mode
            agent = self.agents.get('strategy-optimizer')
            if not agent:
                raise ValueError("Strategy optimizer agent not found")
                
            # Execute each action in the strategy
            execution_results = []
            for action in strategy_data.get('actions', []):
                # Get best protocol for this action
                protocol = strategy_data.get('execution_details', {}).get('best_protocols', {}).get(action.get('asset'), 'Ref Finance')
                
                # Execute the action
                result = await agent.execute(
                    operation="execute_transaction",
                    action=action,
                    protocol=protocol
                )
                
                execution_results.append({
                    "action": action,
                    "protocol": protocol,
                    "result": result
                })
                
                # Add to transaction history
                self.transaction_history.append({
                    "action": action,
                    "protocol": protocol,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Update strategy status
            target_strategy['status'] = 'executed'
            target_strategy['executed_at'] = datetime.now().isoformat()
            target_strategy['execution_results'] = execution_results
            
            # Store execution in memory
            await self.memory.store(
                category="executions",
                data={
                    "strategy": strategy_data,
                    "execution_results": execution_results
                },
                context={
                    "timestamp": datetime.now().isoformat(),
                    "account_id": self.config.near.account_id
                }
            )
            
            return {
                "status": "executed",
                "execution_results": execution_results,
                "strategy": strategy_data,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Strategy execution error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _check_automated_actions(self) -> None:
        """Check if any automated actions should be triggered"""
        # Skip if auto-execution is disabled
        if not self.settings['auto_execution']:
            return
            
        try:
            # Check for emergency conditions first
            if self.settings['emergency_stop_loss'] and await self._check_emergency_conditions():
                await self._execute_emergency_actions()
                return
                
            # Check if we should generate a new strategy
            should_rebalance = await self._should_rebalance()
            
            if should_rebalance:
                logger.info("Auto-generating rebalance strategy based on current conditions")
                strategy = await self.generate_strategy(operation_type="rebalance")
                
                # Execute immediately if confidence is high enough
                if strategy.get('confidence', 0) >= 0.8:
                    logger.info("Auto-executing high-confidence strategy")
                    await self.execute_strategy()
                
        except Exception as e:
            logger.error(f"Automated actions error: {e}")
    
    async def _check_emergency_conditions(self) -> bool:
        """Check for emergency conditions that require immediate action"""
        # In a production version, this would check for:
        # - Extreme market volatility
        # - Large unexpected portfolio losses
        # - Smart contract vulnerabilities
        # - Suspicious transaction patterns
        
        try:
            # Example check for significant NEAR price drop
            if self.market_analysis.get('price_change_24h', 0) < -15:
                return True
                
            # Example check for high risk score
            if self.risk_assessment.get('risk_score', 0) > 85:
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Emergency condition check error: {e}")
            return False
    
    async def _execute_emergency_actions(self) -> None:
        """Execute emergency actions to protect the portfolio"""
        logger.warning("EMERGENCY ACTIONS TRIGGERED - Executing protective measures")
        
        try:
            # Generate emergency exit strategy
            strategy = await self.generate_strategy(operation_type="exit")
            
            # Execute immediately
            await self.execute_strategy()
            
            # Send notification (in a production version)
            logger.critical("Emergency exit strategy executed")
            
        except Exception as e:
            logger.error(f"Emergency action execution error: {e}")
    
    async def _should_rebalance(self) -> bool:
        """Determine if portfolio should be rebalanced"""
        try:
            # Check time since last rebalance
            last_rebalance = None
            
            for strategy in reversed(self.active_strategies):
                if strategy.get('strategy', {}).get('operation_type') == "rebalance" and strategy.get('status') == "executed":
                    last_rebalance = strategy.get('executed_at')
                    break
            
            if last_rebalance:
                # Parse timestamp and check if enough time has passed
                last_time = datetime.fromisoformat(last_rebalance)
                time_diff = (datetime.now() - last_time).total_seconds()
                
                # Default 24 hours between rebalances
                if time_diff < 86400:
                    return False
            
            # Check for significant market changes
            if abs(self.market_analysis.get('price_change_24h', 0)) > 10:
                return True
                
            # Check for high risk score
            if self.risk_assessment.get('risk_score', 0) > 75:
                return True
                
            # Check for significant asset drift
            for asset in self.portfolio.get('assets', []):
                target_allocation = 25  # Simple equal allocation for example
                current_allocation = asset.get('allocation', 0)
                
                if abs(current_allocation - target_allocation) > 10:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Rebalance check error: {e}")
            return False
    
    async def get_portfolio_insights(self) -> Dict[str, Any]:
        """Get comprehensive insights about the portfolio"""
        try:
            # Combine data from different sources
            insights = {
                "portfolio_summary": {
                    "total_value": self.portfolio.get('total_value_usd', 0),
                    "asset_count": len(self.portfolio.get('assets', [])),
                    "updated_at": self.portfolio.get('updated_at')
                },
                "asset_allocation": [
                    {
                        "symbol": asset.get('symbol'),
                        "allocation": asset.get('allocation'),
                        "value_usd": asset.get('value_usd')
                    }
                    for asset in self.portfolio.get('assets', [])
                ],
                "risk_metrics": {
                    "overall_risk": self.risk_assessment.get('risk_score', 50),
                    "risk_factors": self.risk_assessment.get('risk_factors', []),
                    "diversification": self.risk_assessment.get('raw_metrics', {}).get('diversification_score', 0)
                },
                "market_trends": {
                    "trends": self.market_analysis.get('trends', []),
                    "sentiment": self.market_analysis.get('sentiment_score', 50),
                    "opportunities": self.market_analysis.get('opportunities', [])
                },
                "recent_strategies": self._get_recent_strategies(5),
                "generated_at": datetime.now().isoformat()
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Portfolio insights error: {e}")
            return {
                "error": str(e),
                "generated_at": datetime.now().isoformat()
            }
    
    def _get_recent_strategies(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent strategies with simplified data"""
        recent = []
        
        for strategy in reversed(self.active_strategies):
            if len(recent) >= limit:
                break
                
            recent.append({
                "operation_type": strategy.get('strategy', {}).get('operation_type', 'unknown'),
                "confidence": strategy.get('strategy', {}).get('confidence', 0),
                "status": strategy.get('status'),
                "action_count": len(strategy.get('strategy', {}).get('actions', [])),
                "created_at": strategy.get('created_at'),
                "executed_at": strategy.get('executed_at', None)
            })
            
        return recent
    
    async def get_transaction_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent transaction history"""
        try:
            # In a production version, this would fetch from blockchain
            # For the hackathon, we'll return the stored history
            
            history = list(reversed(self.transaction_history))[:limit]
            
            # Format for better readability
            formatted_history = []
            for tx in history:
                formatted_history.append({
                    "action_type": tx.get('action', {}).get('type'),
                    "asset": tx.get('action', {}).get('asset'),
                    "percentage": tx.get('action', {}).get('percentage'),
                    "protocol": tx.get('protocol'),
                    "status": tx.get('result', {}).get('status'),
                    "tx_hash": tx.get('result', {}).get('transaction_hash'),
                    "timestamp": tx.get('timestamp')
                })
                
            return formatted_history
            
        except Exception as e:
            logger.error(f"Transaction history error: {e}")
            return []
    
    async def update_settings(self, new_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update system settings"""
        try:
            # Validate and apply new settings
            for key, value in new_settings.items():
                if key in self.settings:
                    # Type checking
                    if isinstance(self.settings[key], bool) and not isinstance(value, bool):
                        continue
                        
                    # Specific validation for certain settings
                    if key == 'risk_tolerance' and value not in ['low', 'medium', 'high']:
                        continue
                    
                    # Apply validated setting
                    self.settings[key] = value
                    logger.info(f"Updated setting {key} to {value}")
            
            return {
                "status": "success",
                "settings": self.settings,
                "updated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Settings update error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "updated_at": datetime.now().isoformat()
            }
    
    async def cleanup(self) -> None:
        """Clean up all resources"""
        try:
            # Clean up agents
            for name, agent in self.agents.items():
                try:
                    await agent.cleanup()
                    logger.info(f"Agent {name} cleaned up")
                except Exception as e:
                    logger.error(f"Error cleaning up agent {name}: {e}")
            
            # Clean up NEAR connection
            if self.near:
                await self.near.close()
                logger.info("NEAR connection closed")
            
            # Clean up market data
            if self.market_data:
                await self.market_data.close()
                logger.info("Market data connection closed")
            
            logger.info("DeFAI Asset Guardian shutdown complete")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
        return None