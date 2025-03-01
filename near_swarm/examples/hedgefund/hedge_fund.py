"""
NearAI Hedge Fund - Core Implementation

This module implements the core functionality of the NearAI Hedge Fund,
integrating AI agents with NEAR blockchain for autonomous investment.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from near_swarm.core.agent import AgentConfig
from near_swarm.core.swarm_agent import SwarmAgent, SwarmConfig
from near_swarm.core.market_data import MarketDataManager
from near_swarm.core.consensus import ConsensusManager, Vote
from near_swarm.core.near_integration import NEARConnection, NEARConfig
from near_swarm.core.memory_manager import MemoryManager, StrategyOutcome

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NearAIHedgeFund:
    """
    NearAI Hedge Fund core class that integrates AI swarm intelligence
    with NEAR blockchain for autonomous investment strategies.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the hedge fund with configuration."""
        # Load environment variables
        load_dotenv(config_path)
        
        # Initialize managers
        self.market_data = None
        self.memory = None
        self.near = None
        
        # Initialize agents
        self.market_analyzer = None
        self.risk_manager = None
        self.strategy_optimizer = None
        self.portfolio_manager = None
        
        # Initialize consensus manager
        self.consensus = ConsensusManager(
            min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.7")),
            min_votes=int(os.getenv("MIN_VOTES", "2")),
            timeout=float(os.getenv("CONSENSUS_TIMEOUT", "5.0"))
        )
        
        # Track portfolio stats
        self.portfolio_stats = {
            "total_value": 0.0,
            "assets": {},
            "performance": [],
            "current_positions": []
        }
        
        # Track strategies and their performance
        self.active_strategies = []
        
    async def initialize(self) -> None:
        """Initialize all components of the hedge fund."""
        try:
            logger.info("Initializing NearAI Hedge Fund...")
            
            # Create agent configuration
            agent_config = AgentConfig(
                network=os.getenv("NEAR_NETWORK"),
                account_id=os.getenv("NEAR_ACCOUNT_ID"),
                private_key=os.getenv("NEAR_PRIVATE_KEY"),
                llm_provider=os.getenv("LLM_PROVIDER", "hyperbolic"),
                llm_api_key=os.getenv("LLM_API_KEY"),
                llm_model=os.getenv("LLM_MODEL"),
                api_url=os.getenv("LLM_API_URL")
            )
            
            # Initialize NEAR connection
            near_config = NEARConfig(
                network=os.getenv("NEAR_NETWORK"),
                account_id=os.getenv("NEAR_ACCOUNT_ID"),
                private_key=os.getenv("NEAR_PRIVATE_KEY"),
                node_url=os.getenv("NEAR_RPC_URL")
            )
            self.near = await NEARConnection(
                network=near_config.network,
                account_id=near_config.account_id,
                private_key=near_config.private_key,
                node_url=near_config.node_url
            )
            
            # Initialize market data and memory
            self.market_data = MarketDataManager()
            self.memory = MemoryManager()
            
            # Initialize specialized agents
            logger.info("Initializing AI Agent Swarm...")
            
            # Market Analyzer - Analyzes market trends and conditions
            self.market_analyzer = SwarmAgent(
                agent_config,
                SwarmConfig(
                    role="market_analyzer",
                    min_confidence=0.7
                )
            )
            
            # Risk Manager - Manages risk exposure and position sizing
            self.risk_manager = SwarmAgent(
                agent_config,
                SwarmConfig(
                    role="risk_manager",
                    min_confidence=0.8  # Higher threshold for risk decisions
                )
            )
            
            # Strategy Optimizer - Optimizes trading strategies
            self.strategy_optimizer = SwarmAgent(
                agent_config,
                SwarmConfig(
                    role="strategy_optimizer",
                    min_confidence=0.7
                )
            )
            
            # Portfolio Manager - Manages overall portfolio allocation
            self.portfolio_manager = SwarmAgent(
                agent_config,
                SwarmConfig(
                    role="portfolio_manager",
                    min_confidence=0.75
                )
            )
            
            # Form swarm network by connecting agents
            await self.market_analyzer.join_swarm([
                self.risk_manager,
                self.strategy_optimizer,
                self.portfolio_manager
            ])
            
            # Load initial portfolio state from NEAR blockchain
            await self._load_portfolio_state()
            
            logger.info("NearAI Hedge Fund initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing hedge fund: {str(e)}")
            raise
            
    async def _load_portfolio_state(self) -> None:
        """Load current portfolio state from NEAR blockchain."""
        try:
            # Get account balance
            balance = await self.near.get_account_balance()
            self.portfolio_stats["total_value"] = float(balance["available"])
            # TODO: Load token balances and positions from fund smart contract
            # For now, we just use the native NEAR balance
            self.portfolio_stats["assets"] = {
                "NEAR": float(balance["available"])
            }
            
            logger.info(f"Loaded portfolio value: {self.portfolio_stats['total_value']} NEAR")
            
        except Exception as e:
            logger.error(f"Error loading portfolio state: {str(e)}")
            raise
            
    async def analyze_market(self) -> Dict[str, Any]:
        """Analyze current market conditions."""
        try:
            # Get market data
            near_data = await self.market_data.get_token_price("near")
            # Create analysis context
            context = {
                "price": near_data["price"],
                "change_24h": near_data.get("price_change_24h", 0),
                "timestamp": near_data.get("last_updated", ""),
                "request": """Analyze the current NEAR token price and market conditions:
                1. Evaluate recent price movements and identify significant patterns
                2. Assess market sentiment and external factors
                3. Calculate key technical indicators (support/resistance, momentum)
                4. Identify any anomalies or unusual market behavior
                5. Provide a clear recommendation with confidence level
                
                Format your analysis with clear observations, reasoning, and actionable conclusions."""
            }
            
            # Get market analysis from the market analyzer agent
            analysis = await self.market_analyzer.evaluate(context)
            
            # Store analysis in memory
            await self.memory.store(
                category="market_analysis",
                data=analysis,
                context={"price": near_data["price"], "timestamp": near_data.get("last_updated", "")}
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market: {str(e)}")
            raise
            
    async def evaluate_strategy(self, market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate investment strategy based on market analysis."""
        try:
            # Get current price
            near_data = await self.market_data.get_token_price("near")
            current_price = near_data["price"]
            
            # Create strategy context
            context = {
                "market_analysis": market_analysis,
                "current_price": current_price,
                "portfolio": self.portfolio_stats,
                "request": """Based on the market analysis, evaluate potential investment strategies:
                1. Consider current market conditions and risk levels
                2. Evaluate potential entry and exit points
                3. Calculate optimal position sizes based on risk tolerance
                4. Design a comprehensive strategy with specific actions
                5. Provide clear reasoning and confidence levels for your recommendations
                
                Format your strategy with context, recommendations, rationale, and specific actions."""
            }
            
            # Get strategy evaluation from the strategy optimizer agent
            strategy = await self.strategy_optimizer.evaluate(context)
            
            # Store strategy in memory
            await self.memory.store(
                category="strategy_evaluation",
                data=strategy,
                context={"price": current_price, "timestamp": near_data.get("last_updated", "")}
            )
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error evaluating strategy: {str(e)}")
            raise
            
    async def assess_risk(
        self,
        market_analysis: Dict[str, Any],
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess risk of the proposed strategy."""
        try:
            # Get current price
            near_data = await self.market_data.get_token_price("near")
            current_price = near_data["price"]
            
            # Create risk assessment context
            context = {
                "market_analysis": market_analysis,
                "proposed_strategy": strategy,
                "current_price": current_price,
                "portfolio": self.portfolio_stats,
                "request": """Assess the risk of the proposed investment strategy:
                1. Evaluate overall market risk based on conditions and volatility
                2. Calculate specific risk metrics for the proposed strategy
                3. Assess portfolio impact and potential drawdown
                4. Identify key risk factors and potential mitigations
                5. Provide a comprehensive risk assessment with confidence level
                
                Format your assessment with risk level, exposure assessment, position recommendations,
                risk mitigations, and stop-loss recommendations."""
            }
            
            # Get risk assessment from the risk manager agent
            assessment = await self.risk_manager.evaluate(context)
            # Store assessment in memory
            await self.memory.store(
                category="risk_assessment",
                data=assessment,
                context={"price": current_price, "strategy": strategy.get("action", "")}
            )
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing risk: {str(e)}")
            raise
            
    async def optimize_portfolio(
        self,
        market_analysis: Dict[str, Any],
        strategy: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize portfolio allocation based on strategy and risk assessment."""
        try:
            # Get current price
            near_data = await self.market_data.get_token_price("near")
            current_price = near_data["price"]
            
            # Create portfolio optimization context
            context = {
                "market_analysis": market_analysis,
                "proposed_strategy": strategy,
                "risk_assessment": risk_assessment,
                "current_price": current_price,
                "portfolio": self.portfolio_stats,
                "request": """Optimize the portfolio allocation based on the proposed strategy and risk assessment:
                1. Evaluate current portfolio composition and performance
                2. Calculate optimal asset allocation considering the strategy
                3. Determine position sizing based on risk parameters
                4. Create specific rebalancing recommendations
                5. Provide transaction details for execution
                
                Format your optimization with portfolio analysis, allocation adjustments,
                position sizing, and specific transaction details."""
            }
            
            # Get portfolio optimization from the portfolio manager agent
            optimization = await self.portfolio_manager.evaluate(context)
            
            # Store optimization in memory
            await self.memory.store(
                category="portfolio_optimization",
                data=optimization,
                context={"price": current_price, "strategy": strategy.get("action", "")}
            )
            
            return optimization
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {str(e)}")
            raise
            
    async def build_consensus(
        self,
        market_analysis: Dict[str, Any],
        strategy: Dict[str, Any],
        risk_assessment: Dict[str, Any],
        portfolio_optimization: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Build consensus among agents for the proposed strategy."""
        try:
            # Create proposal based on all analyses
            proposal = {
                "type": "investment_decision",
                "params": {
                    "action": strategy.get("action", ""),
                    "confidence": strategy.get("confidence", 0),
                    "market_analysis": {
                        "observation": market_analysis.get("observation", ""),
                        "trend": market_analysis.get("trend", ""),
                        "confidence": market_analysis.get("confidence", 0)
                    },
                    "risk_assessment": {
                        "risk_level": risk_assessment.get("risk_level", ""),
                        "stop_loss": risk_assessment.get("stop_loss", "")
                    },
                    "position_size": portfolio_optimization.get("position_size", 0),
                    "current_price": portfolio_optimization.get("current_price", 0)
                }
            }
            
            # Get consensus through the swarm network
            result = await self.market_analyzer.propose_action(
                action_type=proposal["type"],
                params=proposal["params"]
            )
            
            # Store consensus result
            await self.memory.store(
                category="consensus_decisions",
                data=result,
                context={"action": proposal["params"]["action"], "consensus": result["consensus"]}
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error building consensus: {str(e)}")
            raise
            
    async def execute_transaction(
        self,
        consensus: Dict[str, Any],
        portfolio_optimization: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute transaction based on consensus decision."""
        try:
            # Check if consensus was reached
            if not consensus["consensus"]:
                logger.info("No consensus reached, skipping transaction execution")
                return {"status": "skipped", "reason": "No consensus"}
            
            # Extract transaction details
            action = portfolio_optimization.get("action", "")
            amount = portfolio_optimization.get("amount", 0)
            
            logger.info(f"Executing transaction: {action} {amount} NEAR")
            
            # Execute different actions based on the type
            if action == "buy":
                # Example: Buy NEAR tokens on a DEX (simplified)
                result = await self._execute_buy(amount)
                logger.info(f"Buy transaction result: {result}")
                return result
                
            elif action == "sell":
                # Example: Sell NEAR tokens on a DEX (simplified)
                result = await self._execute_sell(amount)
                logger.info(f"Sell transaction result: {result}")
                return result
                
            elif action == "hold":
                logger.info("Hold position, no transaction needed")
                return {"status": "skipped", "reason": "Hold position"}
                
            else:
                logger.warning(f"Unknown action: {action}")
                return {"status": "error", "reason": f"Unknown action: {action}"}
                
        except Exception as e:
            logger.error(f"Error executing transaction: {str(e)}")
            raise
            
    async def _execute_buy(self, amount: float) -> Dict[str, Any]:
        """Execute a buy order (simplified implementation)."""
        # In a real implementation, this would interact with a DEX on NEAR
        # For this example, we'll simulate by transferring to self
        try:
            result = await self.near.send_transaction(
                receiver_id=self.near.account_id,  # Send to self (simulation)
                amount=amount
            )
            
            # Record transaction
            await self.memory.store(
                category="transactions",
                data={
                    "type": "buy",
                    "amount": amount,
                    "timestamp": result.get("timestamp", ""),
                    "transaction_id": result.get("transaction_id", ""),
                    "status": "success"
                }
            )
            
            return {
                "status": "success",
                "transaction_id": result.get("transaction_id", ""),
                "type": "buy",
                "amount": amount
            }
            
        except Exception as e:
            logger.error(f"Buy transaction failed: {str(e)}")
            await self.memory.store(
                category="transactions",
                data={
                    "type": "buy",
                    "amount": amount,
                    "status": "failed",
                    "error": str(e)
                }
            )
            raise
            
    async def _execute_sell(self, amount: float) -> Dict[str, Any]:
        """Execute a sell order (simplified implementation)."""
        # In a real implementation, this would interact with a DEX on NEAR
        # For this example, we'll simulate by transferring to self
        try:
            result = await self.near.send_transaction(
                receiver_id=self.near.account_id,  # Send to self (simulation)
                amount=amount
            )
            
            # Record transaction
            await self.memory.store(
                category="transactions",
                data={
                    "type": "sell",
                    "amount": amount,
                    "timestamp": result.get("timestamp", ""),
                    "transaction_id": result.get("transaction_id", ""),
                    "status": "success"
                }
            )
            
            return {
                "status": "success",
                "transaction_id": result.get("transaction_id", ""),
                "type": "sell",
                "amount": amount
            }
            
        except Exception as e:
            logger.error(f"Sell transaction failed: {str(e)}")
            await self.memory.store(
                category="transactions",
                data={
                    "type": "sell",
                    "amount": amount,
                    "status": "failed",
                    "error": str(e)
                }
            )
            raise
            
    async def run_investment_cycle(self) -> Dict[str, Any]:
        """Execute a complete investment cycle."""
        try:
            logger.info("Starting investment cycle...")
            
            # Step 1: Analyze market
            logger.info("Analyzing market conditions...")
            market_analysis = await self.analyze_market()
            logger.info(f"Market analysis confidence: {market_analysis.get('confidence', 0):.2%}")
            
            # Step 2: Evaluate strategy
            logger.info("Evaluating investment strategy...")
            strategy = await self.evaluate_strategy(market_analysis)
            logger.info(f"Strategy recommendation: {strategy.get('action', 'unknown')}")
            
            # Step 3: Assess risk
            logger.info("Assessing risk...")
            risk_assessment = await self.assess_risk(market_analysis, strategy)
            logger.info(f"Risk level: {risk_assessment.get('risk_level', 'unknown')}")
            
            # Step 4: Optimize portfolio
            logger.info("Optimizing portfolio...")
            portfolio_optimization = await self.optimize_portfolio(
                market_analysis,
                strategy,
                risk_assessment
            )
            logger.info(f"Portfolio action: {portfolio_optimization.get('action', 'unknown')}")
            
            # Step 5: Build consensus
            logger.info("Building consensus...")
            consensus = await self.build_consensus(
                market_analysis,
                strategy,
                risk_assessment,
                portfolio_optimization
            )
            logger.info(f"Consensus reached: {consensus['consensus']}")
            logger.info(f"Approval rate: {consensus['approval_rate']:.2%}")
            
            # Step 6: Execute transaction if consensus reached
            if consensus["consensus"]:
                logger.info("Consensus reached, executing transaction...")
                transaction_result = await self.execute_transaction(
                    consensus,
                    portfolio_optimization
                )
                logger.info(f"Transaction status: {transaction_result['status']}")
            else:
                logger.info("No consensus reached, skipping transaction")
                transaction_result = {
                    "status": "skipped",
                    "reason": "No consensus"
                }
            
            # Step 7: Record cycle outcome
            cycle_result = {
                "timestamp": "",  # Filled by memory manager
                "market_analysis": market_analysis,
                "strategy": strategy,
                "risk_assessment": risk_assessment,
                "portfolio_optimization": portfolio_optimization,
                "consensus": consensus,
                "transaction": transaction_result
            }
            
            await self.memory.store(
                category="investment_cycles",
                data=cycle_result
            )
            
            logger.info("Investment cycle completed successfully")
            return cycle_result
            
        except Exception as e:
            logger.error(f"Error in investment cycle: {str(e)}")
            raise
            
    async def close(self) -> None:
        """Close all connections and clean up resources."""
        try:
            logger.info("Shutting down NearAI Hedge Fund...")
            
            # Close agent connections
            if self.market_analyzer:
                await self.market_analyzer.close()
            if self.risk_manager:
                await self.risk_manager.close()
            if self.strategy_optimizer:
                await self.strategy_optimizer.close()
            if self.portfolio_manager:
                await self.portfolio_manager.close()
                
            # Close market data connection
            if self.market_data:
                await self.market_data.close()
                
            # Close NEAR connection
            if self.near:
                await self.near.close()
                
            logger.info("NearAI Hedge Fund shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            raise
            
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        return None

async def main():
    """Run the NearAI Hedge Fund example."""
    fund = None
    try:
        # Initialize hedge fund
        fund = NearAIHedgeFund()
        await fund.initialize()
        
        # Run investment cycle
        result = await fund.run_investment_cycle()
        
        print("\n=== NearAI Hedge Fund Investment Cycle Results ===")
        print(f"Market Trend: {result['market_analysis'].get('trend', 'unknown')}")
        print(f"Strategy: {result['strategy'].get('action', 'unknown')}")
        print(f"Risk Level: {result['risk_assessment'].get('risk_level', 'unknown')}")
        print(f"Consensus Reached: {result['consensus']['consensus']}")
        print(f"Approval Rate: {result['consensus']['approval_rate']:.2%}")
        print(f"Transaction Status: {result['transaction']['status']}")
        
        if result['transaction']['status'] == 'success':
            print(f"Transaction ID: {result['transaction']['transaction_id']}")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise
    finally:
        if fund:
            await fund.close()

if __name__ == "__main__":
    asyncio.run(main())