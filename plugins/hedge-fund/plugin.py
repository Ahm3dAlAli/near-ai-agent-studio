"""
NearAI Hedge Fund Plugin
AI-powered hedge fund manager for NEAR blockchain
"""

from typing import Dict, Any, Optional, List
import logging
import asyncio
import os
from datetime import datetime

from near_swarm.plugins.base import AgentPlugin
from near_swarm.core.market_data import MarketDataManager
from near_swarm.core.consensus import ConsensusManager, Vote
from near_swarm.core.memory_manager import MemoryManager, StrategyOutcome
from near_swarm.core.near_integration import NEARConnection, NEARConfig
from near_swarm.core.llm_provider import create_llm_provider, LLMConfig

logger = logging.getLogger(__name__)

class HedgeFundPlugin(AgentPlugin):
    """
    Hedge Fund Manager Agent for NEAR blockchain
    
    This plugin implements an AI-powered hedge fund that uses swarm intelligence
    to make investment decisions on the NEAR blockchain.
    """
    
    async def initialize(self) -> None:
        """Initialize hedge fund resources."""
        try:
            logger.info("Initializing Hedge Fund Manager Agent...")
            
            # Initialize LLM provider
            llm_config = LLMConfig(
                provider=self.agent_config.llm.provider,
                api_key=self.agent_config.llm.api_key,
                model=self.agent_config.llm.model,
                temperature=self.agent_config.llm.temperature,
                max_tokens=self.agent_config.llm.max_tokens,
                api_url=self.agent_config.llm.api_url,
                system_prompt=self.plugin_config.system_prompt
            )
            self.llm_provider = create_llm_provider(llm_config)
            
            # Initialize NEAR connection
            near_config = NEARConfig(
                network=os.getenv("NEAR_NETWORK", "testnet"),
                account_id=os.getenv("NEAR_ACCOUNT_ID"),
                private_key=os.getenv("NEAR_PRIVATE_KEY"),
                node_url=os.getenv("NEAR_RPC_URL", "https://rpc.testnet.fastnear.com")
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
            
            # Initialize consensus manager
            self.consensus = ConsensusManager(
                min_confidence=float(self.plugin_config.settings.get('min_confidence_threshold', 0.7)),
                min_votes=int(self.plugin_config.settings.get('min_votes', 2)),
                timeout=float(self.plugin_config.settings.get('decision_interval', 300)) / 60
            )
            
            # Track portfolio stats
            self.portfolio_stats = {
                "total_value": 0.0,
                "assets": {},
                "performance": [],
                "current_positions": []
            }
            
            # Track active strategies
            self.active_strategies = []
            
            # Load initial portfolio state
            await self._load_portfolio_state()
            
            logger.info("Hedge Fund Manager Agent initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing hedge fund agent: {str(e)}")
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
    
    async def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate market conditions and make investment decisions.
        
        The evaluate method is the main entry point for the agent plugin.
        It processes the context and executes the requested operation.
        """
        try:
            # Check if a specific operation is requested
            operation = context.get("operation", "analyze")
            
            if operation == "analyze":
                # Analyze market conditions
                return await self._analyze_market(context)
                
            elif operation == "strategy":
                # Evaluate investment strategy
                return await self._evaluate_strategy(context)
                
            elif operation == "risk":
                # Assess risk
                return await self._assess_risk(context)
                
            elif operation == "portfolio":
                # Optimize portfolio
                return await self._optimize_portfolio(context)
                
            elif operation == "execute":
                # Execute a complete investment cycle
                return await self._run_investment_cycle()
                
            else:
                logger.warning(f"Unknown operation: {operation}")
                return {
                    "error": f"Unknown operation: {operation}",
                    "available_operations": ["analyze", "strategy", "risk", "portfolio", "execute"]
                }
                
        except Exception as e:
            logger.error(f"Error in evaluate: {str(e)}")
            return {"error": str(e)}
    
    async def _analyze_market(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current market conditions."""
        try:
            # Get market data
            near_data = await self.market_data.get_token_price("near")
            
            # Create analysis prompt
            prompt = f"""Analyze the current NEAR token price and market conditions:

Current Price: ${near_data["price"]:.2f}
24h Change: {near_data.get("price_change_24h", 0):.2f}%
Last Updated: {near_data.get("last_updated", "")}

1. Evaluate recent price movements and identify significant patterns
2. Assess market sentiment and external factors
3. Calculate key technical indicators (support/resistance, momentum)
4. Identify any anomalies or unusual market behavior
5. Provide a clear recommendation with confidence level

Format your analysis with clear observations, reasoning, and actionable conclusions.
Respond in JSON format with the following fields:
- observation: Your detailed observations
- reasoning: Your analysis process
- conclusion: Your actionable conclusion
- trend: The market trend (bullish/bearish/neutral)
- confidence: Your confidence level (0-1)
"""
            
            # Get analysis from LLM
            analysis = await self.llm_provider.query(prompt)
            
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
    
    async def _evaluate_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate investment strategy based on market analysis."""
        try:
            # Get market analysis from context or perform analysis
            market_analysis = context.get("market_analysis")
            if not market_analysis:
                market_analysis = await self._analyze_market(context)
            
            # Get current price
            near_data = await self.market_data.get_token_price("near")
            current_price = near_data["price"]
            
            # Create strategy prompt
            prompt = f"""Based on the market analysis, evaluate potential investment strategies:

Market Analysis:
{market_analysis}

Current Price: ${current_price:.2f}
Portfolio: {self.portfolio_stats}

1. Consider current market conditions and risk levels
2. Evaluate potential entry and exit points
3. Calculate optimal position sizes based on risk tolerance
4. Design a comprehensive strategy with specific actions
5. Provide clear reasoning and confidence levels for your recommendations

Format your strategy with context, recommendations, rationale, and specific actions.
Respond in JSON format with the following fields:
- context: Your understanding of the situation
- strategy: Your recommended approach
- rationale: Detailed explanation of your decision
- action: Specific steps to take (buy/sell/hold)
- amount: Recommended position size
- confidence: Your confidence level (0-1)
"""
            
            # Get strategy from LLM
            strategy = await self.llm_provider.query(prompt)
            
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
    
    async def _assess_risk(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk of the proposed strategy."""
        try:
            # Get market analysis and strategy from context or perform evaluation
            market_analysis = context.get("market_analysis")
            if not market_analysis:
                market_analysis = await self._analyze_market(context)
                
            strategy = context.get("strategy")
            if not strategy:
                strategy = await self._evaluate_strategy({
                    "market_analysis": market_analysis
                })
            
            # Get current price
            near_data = await self.market_data.get_token_price("near")
            current_price = near_data["price"]
            
            # Create risk assessment prompt
            prompt = f"""Assess the risk of the proposed investment strategy:

Market Analysis:
{market_analysis}

Proposed Strategy:
{strategy}

Current Price: ${current_price:.2f}
Portfolio: {self.portfolio_stats}

1. Evaluate overall market risk based on conditions and volatility
2. Calculate specific risk metrics for the proposed strategy
3. Assess portfolio impact and potential drawdown
4. Identify key risk factors and potential mitigations
5. Provide a comprehensive risk assessment with confidence level

Format your assessment with risk level, exposure assessment, position recommendations,
risk mitigations, and stop-loss recommendations.
Respond in JSON format with the following fields:
- risk_level: Overall risk level (low/medium/high)
- exposure_assessment: Your assessment of current and proposed exposure
- position_recommendations: Recommended position size adjustments
- risk_mitigations: Recommended risk mitigation strategies
- stop_loss: Recommended stop-loss level
- confidence: Your confidence level (0-1)
"""
            
            # Get risk assessment from LLM
            assessment = await self.llm_provider.query(prompt)
            
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
    
    async def _optimize_portfolio(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize portfolio allocation based on strategy and risk assessment."""
        try:
            # Get previous analyses from context or perform them
            market_analysis = context.get("market_analysis")
            if not market_analysis:
                market_analysis = await self._analyze_market(context)
                
            strategy = context.get("strategy")
            if not strategy:
                strategy = await self._evaluate_strategy({
                    "market_analysis": market_analysis
                })
                
            risk_assessment = context.get("risk_assessment")
            if not risk_assessment:
                risk_assessment = await self._assess_risk({
                    "market_analysis": market_analysis,
                    "strategy": strategy
                })
            
            # Get current price
            near_data = await self.market_data.get_token_price("near")
            current_price = near_data["price"]
            
            # Create portfolio optimization prompt
            prompt = f"""Optimize the portfolio allocation based on the proposed strategy and risk assessment:

Market Analysis:
{market_analysis}

Proposed Strategy:
{strategy}

Risk Assessment:
{risk_assessment}

Current Price: ${current_price:.2f}
Portfolio: {self.portfolio_stats}

1. Evaluate current portfolio composition and performance
2. Calculate optimal asset allocation considering the strategy
3. Determine position sizing based on risk parameters
4. Create specific rebalancing recommendations
5. Provide transaction details for execution

Format your optimization with portfolio analysis, allocation adjustments,
position sizing, and specific transaction details.
Respond in JSON format with the following fields:
- analysis: Your portfolio analysis
- action: Recommended action (buy/sell/hold)
- amount: Recommended position size
- allocation: Recommended asset allocation
- rebalancing: Specific rebalancing steps
- confidence: Your confidence level (0-1)
"""
            
            # Get portfolio optimization from LLM
            optimization = await self.llm_provider.query(prompt)
            
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
    
    async def _build_consensus(
        self,
        market_analysis: Dict[str, Any],
        strategy: Dict[str, Any],
        risk_assessment: Dict[str, Any],
        portfolio_optimization: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Build consensus for the proposed strategy."""
        try:
            # Create proposal based on all analyses
            proposal = {
                "market_analysis": market_analysis,
                "strategy": strategy,
                "risk_assessment": risk_assessment,
                "portfolio_optimization": portfolio_optimization
            }
            
            # Simulate votes from different agents
            market_analyzer_vote = Vote(
                agent_id="market_analyzer",
                decision=market_analysis.get("confidence", 0) >= 0.7,
                confidence=market_analysis.get("confidence", 0),
                reasoning=market_analysis.get("conclusion", "")
            )
            
            strategy_optimizer_vote = Vote(
                agent_id="strategy_optimizer",
                decision=strategy.get("confidence", 0) >= 0.7,
                confidence=strategy.get("confidence", 0),
                reasoning=strategy.get("rationale", "")
            )
            
            risk_manager_vote = Vote(
                agent_id="risk_manager",
                decision=risk_assessment.get("confidence", 0) >= 0.8,  # Higher threshold for risk
                confidence=risk_assessment.get("confidence", 0),
                reasoning=risk_assessment.get("exposure_assessment", "")
            )
            
            portfolio_manager_vote = Vote(
                agent_id="portfolio_manager",
                decision=portfolio_optimization.get("confidence", 0) >= 0.75,
                confidence=portfolio_optimization.get("confidence", 0),
                reasoning=portfolio_optimization.get("analysis", "")
            )
            
            # Collect votes
            votes = [
                market_analyzer_vote,
                strategy_optimizer_vote, 
                risk_manager_vote,
                portfolio_manager_vote
            ]
            
            # Determine consensus
            consensus = self.consensus.reach_consensus(votes)
            
            # Store consensus result
            await self.memory.store(
                category="consensus_decisions",
                data=consensus,
                context={
                    "action": portfolio_optimization.get("action", ""),
                    "confidence_scores": consensus["confidence_scores"],
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            return consensus
            
        except Exception as e:
            logger.error(f"Error building consensus: {str(e)}")
            raise
    
    async def _execute_transaction(
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
            amount = float(portfolio_optimization.get("amount", 0))
            
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
    
    async def _run_investment_cycle(self) -> Dict[str, Any]:
        """Execute a complete investment cycle."""
        try:
            logger.info("Starting investment cycle...")
            
            # Step 1: Analyze market
            logger.info("Analyzing market conditions...")
            market_analysis = await self._analyze_market({})
            logger.info(f"Market analysis confidence: {market_analysis.get('confidence', 0):.2%}")
            
            # Step 2: Evaluate strategy
            logger.info("Evaluating investment strategy...")
            strategy = await self._evaluate_strategy({"market_analysis": market_analysis})
            logger.info(f"Strategy recommendation: {strategy.get('action', 'unknown')}")
            
            # Step 3: Assess risk
            logger.info("Assessing risk...")
            risk_assessment = await self._assess_risk({
                "market_analysis": market_analysis,
                "strategy": strategy
            })
            logger.info(f"Risk level: {risk_assessment.get('risk_level', 'unknown')}")
            
            # Step 4: Optimize portfolio
            logger.info("Optimizing portfolio...")
            portfolio_optimization = await self._optimize_portfolio({
                "market_analysis": market_analysis,
                "strategy": strategy,
                "risk_assessment": risk_assessment
            })
            logger.info(f"Portfolio action: {portfolio_optimization.get('action', 'unknown')}")
            
            # Step 5: Build consensus
            logger.info("Building consensus...")
            consensus = await self._build_consensus(
                market_analysis,
                strategy,
                risk_assessment,
                portfolio_optimization
            )
            logger.info(f"Consensus reached: {consensus['consensus']}")
            logger.info(f"Approval rate: {consensus['approval_rate']:.2%}")
            
            # Step 6: Execute transaction if consensus reached
            transaction_result = {"status": "skipped", "reason": "No consensus"}
            if consensus["consensus"]:
                logger.info("Consensus reached, executing transaction...")
                transaction_result = await self._execute_transaction(
                    consensus,
                    portfolio_optimization
                )
                logger.info(f"Transaction status: {transaction_result['status']}")
            else:
                logger.info("No consensus reached, skipping transaction")
            
            # Step 7: Record cycle outcome
            cycle_result = {
                "timestamp": datetime.now().isoformat(),
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
    
    async def execute(self, operation: Optional[str] = None, **kwargs) -> Any:
        """Execute a specific operation."""
        context = kwargs
        if operation:
            context["operation"] = operation
        return await self.evaluate(context)
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            logger.info("Cleaning up Hedge Fund Manager Agent resources...")
            
            # Close market data connection
            if hasattr(self, 'market_data'):
                await self.market_data.close()
                
            # Close NEAR connection
            if hasattr(self, 'near'):
                await self.near.close()
                
            # Close LLM provider
            if hasattr(self, 'llm_provider'):
                await self.llm_provider.close()
                
            logger.info("Hedge Fund Manager Agent cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise

# Register the plugin
from near_swarm.plugins import register_plugin
register_plugin("hedge-fund", HedgeFundPlugin)