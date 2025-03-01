#!/usr/bin/env python3
"""
NearAI Hedge Fund Runner Script
Run the NearAI Hedge Fund example with the NEAR AI Agent Studio framework
"""

import asyncio
import logging
import os
import argparse
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from near_swarm.plugins import PluginLoader
from near_swarm.core.market_data import MarketDataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_hedge_fund(operation: str = "execute", simulation: bool = False) -> None:
    """Run the NearAI Hedge Fund."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Check required environment variables
        required_vars = ['NEAR_ACCOUNT_ID', 'NEAR_PRIVATE_KEY', 'LLM_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            logger.error("Please set these variables in your .env file and try again.")
            return
        
        # Initialize plugin loader
        plugin_loader = PluginLoader()
        
        # Load hedge fund plugin
        logger.info("Loading Hedge Fund plugin...")
        plugin = await plugin_loader.load_plugin("hedge-fund")
        if not plugin:
            logger.error("Failed to load hedge-fund plugin")
            logger.error("Make sure you have installed the plugin correctly.")
            return
        
        # Print welcome message
        print("\n" + "=" * 80)
        print(f"{'NearAI Hedge Fund'.center(80)}")
        print(f"{'AI-powered Investment on NEAR Blockchain'.center(80)}")
        print("=" * 80)
        
        # Print connection info
        print(f"\nConnected to: {os.getenv('NEAR_NETWORK', 'testnet')}")
        print(f"Account: {os.getenv('NEAR_ACCOUNT_ID')}")
        print(f"Mode: {'Simulation' if simulation else 'Live Trading'}")
        print(f"LLM Provider: {os.getenv('LLM_PROVIDER', 'hyperbolic')}")
        print(f"LLM Model: {os.getenv('LLM_MODEL', 'meta-llama/Llama-3.3-70B-Instruct')}")
        
        # Initialize market data manager to get current price
        async with MarketDataManager() as market:
            near_data = await market.get_token_price('near')
            print(f"\nCurrent NEAR Price: ${near_data['price']:.2f}")
            print(f"24h Change: {near_data.get('price_change_24h', 0):.2f}%")
            
        # Execute the specified operation
        print(f"\nExecuting operation: {operation}\n")
        
        # Execute operation
        result = await plugin.execute(operation=operation)
        
        # Display results based on operation
        if operation == "analyze":
            print("\n=== Market Analysis Results ===")
            print(f"Observation: {result.get('observation', 'N/A')}")
            print(f"Trend: {result.get('trend', 'N/A')}")
            print(f"Confidence: {result.get('confidence', 0):.2%}\n")
            print(f"Conclusion: {result.get('conclusion', 'N/A')}")
            
        elif operation == "strategy":
            print("\n=== Strategy Evaluation Results ===")
            print(f"Strategy: {result.get('strategy', 'N/A')}")
            print(f"Action: {result.get('action', 'N/A')}")
            print(f"Amount: {result.get('amount', 'N/A')}")
            print(f"Confidence: {result.get('confidence', 0):.2%}\n")
            print(f"Rationale: {result.get('rationale', 'N/A')}")
            
        elif operation == "risk":
            print("\n=== Risk Assessment Results ===")
            print(f"Risk Level: {result.get('risk_level', 'N/A')}")
            print(f"Stop Loss: {result.get('stop_loss', 'N/A')}")
            print(f"Confidence: {result.get('confidence', 0):.2%}\n")
            print(f"Assessment: {result.get('exposure_assessment', 'N/A')}")
            
        elif operation == "portfolio":
            print("\n=== Portfolio Optimization Results ===")
            print(f"Action: {result.get('action', 'N/A')}")
            print(f"Amount: {result.get('amount', 'N/A')}")
            print(f"Confidence: {result.get('confidence', 0):.2%}\n")
            print(f"Analysis: {result.get('analysis', 'N/A')}")
            
        elif operation == "execute":
            # For the complete investment cycle, we show a comprehensive summary
            print("\n=== Investment Cycle Results ===")
            
            # Market Analysis
            market_analysis = result.get('market_analysis', {})
            print("\nMarket Analysis:")
            print(f"Trend: {market_analysis.get('trend', 'N/A')}")
            print(f"Confidence: {market_analysis.get('confidence', 0):.2%}")
            
            # Strategy
            strategy = result.get('strategy', {})
            print("\nStrategy:")
            print(f"Action: {strategy.get('action', 'N/A')}")
            print(f"Confidence: {strategy.get('confidence', 0):.2%}")
            
            # Risk Assessment
            risk = result.get('risk_assessment', {})
            print("\nRisk Assessment:")
            print(f"Risk Level: {risk.get('risk_level', 'N/A')}")
            print(f"Confidence: {risk.get('confidence', 0):.2%}")
            
            # Portfolio Optimization
            portfolio = result.get('portfolio_optimization', {})
            print("\nPortfolio Optimization:")
            print(f"Action: {portfolio.get('action', 'N/A')}")
            print(f"Amount: {portfolio.get('amount', 'N/A')}")
            print(f"Confidence: {portfolio.get('confidence', 0):.2%}")
            
            # Consensus
            consensus = result.get('consensus', {})
            print("\nConsensus:")
            print(f"Reached: {consensus.get('consensus', False)}")
            print(f"Approval Rate: {consensus.get('approval_rate', 0):.2%}")
            
            # Transaction
            transaction = result.get('transaction', {})
            print("\nTransaction:")
            print(f"Status: {transaction.get('status', 'N/A')}")
            if transaction.get('status') == "success":
                print(f"Type: {transaction.get('type', 'N/A')}")
                print(f"Amount: {transaction.get('amount', 'N/A')}")
                print(f"Transaction ID: {transaction.get('transaction_id', 'N/A')}")
            elif transaction.get('status') == "skipped":
                print(f"Reason: {transaction.get('reason', 'N/A')}")
        
        print("\nOperation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running hedge fund: {e}")
    finally:
        # Clean up
        if 'plugin' in locals() and plugin:
            await plugin_loader.unload_plugin("hedge-fund")

def main():
    """Main entry point for the command line interface."""
    parser = argparse.ArgumentParser(description="Run the NearAI Hedge Fund")
    parser.add_argument("--operation", "-o", default="execute", choices=["analyze", "strategy", "risk", "portfolio", "execute"],
                      help="Operation to perform (default: execute)")
    parser.add_argument("--simulation", "-s", action="store_true", help="Run in simulation mode (no real trades)")
    args = parser.parse_args()
    
    asyncio.run(run_hedge_fund(args.operation, args.simulation))

if __name__ == "__main__":
    main()