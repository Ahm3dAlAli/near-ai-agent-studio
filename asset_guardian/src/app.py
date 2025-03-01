"""
DeFAI Asset Guardian - Main Application
Intelligent Portfolio Management on NEAR Blockchain

This is the main application entry point for the DeFAI Asset Guardian system.
It demonstrates the core functionality of the system, including:
1. Real-time portfolio monitoring
2. Risk assessment and mitigation
3. Automated strategy execution
4. Market insights and predictions

This application was created for the "Seeds of Agentic Future" hackathon.
"""

import asyncio
import logging
import json
import os
from datetime import datetime
import time
from typing import Dict, Any, List
import argparse
from dotenv import load_dotenv

# Import our DeFAI Asset Guardian system
from .defai_asset_guardian import DeFAIAssetGuardian

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("defai_guardian.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ASCII Art Banner
BANNER = """
██████╗ ███████╗███████╗ █████╗ ██╗     █████╗ ███████╗███████╗███████╗████████╗     ██████╗ ██╗   ██╗ █████╗ ██████╗ ██████╗ ██╗ █████╗ ███╗   ██╗
██╔══██╗██╔════╝██╔════╝██╔══██╗██║    ██╔══██╗██╔════╝██╔════╝██╔════╝╚══██╔══╝    ██╔════╝ ██║   ██║██╔══██╗██╔══██╗██╔══██╗██║██╔══██╗████╗  ██║
██║  ██║█████╗  █████╗  ███████║██║    ███████║███████╗███████╗█████╗     ██║       ██║  ███╗██║   ██║███████║██████╔╝██║  ██║██║███████║██╔██╗ ██║
██║  ██║██╔══╝  ██╔══╝  ██╔══██║██║    ██╔══██║╚════██║╚════██║██╔══╝     ██║       ██║   ██║██║   ██║██╔══██║██╔══██╗██║  ██║██║██╔══██║██║╚██╗██║
██████╔╝███████╗██║     ██║  ██║██║    ██║  ██║███████║███████║███████╗   ██║       ╚██████╔╝╚██████╔╝██║  ██║██║  ██║██████╔╝██║██║  ██║██║ ╚████║
╚═════╝ ╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝    ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝   ╚═╝        ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝
"""

class DeFAIAssetGuardianApp:
    """
    Main application for demonstrating DeFAI Asset Guardian capabilities
    
    This application provides a simple console interface to interact with
    the DeFAI Asset Guardian system and demonstrate its capabilities.
    """
    
    def __init__(self, demo_mode: bool = False):
        """Initialize application"""
        self.guardian = None
        self.running = False
        self.last_update = None
        self.update_interval = 300  # 5 minutes between updates
        self.demo_mode = demo_mode
        
        # Demo mode uses shorter intervals
        if demo_mode:
            self.update_interval = 60  # 1 minute between updates
    
    async def initialize(self) -> None:
        """Initialize the application"""
        logger.info("Initializing DeFAI Asset Guardian application")
        
        # Load environment variables
        load_dotenv()
        
        # Print banner
        print(BANNER)
        print("\nInitializing DeFAI Asset Guardian for NEAR Protocol...")
        
        # Create and initialize the guardian
        self.guardian = DeFAIAssetGuardian()
        await self.guardian.initialize()
        
        # Set initial settings
        await self.guardian.update_settings({
            'simulation_mode': True,  # Always start in simulation mode for safety
            'auto_execution': False,  # Start with manual execution
            'risk_tolerance': 'medium',
            'update_interval': self.update_interval
        })
        
        # Initial update
        await self.guardian.update()
        self.last_update = datetime.now()
        
        logger.info("DeFAI Asset Guardian application initialized")
        print("\n✅ DeFAI Asset Guardian initialized successfully!\n")
    
    async def run(self) -> None:
        """Run the application main loop"""
        if not self.guardian:
            raise ValueError("Application not initialized")
            
        self.running = True
        
        # Print welcome message
        self._print_welcome()
        
        try:
            while self.running:
                # Check for updates
                now = datetime.now()
                time_diff = (now - self.last_update).total_seconds()
                
                if time_diff >= self.update_interval:
                    print("\n🔄 Updating portfolio and market data...")
                    await self.guardian.update()
                    self.last_update = now
                    print("✅ Update complete!")
                
                # Process user command
                await self._process_command()
                
                # Short sleep to prevent CPU usage
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\nExiting application...")
            self.running = False
            
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            print(f"\n❌ Error: {e}")
            self.running = False
            
        finally:
            await self.cleanup()
    
    def _print_welcome(self) -> None:
        """Print welcome message and instructions"""
        print("\n" + "=" * 80)
        print("Welcome to DeFAI Asset Guardian - Intelligent Portfolio Management on NEAR")
        print("=" * 80)
        print("\nThis application demonstrates AI-powered portfolio management using")
        print("a swarm of specialized agents that collaborate to monitor, analyze,")
        print("and optimize your NEAR ecosystem investments.")
        print("\nCurrent Settings:")
        print(f"• Simulation Mode: {self.guardian.settings['simulation_mode']}")
        print(f"• Auto-Execution: {self.guardian.settings['auto_execution']}")
        print(f"• Risk Tolerance: {self.guardian.settings['risk_tolerance']}")
        print(f"• Update Interval: {self.guardian.settings['update_interval']} seconds")
        print("\nAvailable Commands:")
        print("1. portfolio   - View your current portfolio")
        print("2. insights    - Get detailed portfolio insights and recommendations")
        print("3. market      - View current market analysis")
        print("4. risk        - View risk assessment")
        print("5. strategy    - Generate a new strategy")
        print("6. execute     - Execute the latest strategy")
        print("7. history     - View transaction history")
        print("8. settings    - Update system settings")
        print("9. help        - Show this help message")
        print("10. exit       - Exit the application")
        print("\nEnter a command to get started:")
    
    async def _process_command(self) -> None:
        """Process user command"""
        command = input("\n> ").strip().lower()
        
        if command == 'exit' or command == 'quit':
            self.running = False
            return
            
        elif command == 'help':
            self._print_welcome()
            
        elif command == 'portfolio':
            self._print_portfolio()
            
        elif command == 'insights':
            await self._print_insights()
            
        elif command == 'market':
            self._print_market_analysis()
            
        elif command == 'risk':
            self._print_risk_assessment()
            
        elif command == 'strategy':
            await self._generate_strategy()
            
        elif command == 'execute':
            await self._execute_strategy()
            
        elif command == 'history':
            await self._print_history()
            
        elif command == 'settings':
            await self._update_settings()
            
        elif command == 'demo':
            await self._run_demo()
            
        else:
            print(f"\n❌ Unknown command: {command}")
            print("Type 'help' to see available commands")
    
    def _print_portfolio(self) -> None:
        """Print current portfolio"""
        portfolio = self.guardian.portfolio
        
        print("\n📊 Current Portfolio")
        print("=" * 50)
        print(f"Total Value: ${portfolio.get('total_value_usd', 0):,.2f}")
        print(f"Last Updated: {portfolio.get('updated_at', 'Unknown')}")
        print("\nAssets:")
        
        assets = portfolio.get('assets', [])
        for asset in assets:
            print(f"\n• {asset.get('symbol', 'Unknown')}")
            print(f"  Balance: {asset.get('balance', 0):,.6f}")
            print(f"  Value: ${asset.get('value_usd', 0):,.2f}")
            print(f"  Allocation: {asset.get('allocation', 0):.2f}%")
            print(f"  Price: ${asset.get('price_usd', 0):,.2f}")
    
    async def _print_insights(self) -> None:
        """Print detailed portfolio insights"""
        print("\n🔍 Generating portfolio insights...")
        insights = await self.guardian.get_portfolio_insights()
        
        if 'error' in insights:
            print(f"\n❌ Error: {insights['error']}")
            return
            
        print("\n📈 Portfolio Insights")
        print("=" * 50)
        
        # Portfolio summary
        summary = insights.get('portfolio_summary', {})
        print(f"Total Value: ${summary.get('total_value', 0):,.2f}")
        print(f"Asset Count: {summary.get('asset_count', 0)}")
        
        # Asset allocation
        print("\nAsset Allocation:")
        for asset in insights.get('asset_allocation', []):
            print(f"• {asset.get('symbol', 'Unknown')}: {asset.get('allocation', 0):.2f}%")
        
        # Risk metrics
        risk = insights.get('risk_metrics', {})
        print(f"\nRisk Profile: {risk.get('overall_risk', 0)}/100")
        print(f"Diversification Score: {risk.get('diversification', 0)}/10")
        print("\nKey Risk Factors:")
        for factor in risk.get('risk_factors', [])[:3]:
            print(f"• {factor}")
        
        # Market trends
        market = insights.get('market_trends', {})
        print(f"\nMarket Sentiment: {market.get('sentiment', 50)}/100")
        print("\nCurrent Trends:")
        for trend in market.get('trends', [])[:3]:
            print(f"• {trend}")
        
        # Recent strategies
        print("\nRecent Strategies:")
        for strategy in insights.get('recent_strategies', []):
            print(f"• {strategy.get('operation_type', 'unknown')} - {strategy.get('status', 'unknown')} - {strategy.get('confidence', 0):.2f} confidence")
    
    def _print_market_analysis(self) -> None:
        """Print current market analysis"""
        analysis = self.guardian.market_analysis
        
        print("\n📊 Market Analysis")
        print("=" * 50)
        
        if 'error' in analysis:
            print(f"\n❌ Error: {analysis['error']}")
            return
            
        print(f"Confidence: {analysis.get('confidence', 0):.2f}")
        print(f"Sentiment Score: {analysis.get('sentiment_score', 50)}/100")
        
        print("\nMarket Trends:")
        for trend in analysis.get('trends', []):
            print(f"• {trend}")
            
        print("\nOpportunities:")
        for opportunity in analysis.get('opportunities', []):
            print(f"• {opportunity}")
            
        print("\nRisk Factors:")
        for risk in analysis.get('risk_factors', []):
            print(f"• {risk}")
    
    def _print_risk_assessment(self) -> None:
        """Print current risk assessment"""
        assessment = self.guardian.risk_assessment
        
        print("\n⚠️ Risk Assessment")
        print("=" * 50)
        
        if 'error' in assessment:
            print(f"\n❌ Error: {assessment['error']}")
            return
            
        print(f"Overall Risk Score: {assessment.get('risk_score', 50)}/100")
        print(f"Confidence: {assessment.get('confidence', 0):.2f}")
        
        print("\nRisk Factors:")
        for factor in assessment.get('risk_factors', []):
            print(f"• {factor}")
            
        print("\nExposure Analysis:")
        print(assessment.get('exposure_analysis', 'No exposure analysis available'))
        
        print("\nRisk Mitigation:")
        for mitigation in assessment.get('risk_mitigation', []):
            print(f"• {mitigation}")
    
    async def _generate_strategy(self) -> None:
        """Generate a new strategy"""
        print("\n🧠 Generating investment strategy...")
        
        # Ask for strategy type
        print("\nStrategy Type:")
        print("1. Rebalance - Optimize current portfolio allocation")
        print("2. Entry - Identify new investment opportunities")
        print("3. Exit - Recommend positions to close")
        
        strategy_choice = input("\nChoose strategy type (1-3): ").strip()
        
        strategy_type = "rebalance"  # Default
        if strategy_choice == "2":
            strategy_type = "entry"
        elif strategy_choice == "3":
            strategy_type = "exit"
            
        print(f"\nGenerating {strategy_type} strategy...")
        
        # Generate strategy
        strategy = await self.guardian.generate_strategy(operation_type=strategy_type)
        
        if 'error' in strategy:
            print(f"\n❌ Error: {strategy['error']}")
            return
            
        print("\n✅ Strategy Generated")
        print("=" * 50)
        print(f"Confidence: {strategy.get('confidence', 0):.2f}")
        print(f"Expected Outcome: {strategy.get('expected_outcome', 'Unknown')}")
        
        print("\nRecommended Actions:")
        for i, action in enumerate(strategy.get('actions', [])):
            print(f"\n• Action {i+1}: {action.get('type', 'unknown').upper()} {action.get('asset', 'unknown')}")
            print(f"  Amount: {action.get('percentage', 0)}% of portfolio")
            if 'reason' in action:
                print(f"  Reason: {action.get('reason')}")
        
        print("\nReasoning:")
        for reason in strategy.get('reasoning', []):
            print(f"• {reason}")
            
        print("\nExecution Details:")
        execution = strategy.get('execution_details', {})
        print(f"• Estimated Gas: {execution.get('estimated_gas', 0)}")
        print("\nBest Protocols:")
        for asset, protocol in execution.get('best_protocols', {}).items():
            print(f"• {asset}: {protocol}")
    
    async def _execute_strategy(self) -> None:
        """Execute the latest strategy"""
        print("\n⚙️ Executing latest strategy...")
        
        # Check if we have any active strategies
        if not self.guardian.active_strategies:
            print("\n❌ No active strategies to execute")
            return
            
        # Confirm execution
        latest = self.guardian.active_strategies[-1]
        strategy_data = latest.get('strategy', {})
        
        print("\nLatest Strategy:")
        print(f"• Type: {strategy_data.get('operation_type', 'unknown')}")
        print(f"• Confidence: {strategy_data.get('confidence', 0):.2f}")
        print(f"• Actions: {len(strategy_data.get('actions', []))}")
        
        if self.guardian.settings['simulation_mode']:
            print("\n⚠️ Running in SIMULATION mode - no real transactions will be executed")
        else:
            print("\n⚠️ Running in PRODUCTION mode - real transactions will be executed")
            
        confirm = input("\nAre you sure you want to execute this strategy? (y/n): ").strip().lower()
        
        if confirm != 'y':
            print("\nExecution cancelled")
            return
            
        # Execute strategy
        print("\nExecuting strategy...")
        result = await self.guardian.execute_strategy()
        
        if result.get('status') == 'error':
            print(f"\n❌ Execution failed: {result.get('error')}")
            return
            
        print(f"\n✅ Strategy {result.get('status')}")
        
        if result.get('status') == 'simulated':
            print("\n⚠️ This was a simulation - no real transactions were executed")
            print("To execute real transactions, change simulation_mode setting to False")
            return
            
        print("\nExecution Results:")
        for exec_result in result.get('execution_results', []):
            action = exec_result.get('action', {})
            result_data = exec_result.get('result', {})
            
            print(f"\n• {action.get('type', 'unknown').upper()} {action.get('asset', 'unknown')}")
            print(f"  Protocol: {exec_result.get('protocol', 'unknown')}")
            print(f"  Status: {result_data.get('status', 'unknown')}")
            print(f"  Transaction Hash: {result_data.get('transaction_hash', 'N/A')}")
    
    async def _print_history(self) -> None:
        """Print transaction history"""
        print("\n📜 Transaction History")
        print("=" * 50)
        
        history = await self.guardian.get_transaction_history()
        
        if not history:
            print("\nNo transaction history available")
            return
            
        for tx in history:
            print(f"\n• {tx.get('action_type', 'unknown').upper()} {tx.get('asset', 'unknown')}")
            print(f"  Protocol: {tx.get('protocol', 'unknown')}")
            print(f"  Amount: {tx.get('percentage', 0)}% of portfolio")
            print(f"  Status: {tx.get('status', 'unknown')}")
            print(f"  Transaction Hash: {tx.get('tx_hash', 'N/A')}")
            print(f"  Timestamp: {tx.get('timestamp', 'unknown')}")
    
    async def _update_settings(self) -> None:
        """Update system settings"""
        print("\n⚙️ System Settings")
        print("=" * 50)
        
        print("\nCurrent Settings:")
        for key, value in self.guardian.settings.items():
            print(f"• {key}: {value}")
            
        print("\nUpdate Settings:")
        print("1. simulation_mode - Enable/disable simulation mode")
        print("2. auto_execution - Enable/disable automatic strategy execution")
        print("3. risk_tolerance - Set risk tolerance (low/medium/high)")
        print("4. update_interval - Set update interval in seconds")
        print("5. emergency_stop_loss - Enable/disable emergency stop loss")
        print("6. notification_level - Set notification level (low/medium/high)")
        print("7. Back to main menu")
        
        choice = input("\nChoose setting to update (1-7): ").strip()
        
        if choice == '7':
            return
        
        if choice == '1':
            value = input("Enable simulation mode? (true/false): ").strip().lower()
            if value in ['true', 'false']:
                await self.guardian.update_settings({'simulation_mode': value == 'true'})
                
        elif choice == '2':
            value = input("Enable auto execution? (true/false): ").strip().lower()
            if value in ['true', 'false']:
                await self.guardian.update_settings({'auto_execution': value == 'true'})
                
        elif choice == '3':
            value = input("Set risk tolerance (low/medium/high): ").strip().lower()
            if value in ['low', 'medium', 'high']:
                await self.guardian.update_settings({'risk_tolerance': value})
                
        elif choice == '4':
            try:
                value = int(input("Set update interval in seconds: ").strip())
                if value > 0:
                    await self.guardian.update_settings({'update_interval': value})
                    self.update_interval = value
            except ValueError:
                print("Invalid value - must be a positive integer")
                
        elif choice == '5':
            value = input("Enable emergency stop loss? (true/false): ").strip().lower()
            if value in ['true', 'false']:
                await self.guardian.update_settings({'emergency_stop_loss': value == 'true'})
                
        elif choice == '6':
            value = input("Set notification level (low/medium/high): ").strip().lower()
            if value in ['low', 'medium', 'high']:
                await self.guardian.update_settings({'notification_level': value})
                
        print("\n✅ Settings updated")
        print("\nCurrent Settings:")
        for key, value in self.guardian.settings.items():
            print(f"• {key}: {value}")
    
    async def _run_demo(self) -> None:
        """Run automated demo to showcase functionality"""
        print("\n🎮 Running Automated Demo")
        print("=" * 50)
        print("\nThis demo will showcase the core capabilities of DeFAI Asset Guardian:")
        print("1. Portfolio monitoring and analysis")
        print("2. Risk assessment and mitigation")
        print("3. Strategy generation and execution (simulated)")
        print("4. Multi-agent collaboration")
        
        input("\nPress Enter to start the demo...")
        
        try:
            # Step 1: Portfolio and Market Analysis
            print("\n📊 Step 1: Portfolio and Market Analysis")
            print("Getting current portfolio data...")
            await asyncio.sleep(1)
            self._print_portfolio()
            
            print("\nAnalyzing market conditions...")
            await asyncio.sleep(2)
            self._print_market_analysis()
            
            input("\nPress Enter to continue to Risk Assessment...")
            
            # Step 2: Risk Assessment
            print("\n⚠️ Step 2: Risk Assessment")
            print("Performing comprehensive risk assessment...")
            await asyncio.sleep(2)
            self._print_risk_assessment()
            
            input("\nPress Enter to continue to Strategy Generation...")
            
            # Step 3: Strategy Generation
            print("\n🧠 Step 3: Strategy Generation")
            print("Generating optimized portfolio strategy...")
            await asyncio.sleep(2)
            strategy = await self.guardian.generate_strategy(operation_type="rebalance")
            
            print("\n✅ Strategy Generated")
            print("=" * 50)
            print(f"Confidence: {strategy.get('confidence', 0):.2f}")
            print(f"Expected Outcome: {strategy.get('expected_outcome', 'Unknown')}")
            
            print("\nRecommended Actions:")
            for i, action in enumerate(strategy.get('actions', [])):
                print(f"\n• Action {i+1}: {action.get('type', 'unknown').upper()} {action.get('asset', 'unknown')}")
                print(f"  Amount: {action.get('percentage', 0)}% of portfolio")
                if 'reason' in action:
                    print(f"  Reason: {action.get('reason')}")
            
            input("\nPress Enter to continue to Strategy Execution (Simulation)...")
            
            # Step 4: Strategy Execution
            print("\n⚙️ Step 4: Strategy Execution (Simulation)")
            print("Executing strategy in simulation mode...")
            await asyncio.sleep(2)
            result = await self.guardian.execute_strategy()
            
            print(f"\n✅ Strategy {result.get('status')}")
            print("\n⚠️ This was a simulation - no real transactions were executed")
            
            # Step 5: Portfolio Insights
            print("\n🔍 Step 5: Portfolio Insights")
            print("Generating comprehensive portfolio insights...")
            await asyncio.sleep(2)
            await self._print_insights()
            
            print("\n🎉 Demo Complete!")
            print("=" * 50)
            print("\nThis demo showcased the core capabilities of DeFAI Asset Guardian.")
            print("In a real-world scenario, the system would continuously monitor your")
            print("portfolio, assess risks, and execute optimized strategies based on")
            print("market conditions and your risk tolerance.")
            
        except Exception as e:
            logger.error(f"Demo error: {e}")
            print(f"\n❌ Demo error: {e}")
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        if self.guardian:
            await self.guardian.cleanup()
            logger.info("DeFAI Asset Guardian application cleaned up")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='DeFAI Asset Guardian Application')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode')
    args = parser.parse_args()
    
    app = DeFAIAssetGuardianApp(demo_mode=args.demo)
    try:
        await app.initialize()
        await app.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"\n❌ Application error: {e}")
    finally:
        await app.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nApplication terminated by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        print(f"\n❌ Unhandled exception: {e}")