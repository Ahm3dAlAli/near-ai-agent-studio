# NearAI Hedge Fund Example

This example demonstrates how to create an AI-powered hedge fund using the NEAR AI Agent Studio framework. The hedge fund uses swarm intelligence to combine specialized AI agents with different expertise areas to make collective investment decisions on the NEAR blockchain.

## Features

- **AI Agent Swarm**: Multi-agent system with specialized roles for market analysis, risk management, strategy optimization, and portfolio management
- **Decentralized Consensus**: Collective decision-making with configurable consensus parameters
- **NEAR Blockchain Integration**: Fund management and transaction execution on NEAR
- **Smart Contract Management**: Investor shares, deposits, withdrawals, and portfolio tracking
- **Real-time Market Analysis**: Integration with market data sources

## Prerequisites

Before running this example, make sure you have:

1. Set up your NEAR testnet account and added credentials to your `.env` file
2. Installed the NEAR AI Agent Studio framework and its dependencies
3. Configured your LLM provider settings in the `.env` file

## Configuration

Create or update your `.env` file with the following variables:

```
# NEAR Configuration
NEAR_NETWORK=testnet
NEAR_ACCOUNT_ID=your-account.testnet
NEAR_PRIVATE_KEY=your-private-key
NEAR_RPC_URL=https://rpc.testnet.fastnear.com

# LLM Configuration
LLM_PROVIDER=hyperbolic
LLM_API_KEY=your-llm-api-key
LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000
LLM_API_URL=https://api.hyperbolic.xyz/v1
```

## Usage

### Running the Hedge Fund

To run the hedge fund example, use the provided runner script:

```bash
# Run a complete investment cycle (analyze, strategy, risk, portfolio, consensus, transaction)
python -m near_swarm.examples.nearai_hedge_fund.run_hedge_fund --operation execute

# Run in simulation mode (no real transactions)
python -m near_swarm.examples.nearai_hedge_fund.run_hedge_fund --operation execute --simulation

# Only perform market analysis
python -m near_swarm.examples.nearai_hedge_fund.run_hedge_fund --operation analyze

# Only evaluate investment strategy
python -m near_swarm.examples.nearai_hedge_fund.run_hedge_fund --operation strategy

# Only perform risk assessment
python -m near_swarm.examples.nearai_hedge_fund.run_hedge_fund --operation risk

# Only optimize portfolio
python -m near_swarm.examples.nearai_hedge_fund.run_hedge_fund --operation portfolio
```

### Using the Hedge Fund Agent in Your Own Applications

You can also use the hedge fund agent in your own applications:

```python
from near_swarm.plugins import PluginLoader

async def main():
    # Initialize plugin loader
    plugin_loader = PluginLoader()
    
    # Load hedge fund plugin
    plugin = await plugin_loader.load_plugin("hedge-fund")
    
    # Analyze market
    market_analysis = await plugin.execute(operation="analyze")
    print(f"Market trend: {market_analysis.get('trend')}")
    print(f"Confidence: {market_analysis.get('confidence')}")
    
    # Clean up
    await plugin_loader.unload_plugin("hedge-fund")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Understanding the Architecture

The NearAI Hedge Fund is built on the following components:

1. **Hedge Fund Agent Plugin**: The main agent plugin that coordinates the investment process
2. **Market Data Manager**: Integrates with market data sources to get prices and market information
3. **Consensus Manager**: Handles the consensus-building process among specialized agents
4. **NEAR Integration**: Interacts with the NEAR blockchain for transaction execution
5. **Memory Manager**: Stores and retrieves historical data and decision records

The investment cycle consists of these steps:

1. **Market Analysis**: Analyze current market conditions and trends
2. **Strategy Evaluation**: Develop investment strategies based on market analysis
3. **Risk Assessment**: Evaluate risk exposure and mitigation strategies
4. **Portfolio Optimization**: Optimize portfolio allocation based on strategy and risk
5. **Consensus Building**: Build consensus among specialized agents
6. **Transaction Execution**: Execute trades on the NEAR blockchain

## Smart Contract Integration

This example can be extended with the provided smart contract for full on-chain fund management:

1. Deploy the smart contract to your NEAR testnet account
2. Use the contract methods to manage investor deposits and withdrawals
3. Execute trades through the contract
4. Track fund performance and investor positions

## Dashboard UI Integration

The example includes a React dashboard component that can be integrated into your frontend application:

1. Add the dashboard component to your React application
2. Connect it to your backend API
3. Use it to visualize fund performance and investor positions

## Customization

You can customize the hedge fund by modifying the following:

- **Agent Configuration**: Update settings in `agent.yaml`
- **Investment Strategies**: Add new strategies in the plugin implementation
- **Risk Parameters**: Adjust risk thresholds in the agent settings
- **Consensus Rules**: Modify consensus parameters in the agent settings

## Next Steps

After exploring this example, you can:

1. Implement additional specialized agents
2. Add support for multiple token investments
3. Integrate with additional data sources
4. Implement advanced risk management strategies
5. Create a full-featured investor interface

## License

This example is licensed under the MIT License - see the [LICENSE](../../../LICENSE) file for details.