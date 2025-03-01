# NearAI Hedge Fund - Running Guide

This guide provides step-by-step instructions for setting up and running the NearAI Hedge Fund with the NEAR AI Agent Studio framework.

## Prerequisites

- Python 3.12+ installed
- NEAR CLI installed (`npm install -g near-cli`)
- Rust and Cargo installed (for smart contract compilation)
- Node.js 16+ (for dashboard)
- NEAR testnet account

## Step 1: Set Up Environment

1. **Clone the NEAR AI Agent Studio repository**:
   ```bash
   git clone https://github.com/jbarnes850/near-ai-agent-studio.git
   cd near-ai-agent-studio
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .
   ```

4. **Create or update your `.env` file**:
   ```bash
   cp .env.example .env
   ```

5. **Add your credentials to the `.env` file**:
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

## Step 2: Prepare File Structure

Ensure your project structure includes the NearAI Hedge Fund files:

1. **Create required directories**:
   ```bash
   mkdir -p near_swarm/agents/hedge_fund
   mkdir -p near_swarm/examples/nearai_hedge_fund
   mkdir -p contracts/nearai_hedge_fund/src
   ```

2. **Place the files in their respective locations**:
   - `near_swarm/agents/hedge_fund/agent.yaml`
   - `near_swarm/agents/hedge_fund/plugin.py`
   - `near_swarm/examples/nearai_hedge_fund/run_hedge_fund.py`
   - `near_swarm/examples/nearai_hedge_fund/README.md`
   - `contracts/nearai_hedge_fund/Cargo.toml`
   - `contracts/nearai_hedge_fund/src/lib.rs`

## Step 3 (Optional): Deploy the Smart Contract

If you want to deploy the smart contract to interact with the fund on-chain:

1. **Compile the smart contract**:
   ```bash
   cd contracts/nearai_hedge_fund
   cargo build --target wasm32-unknown-unknown --release
   ```

2. **Deploy the contract to your NEAR testnet account**:
   ```bash
   near deploy --wasmFile target/wasm32-unknown-unknown/release/nearai_hedge_fund.wasm --accountId YOUR_ACCOUNT.testnet
   ```

3. **Initialize the contract**:
   ```bash
   near call YOUR_ACCOUNT.testnet new '{"owner_id": "YOUR_ACCOUNT.testnet"}' --accountId YOUR_ACCOUNT.testnet
   ```

## Step 4: Run the Hedge Fund

### Option 1: Using the Runner Script

1. **Execute a complete investment cycle**:
   ```bash
   python -m near_swarm.examples.nearai_hedge_fund.run_hedge_fund
   ```

   This will:
   - Analyze the market
   - Evaluate investment strategies
   - Assess risk
   - Optimize the portfolio
   - Build consensus among agents
   - Execute a transaction if consensus is reached

2. **Run in simulation mode** (no actual transactions):
   ```bash
   python -m near_swarm.examples.nearai_hedge_fund.run_hedge_fund --simulation
   ```

3. **Run specific operations**:
   ```bash
   # Market analysis only
   python -m near_swarm.examples.nearai_hedge_fund.run_hedge_fund --operation analyze

   # Strategy evaluation only
   python -m near_swarm.examples.nearai_hedge_fund.run_hedge_fund --operation strategy

   # Risk assessment only
   python -m near_swarm.examples.nearai_hedge_fund.run_hedge_fund --operation risk

   # Portfolio optimization only
   python -m near_swarm.examples.nearai_hedge_fund.run_hedge_fund --operation portfolio
   ```

### Option 2: Using the NEAR Swarm CLI

1. **Run the hedge fund agent using the CLI**:
   ```bash
   near-swarm execute hedge-fund --operation execute
   ```

2. **Run specific operations**:
   ```bash
   near-swarm execute hedge-fund --operation analyze
   ```

## Expected Output

When running a complete investment cycle, you should see output similar to:

```
================================================================================
                             NearAI Hedge Fund                              
                   AI-powered Investment on NEAR Blockchain                   
================================================================================

Connected to: testnet
Account: your-account.testnet
Mode: Live Trading
LLM Provider: hyperbolic
LLM Model: meta-llama/Llama-3.3-70B-Instruct

Current NEAR Price: $3.24
24h Change: 2.30%

Executing operation: execute

=== Investment Cycle Results ===

Market Analysis:
Trend: bullish
Confidence: 85.00%

Strategy:
Action: buy
Confidence: 82.00%

Risk Assessment:
Risk Level: medium
Confidence: 78.00%

Portfolio Optimization:
Action: buy
Amount: 1.5
Confidence: 80.00%

Consensus:
Reached: True
Approval Rate: 81.25%

Transaction:
Status: success
Type: buy
Amount: 1.5
Transaction ID: ABCDefgh12345...

Operation completed successfully!
```

## Troubleshooting

### Common Issues and Solutions

1. **Missing environment variables**:
   - Error: "Missing required environment variables"
   - Solution: Ensure all required variables are set in your `.env` file

2. **Agent plugin not found**:
   - Error: "Failed to load hedge-fund plugin"
   - Solution: Verify the plugin files are in the correct location and properly registered

3. **LLM API connection issues**:
   - Error: "Error connecting to LLM API"
   - Solution: Check your API key and internet connection

4. **NEAR wallet connection issues**:
   - Error: "Error connecting to NEAR wallet"
   - Solution: Verify your NEAR credentials and network configuration

5. **Smart contract deployment errors**:
   - Error: "Error deploying contract"
   - Solution: Ensure you have enough NEAR tokens for deployment and correct account permissions

### Debugging

To get more detailed logs, set the logging level to DEBUG:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Next Steps

After successfully running the NearAI Hedge Fund:

1. **Customize the agent behavior** by modifying parameters in `agent.yaml`
2. **Add more specialized agents** to improve decision-making
3. **Connect the dashboard** to visualize fund performance
4. **Implement additional strategies** for different market conditions
5. **Set up automated running** using cron jobs or other schedulers

For more details on the architecture and customization options, refer to the main README file in `near_swarm/examples/nearai_hedge_fund/README.md`.