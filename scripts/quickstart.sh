#!/bin/bash

# Exit on error and enable debugging
set -e
set -x

# Create log directory if it doesn't exist
mkdir -p logs

# Redirect all output to both console and log file
exec 1> >(tee -a "logs/quickstart_$(date +%s).log") 2>&1

clear

# Terminal width (default 80 if can't be detected)
TERM_WIDTH=$(tput cols 2>/dev/null || echo 80)

# Function to center text
center_text() {
    local text="$1"
    local width="${2:-$TERM_WIDTH}"
    local padding=$(( (width - ${#text}) / 2 ))
    printf "%${padding}s%s%${padding}s\n" "" "$text" ""
}

# NEAR brand color (blue)
echo -e "\033[38;5;75m"

# Top border
center_text "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Empty line
echo ""

# ASCII art logo (using heredoc for reliable formatting)
cat << 'EOF'
                 ███    ██ ███████  █████  ██████  
                 ████   ██ ██      ██   ██ ██   ██ 
                 ██ ██  ██ █████   ███████ ██████  
                 ██  ██ ██ ██      ██   ██ ██   ██ 
                 ██   ████ ███████ ██   ██ ██   ██ 
EOF

echo ""

cat << 'EOF'
            ███████ ██     ██  █████  ██████  ███    ███
            ██      ██     ██ ██   ██ ██   ██ ████  ████
            ███████ ██  █  ██ ███████ ██████  ██ ████ ██
                 ██ ██ ███ ██ ██   ██ ██   ██ ██  ██  ██
            ███████  ███ ███  ██   ██ ██   ██ ██      ██
EOF

echo ""
center_text "AI Swarm Intelligence Framework for NEAR Protocol"
echo ""

# Reset color
echo -e "\033[0m"

# Bottom border
center_text "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Warning section in yellow
echo ""
echo -e "\033[1;33m$(center_text "⚠️  TESTNET MODE - For Development Only")\033[0m"
center_text "This template runs on NEAR testnet for safe development and testing."
center_text "Do not use real funds or deploy to mainnet without thorough testing."

# Bottom border
center_text "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Function to show progress with debug support
show_progress() {
    local message="$1"
    local duration="$2"
    local width=50
    local progress=0

    # Print debug message first
    echo "DEBUG: Starting task - $message"

    # Center the progress bar
    local total_width=$(( ${#message} + width + 3 ))
    local padding=$(( (TERM_WIDTH - total_width) / 2 ))

    # Show initial progress bar
    printf "%${padding}s%s " "" "$message"
    for ((i=0; i<width; i++)); do
        echo -n "▱"
    done
    echo ""

    # Show completion
    sleep "$duration"
    printf "%${padding}s%s " "" "$message"
    for ((i=0; i<width; i++)); do
        echo -n "▰"
    done
    echo ""
}

# Section headers in NEAR blue
section_header() {
    echo ""
    echo -e "\033[38;5;75m$1\033[0m"
    echo "$(center_text "━━━━━━━━━━━━━━━━━━━━━")"
}

section_header "🔍 Checking Prerequisites"

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found. Please install Python 3 and try again."
    exit 1
fi

# Check/install near-cli
if ! command -v near &> /dev/null; then
    echo "📦 Installing NEAR CLI..."
    if ! npm install -g near-cli > /dev/null 2>&1; then
        echo "❌ Failed to install NEAR CLI. Please install Node.js and try again."
        exit 1
    fi
    echo "✅ NEAR CLI installed"
else
    python_version=$(python3 --version)
    center_text "✅ $python_version and NEAR CLI found"
fi

section_header "🛠️  Setting Up Development Environment"

# Create virtual environment
center_text "📦 Creating Python virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Existing virtual environment found. Removing..."
    rm -rf venv
fi

show_progress "Creating virtual environment" 2

# Add debug output
echo "Debug: Using Python version $(python3 --version)"
echo "Debug: Python executable path: $(which python3)"
echo "Debug: Current directory: $(pwd)"
echo "Debug: Current user: $(whoami)"

# Check Python venv module
if ! python3 -c "import venv" 2>/dev/null; then
    echo "❌ Python venv module not available. Installing..."
    sudo apt-get update && sudo apt-get install -y python3-venv
fi

# Try to create virtual environment
echo "Debug: Creating new virtual environment"
if ! python3 -m venv venv; then
    echo "❌ Failed to create virtual environment. Error details:"
    python3 -m venv --help
    python3 -m pip debug
    exit 1
fi

# Try to activate with debug output
echo "Debug: Attempting to activate virtual environment"
if ! source venv/bin/activate; then
    echo "❌ Failed to activate virtual environment. Error details:"
    ls -la venv/bin/
    exit 1
fi

# Verify activation
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Virtual environment not properly activated"
    exit 1
fi

center_text "✅ Virtual environment created and activated"

# Install dependencies
echo ""
echo "📚 Installing dependencies..."
show_progress "Installing required packages" 3

# First upgrade pip
if ! pip install --upgrade pip > /dev/null 2>&1; then
    echo "❌ Failed to upgrade pip"
    exit 1
fi

# Install packages with verbose output for debugging
if ! pip install -r requirements.txt; then
    echo "❌ Failed to install dependencies. See error messages above."
    exit 1
fi
echo "✅ Dependencies installed"

# Copy environment template if not exists
if [ ! -f .env ]; then
    echo ""
    echo "⚙️  Setting up environment configuration..."
    show_progress "Creating environment file" 1
    cp .env.example .env
    echo "✅ Environment file created"
fi

echo ""
echo "🔑 NEAR Wallet Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━"

# Function to validate wallet
validate_wallet() {
    echo "Validating wallet..."
    show_progress "Testing wallet connection" 2
    
    # Run wallet validation test
    if python -m pytest tests/core/test_near_integration.py -k test_wallet_validation -v > /dev/null 2>&1; then
        echo "✅ Wallet validated successfully"
        return 0
    else
        echo "❌ Wallet validation failed"
        return 1
    fi
}

# Check if NEAR account exists
if [ ! -f ~/.near-credentials/testnet/*.json ]; then
    echo "Creating new NEAR testnet wallet..."
    show_progress "Setting up NEAR wallet" 3
    ./scripts/create_near_wallet.sh
    
    # Validate new wallet
    echo "Waiting for wallet to be ready..."
    show_progress "Waiting for network propagation" 5
    
    # Try to validate wallet
    attempt=1
    max_attempts=5
    while [ $attempt -le $max_attempts ]; do
        if validate_wallet; then
            break
        fi
        
        if [ $attempt -lt $max_attempts ]; then
            echo "Retrying in $((attempt * 2)) seconds... (Attempt $attempt/$max_attempts)"
            sleep $((attempt * 2))
        fi
        ((attempt++))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        echo "❌ Failed to validate wallet after $max_attempts attempts"
        echo "Please try running quickstart.sh again in a few minutes"
        exit 1
    fi
else
    echo "✅ Existing NEAR testnet wallet found"
    # Update .env with existing credentials
    CREDS_FILE=$(ls ~/.near-credentials/testnet/*.json | head -n 1)
    if [ -f "$CREDS_FILE" ]; then
        ACCOUNT_ID=$(jq -r '.account_id' "$CREDS_FILE")
        PRIVATE_KEY=$(jq -r '.private_key' "$CREDS_FILE")
        
        # Update .env file
        sed -i "s|NEAR_ACCOUNT_ID=.*|NEAR_ACCOUNT_ID=$ACCOUNT_ID|" .env
        sed -i "s|NEAR_PRIVATE_KEY=.*|NEAR_PRIVATE_KEY=$PRIVATE_KEY|" .env
        sed -i "s|NEAR_NODE_URL=.*|NEAR_NODE_URL=https://rpc.testnet.fastnear.com|" .env
        echo "✅ Credentials updated in environment file"
        
        # Validate existing wallet
        validate_wallet
    fi
fi

echo ""
echo "🧪 Running Tests"
echo "━━━━━━━━━━━━━"
show_progress "Running test suite" 2
python -m pytest tests/ > /dev/null 2>&1
echo "✅ All tests passed"

echo ""
echo "📝 Creating Example Strategy"
echo "━━━━━━━━━━━━━━━━━━━━━━"
show_progress "Initializing example strategy" 2
./scripts/near-swarm init arbitrage --name example-strategy > /dev/null 2>&1
echo "✅ Example strategy created"

echo ""
section_header "🚀 Launching Default Swarm Scenario"
echo "Executing sample transaction with swarm agent..."
show_progress "Initializing swarm agent" 2

# Run the simple strategy example
if python3 -c "
import asyncio
from near_swarm.examples.simple_strategy import run_simple_strategy
try:
    asyncio.run(run_simple_strategy())
    print('✅ Sample transaction executed successfully')
except Exception as e:
    print(f'❌ Failed to execute sample transaction: {str(e)}')
    exit(1)
"; then
    echo "✅ Default swarm scenario completed"
else
    echo "❌ Failed to run default scenario"
    echo "Please check the logs for more details"
    exit 1
fi

echo ""
echo "🎉 Setup Complete!"
echo "━━━━━━━━━━━━━━"
echo "Your development environment is ready:"
echo "✅ Python virtual environment"
echo "✅ All dependencies installed"
echo "✅ NEAR testnet wallet configured"
echo "✅ Example strategy created"
echo "✅ Default scenario executed"
echo ""
echo "⚠️  Remember: This is a testnet environment for development"
echo "   Do not use real funds or deploy to mainnet without thorough testing"
echo ""

# Interactive CLI tutorial
echo "Would you like to try the NEAR Swarm CLI? (y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    clear
    echo "🚀 Welcome to the NEAR Swarm CLI Tutorial"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Let's create and run a simple arbitrage strategy."
    echo ""
    echo "Press Enter to continue..."
    read -r
    
    # Show available commands
    echo "1️⃣ First, let's see what commands are available:"
    echo ""
    ./scripts/near-swarm --help
    echo ""
    echo "Press Enter to continue..."
    read -r
    
    # Create new strategy
    echo "2️⃣ Let's create a new arbitrage strategy:"
    echo ""
    ./scripts/near-swarm init arbitrage --name my-first-strategy
    echo ""
    echo "Press Enter to continue..."
    read -r
    
    # Show strategy files
    echo "3️⃣ Let's look at what was created:"
    echo ""
    ls -la my-first-strategy
    echo ""
    echo "Press Enter to continue..."
    read -r
    
    # List strategies
    echo "4️⃣ Now let's list all available strategies:"
    echo ""
    ./scripts/near-swarm list
    echo ""
    echo "Press Enter to continue..."
    read -r
    
    # Show how to run
    echo "5️⃣ To run your strategy, use these commands:"
    echo ""
    echo "cd my-first-strategy"
    echo "./scripts/near-swarm run"
    echo ""
    echo "That's it! You're ready to start developing your own strategies."
else
    echo ""
    echo "📋 Quick Reference:"
    echo "1. Create strategy:  ./scripts/near-swarm init arbitrage --name my-strategy"
    echo "2. List strategies:  ./scripts/near-swarm list"
    echo "3. Run strategy:     ./scripts/near-swarm run"
    echo "4. Monitor status:   ./scripts/near-swarm monitor"
    echo ""
fi

echo "📚 Resources:"
echo "- Documentation: docs/"
echo "- Examples: examples/"
echo "- Support: https://discord.gg/near"               