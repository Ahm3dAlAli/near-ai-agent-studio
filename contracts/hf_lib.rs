use near_sdk::borsh::{self, BorshDeserialize, BorshSerialize};
use near_sdk::collections::{LookupMap, UnorderedMap, Vector};
use near_sdk::json_types::{U128, ValidAccountId};
use near_sdk::serde::{Deserialize, Serialize};
use near_sdk::{env, near_bindgen, AccountId, Balance, PanicOnDefault, Promise, PromiseResult};
use near_sdk::env::is_valid_account_id;
use std::convert::TryInto;

const STRATEGY_STORAGE_KEY: &[u8] = b"s";
const INVESTOR_STORAGE_KEY: &[u8] = b"i";
const TRADE_STORAGE_KEY: &[u8] = b"t";
const PERFORMANCE_STORAGE_KEY: &[u8] = b"p";
const TOKEN_STORAGE_KEY: &[u8] = b"tk";

#[derive(BorshDeserialize, BorshSerialize, Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(crate = "near_sdk::serde")]
pub enum AssetType {
    NEAR,
    NEP141Token(AccountId), // FT Token Contract Address
}

#[derive(BorshDeserialize, BorshSerialize, Serialize, Deserialize, Clone, Debug)]
#[serde(crate = "near_sdk::serde")]
pub enum TradeAction {
    Buy,
    Sell,
    Hold,
}

#[derive(BorshDeserialize, BorshSerialize, Serialize, Deserialize, Clone, Debug)]
#[serde(crate = "near_sdk::serde")]
pub enum TradeStatus {
    Pending,
    Executed,
    Failed,
    Rejected,
}

#[derive(BorshDeserialize, BorshSerialize, Serialize, Deserialize, Clone, Debug)]
#[serde(crate = "near_sdk::serde")]
pub struct Trade {
    pub id: u64,
    pub timestamp: u64,
    pub asset: AssetType,
    pub action: TradeAction,
    pub amount: Balance,
    pub price: Balance,
    pub status: TradeStatus,
    pub transaction_hash: Option<String>,
    pub agent_confidence: u8, // 0-100 representing confidence percentage
    pub consensus_rate: u8,   // 0-100 representing consensus rate percentage
}

#[derive(BorshDeserialize, BorshSerialize, Serialize, Deserialize, Clone, Debug)]
#[serde(crate = "near_sdk::serde")]
pub struct Strategy {
    pub id: u64,
    pub name: String,
    pub description: String,
    pub risk_level: u8, // 1-5 with 5 being highest risk
    pub active: bool,
    pub min_consensus: u8, // Minimum consensus required (0-100)
    pub min_confidence: u8, // Minimum agent confidence required (0-100)
    pub trade_count: u64,
    pub success_rate: u8, // 0-100 success rate percentage
}

#[derive(BorshDeserialize, BorshSerialize, Serialize, Deserialize, Clone, Debug)]
#[serde(crate = "near_sdk::serde")]
pub struct InvestorPosition {
    pub shares: Balance,
    pub invested_amount: Balance,
    pub last_updated: u64,
}

#[derive(BorshDeserialize, BorshSerialize, Serialize, Deserialize, Clone, Debug)]
#[serde(crate = "near_sdk::serde")]
pub struct PerformanceMetric {
    pub timestamp: u64,
    pub total_value: Balance,
    pub share_price: Balance, // NAV per share
    pub token_prices: Vec<(AssetType, Balance)>, // Asset prices at this snapshot
}

#[derive(BorshDeserialize, BorshSerialize, Serialize, Deserialize, Clone, Debug)]
#[serde(crate = "near_sdk::serde")]
pub struct TokenBalance {
    pub token_id: AssetType,
    pub balance: Balance,
}

#[near_bindgen]
#[derive(BorshDeserialize, BorshSerialize, PanicOnDefault)]
pub struct AiHedgeFund {
    // Contract owner that can update strategies and parameters
    pub owner_id: AccountId,
    
    // Contract admin accounts that can execute trades
    pub admins: Vec<AccountId>,
    
    // Available strategies
    pub strategies: UnorderedMap<u64, Strategy>,
    
    // Investor shares and positions
    pub investors: LookupMap<AccountId, InvestorPosition>,
    
    // Trade history
    pub trades: Vector<Trade>,
    
    // Performance history
    pub performance: Vector<PerformanceMetric>,
    
    // Token balances
    pub token_balances: UnorderedMap<AssetType, Balance>,
    
    // Total shares issued
    pub total_shares: Balance,
    
    // Total fund value in yoctoNEAR
    pub total_fund_value: Balance,
    
    // Management fee in basis points (e.g., 100 = 1%)
    pub management_fee_bps: u16,
    
    // Performance fee in basis points (e.g., 2000 = 20%)
    pub performance_fee_bps: u16,
    
    // Last fee collection timestamp
    pub last_fee_collection: u64,
    
    // Is the fund accepting new deposits
    pub deposits_enabled: bool,
    
    // Is the fund accepting withdrawals
    pub withdrawals_enabled: bool,
    
    // Minimum deposit amount in yoctoNEAR
    pub min_deposit: Balance,
    
    // Maximum number of investors
    pub max_investors: u32,
    
    // Active strategy ID
    pub active_strategy_id: Option<u64>,
}

#[near_bindgen]
impl AiHedgeFund {
    /// Initialize the hedge fund contract
    #[init]
    pub fn new(owner_id: AccountId) -> Self {
        assert!(is_valid_account_id(owner_id.as_bytes()), "Invalid owner account ID");
        
        Self {
            owner_id: owner_id.clone(),
            admins: vec![owner_id],
            strategies: UnorderedMap::new(STRATEGY_STORAGE_KEY),
            investors: LookupMap::new(INVESTOR_STORAGE_KEY),
            trades: Vector::new(TRADE_STORAGE_KEY),
            performance: Vector::new(PERFORMANCE_STORAGE_KEY),
            token_balances: UnorderedMap::new(TOKEN_STORAGE_KEY),
            total_shares: 0,
            total_fund_value: 0,
            management_fee_bps: 100, // 1% annual management fee
            performance_fee_bps: 2000, // 20% performance fee
            last_fee_collection: env::block_timestamp(),
            deposits_enabled: true,
            withdrawals_enabled: true,
            min_deposit: 1_000_000_000_000_000_000_000_000, // 1 NEAR
            max_investors: 100,
            active_strategy_id: None,
        }
    }
    
    /************************
     * Admin Functions
     ************************/
    
    /// Add a new admin account
    pub fn add_admin(&mut self, admin_account: AccountId) {
        self.assert_owner();
        assert!(is_valid_account_id(admin_account.as_bytes()), "Invalid admin account ID");
        if !self.admins.contains(&admin_account) {
            self.admins.push(admin_account);
        }
    }
    
    /// Remove an admin account
    pub fn remove_admin(&mut self, admin_account: AccountId) {
        self.assert_owner();
        self.admins.retain(|account| account != &admin_account);
    }
    
    /// Create a new investment strategy
    pub fn create_strategy(
        &mut self,
        name: String,
        description: String,
        risk_level: u8,
        min_consensus: u8,
        min_confidence: u8
    ) -> u64 {
        self.assert_admin();
        assert!(risk_level >= 1 && risk_level <= 5, "Risk level must be between 1 and 5");
        assert!(min_consensus <= 100, "Min consensus must be between 0 and 100");
        assert!(min_confidence <= 100, "Min confidence must be between 0 and 100");
        
        let id = (self.strategies.len() + 1) as u64;
        let strategy = Strategy {
            id,
            name,
            description,
            risk_level,
            active: false,
            min_consensus,
            min_confidence,
            trade_count: 0,
            success_rate: 0,
        };
        
        self.strategies.insert(&id, &strategy);
        id
    }
    
    /// Activate a strategy
    pub fn activate_strategy(&mut self, strategy_id: u64) {
        self.assert_admin();
        let mut strategy = self.strategies.get(&strategy_id).expect("Strategy not found");
        
        // Deactivate current active strategy if any
        if let Some(active_id) = self.active_strategy_id {
            if active_id != strategy_id {
                let mut active_strategy = self.strategies.get(&active_id).expect("Active strategy not found");
                active_strategy.active = false;
                self.strategies.insert(&active_id, &active_strategy);
            }
        }
        
        // Activate new strategy
        strategy.active = true;
        self.strategies.insert(&strategy_id, &strategy);
        self.active_strategy_id = Some(strategy_id);
    }
    
    /// Execute a trade
    pub fn execute_trade(
        &mut self,
        asset: AssetType,
        action: TradeAction,
        amount: U128,
        price: U128,
        agent_confidence: u8,
        consensus_rate: u8
    ) -> u64 {
        self.assert_admin();
        
        // Check if there's an active strategy
        let strategy_id = self.active_strategy_id.expect("No active strategy");
        let mut strategy = self.strategies.get(&strategy_id).expect("Strategy not found");
        
        // Validate confidence and consensus against strategy requirements
        assert!(
            agent_confidence >= strategy.min_confidence,
            "Agent confidence below minimum threshold"
        );
        assert!(
            consensus_rate >= strategy.min_consensus,
            "Consensus rate below minimum threshold"
        );
        
        let id = self.trades.len() as u64;
        let trade = Trade {
            id,
            timestamp: env::block_timestamp(),
            asset: asset.clone(),
            action: action.clone(),
            amount: amount.0,
            price: price.0,
            status: TradeStatus::Pending,
            transaction_hash: None,
            agent_confidence,
            consensus_rate,
        };
        
        self.trades.push(&trade);
        
        // Update strategy stats
        strategy.trade_count += 1;
        self.strategies.insert(&strategy_id, &strategy);
        
        // Process trade based on action
        match action {
            TradeAction::Buy => {
                // Implement buy logic
                // For simplicity, we'll just update the token balance
                let current_balance = self.token_balances.get(&asset).unwrap_or(0);
                self.token_balances.insert(&asset, &(current_balance + amount.0));
            },
            TradeAction::Sell => {
                // Implement sell logic
                // For simplicity, we'll just update the token balance
                let current_balance = self.token_balances.get(&asset).unwrap_or(0);
                assert!(current_balance >= amount.0, "Insufficient token balance");
                self.token_balances.insert(&asset, &(current_balance - amount.0));
            },
            TradeAction::Hold => {
                // No action needed for hold
            },
        }
        
        // Update trade status
        let mut updated_trade = trade.clone();
        updated_trade.status = TradeStatus::Executed;
        updated_trade.transaction_hash = Some("simulated_transaction_hash".to_string());
        self.trades.replace(id as u64, &updated_trade);
        
        // Update fund value
        self.update_fund_value();
        
        id
    }
    
    /// Update fund parameters
    pub fn update_fund_parameters(
        &mut self,
        management_fee_bps: Option<u16>,
        performance_fee_bps: Option<u16>,
        deposits_enabled: Option<bool>,
        withdrawals_enabled: Option<bool>,
        min_deposit: Option<U128>,
        max_investors: Option<u32>
    ) {
        self.assert_owner();
        
        if let Some(fee) = management_fee_bps {
            assert!(fee <= 500, "Management fee cannot exceed 5%");
            self.management_fee_bps = fee;
        }
        
        if let Some(fee) = performance_fee_bps {
            assert!(fee <= 3000, "Performance fee cannot exceed 30%");
            self.performance_fee_bps = fee;
        }
        
        if let Some(enabled) = deposits_enabled {
            self.deposits_enabled = enabled;
        }
        
        if let Some(enabled) = withdrawals_enabled {
            self.withdrawals_enabled = enabled;
        }
        
        if let Some(min) = min_deposit {
            self.min_deposit = min.0;
        }
        
        if let Some(max) = max_investors {
            self.max_investors = max;
        }
    }
    
    /// Record a new performance metric
    pub fn record_performance(&mut self, token_prices: Vec<(AssetType, U128)>) {
        self.assert_admin();
        
        let token_prices_internal: Vec<(AssetType, Balance)> = token_prices
            .into_iter()
            .map(|(asset, price)| (asset, price.0))
            .collect();
        
        // Calculate total fund value
        self.update_fund_value();
        
        // Calculate share price
        let share_price = if self.total_shares > 0 {
            self.total_fund_value / self.total_shares
        } else {
            0
        };
        
        let metric = PerformanceMetric {
            timestamp: env::block_timestamp(),
            total_value: self.total_fund_value,
            share_price,
            token_prices: token_prices_internal,
        };
        
        self.performance.push(&metric);
    }
    
    /// Collect management and performance fees
    pub fn collect_fees(&mut self) {
        self.assert_admin();
        
        // Calculate time since last fee collection
        let current_time = env::block_timestamp();
        let time_passed = current_time - self.last_fee_collection;
        
        // Only collect fees if at least a day has passed
        if time_passed < 86_400_000_000_000 { // 1 day in nanoseconds
            return;
        }
        
        // Collect management fee (annualized, pro-rated for time passed)
        let seconds_in_year = 31_536_000_000_000_000;
        let mgmt_fee = self.total_fund_value * self.management_fee_bps as u128 * time_passed as u128 
                        / (seconds_in_year * 10_000);
        
        // TODO: Implement performance fee collection
        // Would require tracking previous high watermark
        
        if mgmt_fee > 0 && self.total_fund_value > mgmt_fee {
            // Deduct fees from fund value
            self.total_fund_value -= mgmt_fee;
            
            // Send fees to owner (in a real contract, would use Promise)
            // Promise::new(self.owner_id.clone()).transfer(mgmt_fee);
            
            // Update last fee collection timestamp
            self.last_fee_collection = current_time;
        }
    }
    
    /************************
     * Investor Functions
     ************************/
    
    /// Deposit funds into the hedge fund
    #[payable]
    pub fn deposit(&mut self) {
        let deposit_amount = env::attached_deposit();
        let account_id = env::predecessor_account_id();
        
        // Validate deposit
        assert!(self.deposits_enabled, "Deposits are currently disabled");
        assert!(deposit_amount >= self.min_deposit, "Deposit amount below minimum");
        
        // Check investor limit
        if !self.investors.contains_key(&account_id) {
            assert!(
                self.investors.len() < self.max_investors as u64,
                "Maximum number of investors reached"
            );
        }
        
        // Calculate shares to issue
        let shares_to_issue = if self.total_shares == 0 {
            // First deposit, set initial share price to 1:1
            deposit_amount
        } else {
            // Calculate based on current NAV
            deposit_amount * self.total_shares / self.total_fund_value
        };
        
        // Update investor position
        let mut position = self.investors.get(&account_id).unwrap_or(InvestorPosition {
            shares: 0,
            invested_amount: 0,
            last_updated: env::block_timestamp(),
        });
        
        position.shares += shares_to_issue;
        position.invested_amount += deposit_amount;
        position.last_updated = env::block_timestamp();
        
        self.investors.insert(&account_id, &position);
        
        // Update fund totals
        self.total_shares += shares_to_issue;
        self.total_fund_value += deposit_amount;
        
        // Update NEAR token balance
        let near_asset = AssetType::NEAR;
        let current_balance = self.token_balances.get(&near_asset).unwrap_or(0);
        self.token_balances.insert(&near_asset, &(current_balance + deposit_amount));
        
        // Record performance after deposit
        self.record_internal_performance();
    }
    
    /// Withdraw funds from the hedge fund
    pub fn withdraw(&mut self, shares: U128) -> Promise {
        let account_id = env::predecessor_account_id();
        let shares_amount = shares.0;
        
        // Validate withdrawal
        assert!(self.withdrawals_enabled, "Withdrawals are currently disabled");
        
        // Get investor position
        let mut position = self.investors.get(&account_id).expect("No investment found");
        assert!(position.shares >= shares_amount, "Insufficient shares");
        
        // Calculate withdrawal amount
        let withdrawal_amount = shares_amount * self.total_fund_value / self.total_shares;
        
        // Update investor position
        position.shares -= shares_amount;
        if position.invested_amount > withdrawal_amount {
            position.invested_amount -= withdrawal_amount;
        } else {
            position.invested_amount = 0;
        }
        position.last_updated = env::block_timestamp();
        
        if position.shares == 0 {
            self.investors.remove(&account_id);
        } else {
            self.investors.insert(&account_id, &position);
        }
        
        // Update fund totals
        self.total_shares -= shares_amount;
        self.total_fund_value -= withdrawal_amount;
        
        // Update NEAR token balance
        let near_asset = AssetType::NEAR;
        let current_balance = self.token_balances.get(&near_asset).unwrap_or(0);
        assert!(current_balance >= withdrawal_amount, "Insufficient fund balance");
        self.token_balances.insert(&near_asset, &(current_balance - withdrawal_amount));
        
        // Record performance after withdrawal
        self.record_internal_performance();
        
        // Transfer funds to investor
        Promise::new(account_id).transfer(withdrawal_amount)
    }
    
    /************************
     * View Functions
     ************************/
    
    /// Get fund overview
    pub fn get_fund_overview(&self) -> (U128, U128, u16, u16, bool, bool) {
        (
            U128(self.total_fund_value),
            U128(self.total_shares),
            self.management_fee_bps,
            self.performance_fee_bps,
            self.deposits_enabled,
            self.withdrawals_enabled
        )
    }
    
    /// Get investor position
    pub fn get_investor_position(&self, account_id: AccountId) -> Option<InvestorPosition> {
        self.investors.get(&account_id)
    }
    
    /// Get investor's fund value
    pub fn get_investor_value(&self, account_id: AccountId) -> U128 {
        if let Some(position) = self.investors.get(&account_id) {
            if self.total_shares == 0 {
                return U128(0);
            }
            let value = position.shares * self.total_fund_value / self.total_shares;
            U128(value)
        } else {
            U128(0)
        }
    }
    
    /// Get active strategy
    pub fn get_active_strategy(&self) -> Option<Strategy> {
        if let Some(id) = self.active_strategy_id {
            self.strategies.get(&id)
        } else {
            None
        }
    }
    
    /// Get all strategies
    pub fn get_strategies(&self) -> Vec<Strategy> {
        self.strategies
            .iter()
            .map(|(_, strategy)| strategy)
            .collect()
    }
    
    /// Get recent trades
    pub fn get_recent_trades(&self, limit: u64) -> Vec<Trade> {
        let total = self.trades.len();
        let from_index = if total > limit { total - limit } else { 0 };
        
        (from_index..total)
            .map(|i| self.trades.get(i).unwrap())
            .collect()
    }
    
    /// Get performance history
    pub fn get_performance_history(&self, limit: u64) -> Vec<PerformanceMetric> {
        let total = self.performance.len();
        let from_index = if total > limit { total - limit } else { 0 };
        
        (from_index..total)
            .map(|i| self.performance.get(i).unwrap())
            .collect()
    }
    
    /// Get token balances
    pub fn get_token_balances(&self) -> Vec<TokenBalance> {
        self.token_balances
            .iter()
            .map(|(token_id, balance)| TokenBalance {
                token_id,
                balance,
            })
            .collect()
    }
    
    /************************
     * Internal Functions
     ************************/
    
    /// Ensure caller is the contract owner
    fn assert_owner(&self) {
        assert_eq!(
            env::predecessor_account_id(),
            self.owner_id,
            "Only the owner can call this method"
        );
    }
    
    /// Ensure caller is an admin
    fn assert_admin(&self) {
        let caller = env::predecessor_account_id();
        assert!(
            self.admins.contains(&caller),
            "Only admins can call this method"
        );
    }
    
    /// Update the total fund value based on current token balances and prices
    fn update_fund_value(&mut self) {
        // For simplicity, we're just using the NEAR balance for now
        // In a real-world scenario, this would fetch token prices and calculate value of all assets
        let near_balance = self.token_balances.get(&AssetType::NEAR).unwrap_or(0);
        
        // Update total fund value
        self.total_fund_value = near_balance;
    }
    
    /// Record performance internally without requiring token prices
    fn record_internal_performance(&mut self) {
        // Calculate share price
        let share_price = if self.total_shares > 0 {
            self.total_fund_value / self.total_shares
        } else {
            0
        };
        
        // Create performance metric with only NEAR price
        let near_asset = AssetType::NEAR;
        let token_prices = vec![(near_asset, 1_000_000_000_000_000_000_000_000)]; // 1 NEAR = 1 NEAR for simplicity
        
        let metric = PerformanceMetric {
            timestamp: env::block_timestamp(),
            total_value: self.total_fund_value,
            share_price,
            token_prices,
        };
        
        self.performance.push(&metric);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use near_sdk::test_utils::{accounts, VMContextBuilder};
    use near_sdk::testing_env;
    use near_sdk::MockedBlockchain;

    fn get_context(predecessor_account_id: AccountId) -> VMContextBuilder {
        let mut builder = VMContextBuilder::new();
        builder
            .current_account_id(accounts(0))
            .signer_account_id(predecessor_account_id.clone())
            .predecessor_account_id(predecessor_account_id);
        builder
    }

    #[test]
    fn test_new() {
        let context = get_context(accounts(1));
        testing_env!(context.build());
        let contract = AiHedgeFund::new(accounts(1));
        assert_eq!(contract.owner_id, accounts(1));
    }

    #[test]
    fn test_create_strategy() {
        let mut context = get_context(accounts(1));
        testing_env!(context.build());
        let mut contract = AiHedgeFund::new(accounts(1));
        
        let strategy_id = contract.create_strategy(
            "Test Strategy".to_string(),
            "A test strategy".to_string(),
            3,
            70,
            80
        );
        
        assert_eq!(strategy_id, 1);
        
        let strategy = contract.strategies.get(&strategy_id).unwrap();
        assert_eq!(strategy.name, "Test Strategy");
        assert_eq!(strategy.risk_level, 3);
        assert_eq!(strategy.min_consensus, 70);
    }

    #[test]
    fn test_deposit() {
        let mut context = get_context(accounts(1));
        context.attached_deposit(10_000_000_000_000_000_000_000_000); // 10 NEAR
        testing_env!(context.build());
        
        let mut contract = AiHedgeFund::new(accounts(1));
        contract.deposit();
        
        // Verify investor position
        let position = contract.investors.get(&accounts(1)).unwrap();
        assert_eq!(position.shares, 10_000_000_000_000_000_000_000_000);
        assert_eq!(position.invested_amount, 10_000_000_000_000_000_000_000_000);
        
        // Verify fund totals
        assert_eq!(contract.total_shares, 10_000_000_000_000_000_000_000_000);
        assert_eq!(contract.total_fund_value, 10_000_000_000_000_000_000_000_000);
    }

    #[test]
    fn test_withdraw() {
        let mut context = get_context(accounts(1));
        context.attached_deposit(10_000_000_000_000_000_000_000_000); // 10 NEAR
        testing_env!(context.build());
        
        let mut contract = AiHedgeFund::new(accounts(1));
        contract.deposit();
        
        // Change context for withdrawal
        let context = get_context(accounts(1));
        testing_env!(context.build());
        
        // Withdraw half the shares
        contract.withdraw(U128(5_000_000_000_000_000_000_000_000));
        
        // Verify investor position
        let position = contract.investors.get(&accounts(1)).unwrap();
        assert_eq!(position.shares, 5_000_000_000_000_000_000_000_000);
        assert_eq!(position.invested_amount, 5_000_000_000_000_000_000_000_000);
        
        // Verify fund totals
        assert_eq!(contract.total_shares, 5_000_000_000_000_000_000_000_000);
        assert_eq!(contract.total_fund_value, 5_000_000_000_000_000_000_000_000);
    }
}