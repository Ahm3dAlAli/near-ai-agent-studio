"""
NEAR Protocol Integration Module
Using near-api-py to handle transactions more succinctly.
"""

import logging
import os
from typing import Dict, Any, Optional
import asyncio

try:
    import near_api
    from near_api.providers import JsonProvider
    from near_api.signer import KeyPair, Signer
    from near_api.account import Account
except ImportError:
    raise ImportError("Please install near-api-py: pip install near-api-py")

from pydantic import BaseModel, Field
from aiohttp.client_exceptions import ClientError

logger = logging.getLogger(__name__)

class NEARError(Exception):
    pass

class NEARConnectionError(NEARError):
    pass

class NEARRPCError(NEARError):
    pass

class NEARConfig(BaseModel):
    """NEAR connection configuration."""
    network: str = Field(..., description="Network to connect to (testnet/mainnet)")
    account_id: str = Field(..., description="NEAR account ID")
    private_key: str = Field(..., description="Account private key, e.g. ed25519:...")
    node_url: Optional[str] = Field(None, description="Custom RPC endpoint")
    use_backup: bool = Field(False, description="Use backup RPC endpoints")

class NEARConnection:
    """
    NEAR connection handler using near-api-py
    """
    def __init__(self,
                 network: str,
                 account_id: str,
                 private_key: str,
                 node_url: Optional[str] = None,
                 use_backup: bool = False):
        if not network or not account_id or not private_key:
            raise ValueError("network, account_id, and private_key are required")
        self.network = network.lower()
        self.account_id = account_id
        self.private_key = private_key
        self.node_url = node_url or "https://test.rpc.fastnear.com"
        self.use_backup = use_backup

        try:
            # Create the NEAR JSON-RPC provider
            self.provider = JsonProvider(self.node_url)
            # Create KeyPair from private key
            self.key_pair = KeyPair(self.private_key)
            # Create Signer
            self.signer = Signer(self.account_id, self.key_pair)
            # Create Account handle
            self.account = Account(self.provider, self.signer, self.account_id)

            logger.info(f"Initialized NEAR connection using {self.node_url}")
        except Exception as e:
            logger.error(f"Failed to initialize NEAR connection: {str(e)}")
            raise NEARConnectionError(f"Failed to initialize connection: {str(e)}")

    async def check_account(self, account_id: str) -> bool:
        """Check if account exists."""
        try:
            query_response = self.provider.query({
                "request_type": "view_account",
                "finality": "final",
                "account_id": account_id
            })
            return True
        except Exception as e:
            logger.warning(f"Account check failed: {account_id}, reason: {str(e)}")
            return False

    async def get_account_balance(self) -> Dict[str, str]:
        """Get account balance for self.account_id."""
        try:
            query_response = self.provider.query({
                "request_type": "view_account",
                "finality": "final",
                "account_id": self.account_id
            })
            total_str = query_response["amount"]
            locked_str = query_response.get("locked", "0")
            available_int = int(total_str) - int(locked_str)
            return {
                "total": total_str,
                "available": str(available_int)
            }
        except Exception as e:
            logger.error(f"Failed to get balance for {self.account_id}: {str(e)}")
            raise NEARRPCError(str(e))

    async def send_transaction(self, receiver_id: str, amount: float) -> dict:
        """
        Send a transaction to transfer NEAR tokens using near-api-py.
        """
        try:
            # Convert NEAR float to yoctoNEAR as an integer
            amount_yocto = int(amount * 1e24)

            # Use near-api-py's send_money
            outcome = self.account.send_money(receiver_id, amount_yocto)
            tx_hash = outcome.get("transaction_outcome", {}).get("id", None)
            if not tx_hash:
                raise NEARRPCError("No transaction hash returned from send_money")

            explorer_url = f"https://{self.network}.nearblocks.io/txns/{tx_hash}"
            return {
                "transaction_id": tx_hash,
                "explorer_url": explorer_url
            }

        except Exception as e:
            logger.error(f"Transaction to {receiver_id} failed: {str(e)}", exc_info=True)
            raise NEARRPCError(str(e))

    async def close(self):
        """
        near-api-py doesn't keep an open HTTP session by default,
        so this is a no-op.
        """
        pass

async def create_near_connection(config: NEARConfig) -> "NEARConnection":
    """
    Create a new NEAR connection from configuration.
    """
    try:
        conn = NEARConnection(
            network=config.network,
            account_id=config.account_id,
            private_key=config.private_key,
            node_url=config.node_url,
            use_backup=config.use_backup
        )
        # Basic check: see if we can query the account
        exists = await conn.check_account(config.account_id)
        if not exists:
            logger.warning(f"Account {config.account_id} not found on NEAR.")
        return conn
    except Exception as e:
        logger.error(f"Failed to create NEAR connection: {str(e)}")
        raise NEARConnectionError(f"Connection failed: {str(e)}")
