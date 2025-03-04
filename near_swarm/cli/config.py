"""
CLI commands for configuration management
"""

import os
import click
import yaml
from typing import Optional
from pydantic import ValidationError

from ..config.loader import ConfigLoader
from ..config.schema import AgentConfig

@click.group()
def config():
    """Configuration management commands"""
    pass

@config.group()
def agent():
    """Agent configuration commands"""
    pass

@agent.command(name='configure')
@click.argument('name')
@click.option('--role', help='Agent role (e.g., market_analyzer, risk_manager)')
@click.option('--min-confidence', type=float, help='Minimum confidence threshold')
@click.option('--max-retries', type=int, help='Maximum number of retries')
def configure_agent(name: str, role: Optional[str] = None, min_confidence: Optional[float] = None, max_retries: Optional[int] = None):
    """Configure an agent"""
    try:
        # Load existing config or create new
        config_file = f"agents/custom/{name}/agent.yaml"
        config = {}
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f) or {}
        
        # Update configuration
        if role:
            config['role'] = role
        if min_confidence:
            config['min_confidence'] = min_confidence
        if max_retries:
            config['max_retries'] = max_retries
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        # Save configuration
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        click.echo(f"✅ Configured agent: {name}")
        
    except Exception as e:
        click.echo(f"❌ Error configuring agent: {str(e)}")

@config.command()
@click.argument('config_file', required=False)
def validate(config_file: Optional[str] = None):
    """Validate configuration file"""
    try:
        # Initialize config loader
        loader = ConfigLoader()
        
        # Load specific file if provided
        if config_file:
            if not os.path.exists(config_file):
                click.echo(f"Configuration file not found: {config_file}")
                return
            loader.load_config_file(config_file)
        else:
            # Try default locations
            loader.load_defaults()
            loader.load_config_file()
            loader.load_env()
        
        # Validate configuration
        try:
            config = loader.get_config()
            
            # Show validated configuration
            click.echo("Configuration is valid!")
            click.echo("\nConfiguration details:")
            
            # Format configuration for display
            config_dict = config.dict()
            click.echo(yaml.dump(config_dict, default_flow_style=False))
            
            # Show environment variables used
            env_vars = {
                k: v for k, v in os.environ.items()
                if k.startswith(loader.ENV_PREFIX)
            }
            if env_vars:
                click.echo("\nEnvironment variables:")
                for key, value in env_vars.items():
                    # Mask sensitive values
                    if 'KEY' in key or 'SECRET' in key:
                        value = '*' * 8
                    click.echo(f"  {key}: {value}")
            
        except ValidationError as e:
            click.echo("Configuration validation failed!")
            for error in e.errors():
                click.echo(f"\nError in {' -> '.join(error['loc'])}:")
                click.echo(f"  {error['msg']}")
            return
            
    except Exception as e:
        click.echo(f"Error validating configuration: {str(e)}", err=True)

@config.command()
def init():
    """Initialize default configuration"""
    try:
        config_file = "near_swarm.yaml"
        if os.path.exists(config_file):
            click.echo(f"Configuration file already exists: {config_file}")
            return
            
        # Create default configuration
        default_config = {
            "name": "my_swarm",
            "environment": "development",
            "log_level": "INFO",
            "llm": {
                "provider": "${LLM_PROVIDER}",
                "model": "${LLM_MODEL}",
                "temperature": 0.7,
                "max_tokens": 1000
            },
            "plugins": []
        }
        
        # Write configuration file
        with open(config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
            
        click.echo(f"Created default configuration: {config_file}")
        click.echo("\nNext steps:")
        click.echo("1. Set required environment variables:")
        click.echo("   export NEAR_SWARM_LLM_PROVIDER=your_provider")
        click.echo("   export NEAR_SWARM_LLM_MODEL=your_model")
        click.echo("2. Edit configuration file as needed")
        click.echo("3. Run 'near-swarm config validate' to verify")
        
    except Exception as e:
        click.echo(f"Error creating configuration: {str(e)}", err=True)

@config.command()
def show():
    """Show current configuration"""
    try:
        loader = ConfigLoader()
        loader.load_defaults()
        loader.load_config_file()
        loader.load_env()
        
        config = loader.get_config()
        config_dict = config.dict()
        
        click.echo(yaml.dump(config_dict, default_flow_style=False))
        
    except Exception as e:
        click.echo(f"Error showing configuration: {str(e)}") 