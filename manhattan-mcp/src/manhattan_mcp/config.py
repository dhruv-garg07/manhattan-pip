"""
Configuration module for Manhattan MCP.

Handles environment variable loading and validation.
"""

import os
import sys
from typing import Optional

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Default configuration
DEFAULT_API_URL = "https://themanhattanproject.ai/mcp"
DEFAULT_TIMEOUT = 120.0
DEFAULT_AGENT_ID = "84aab1f8-3ea9-4c6a-aa3c-cd8eaa274a5e"


class Config:
    """Configuration class for Manhattan MCP."""
    
    def __init__(self):
        self._api_key: Optional[str] = None
        self._api_url: str = DEFAULT_API_URL
        self._timeout: float = DEFAULT_TIMEOUT
        self._default_agent_id: str = DEFAULT_AGENT_ID
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        self._api_key = os.getenv("MANHATTAN_API_KEY")
        self._api_url = os.getenv("MANHATTAN_API_URL", DEFAULT_API_URL)
        
        timeout_str = os.getenv("MANHATTAN_TIMEOUT")
        if timeout_str:
            try:
                self._timeout = float(timeout_str)
            except ValueError:
                pass
        
        agent_id = os.getenv("MANHATTAN_AGENT_ID")
        if agent_id:
            self._default_agent_id = agent_id
    
    @property
    def api_key(self) -> Optional[str]:
        """Get the API key."""
        return self._api_key
    
    @property
    def api_url(self) -> str:
        """Get the API URL."""
        return self._api_url
    
    @property
    def timeout(self) -> float:
        """Get the request timeout."""
        return self._timeout
    
    @property
    def default_agent_id(self) -> str:
        """Get the default agent ID."""
        return self._default_agent_id
    
    def validate(self) -> bool:
        """
        Validate the configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError if configuration is invalid
        """
        if not self._api_key:
            print("=" * 60, file=sys.stderr)
            print("ERROR: MANHATTAN_API_KEY environment variable is not set!", file=sys.stderr)
            print("", file=sys.stderr)
            print("Please set your API key:", file=sys.stderr)
            print("  export MANHATTAN_API_KEY='your-api-key'", file=sys.stderr)
            print("", file=sys.stderr)
            print("Get your API key at: https://themanhattanproject.ai", file=sys.stderr)
            print("=" * 60, file=sys.stderr)
            raise ValueError("MANHATTAN_API_KEY is required")
        
        return True


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns:
        Config instance with loaded settings
    """
    global _config
    if _config is None:
        _config = Config()
    return _config
