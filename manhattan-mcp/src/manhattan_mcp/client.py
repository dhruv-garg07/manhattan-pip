"""
HTTP Client for Manhattan API.

Provides async functions to communicate with the remote Manhattan memory API.
"""

import json
from typing import Any, Dict, Optional

import httpx

from manhattan_mcp.config import get_config


async def call_api(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make a POST request to the Manhattan API.
    
    Args:
        endpoint: API endpoint (e.g., 'search_memory', 'add_memory')
        payload: Request payload as dictionary
        
    Returns:
        Response data as dictionary
    """
    config = get_config()
    url = f"{config.api_url}/{endpoint}"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}"
    }
    
    async with httpx.AsyncClient(timeout=config.timeout, follow_redirects=True) as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {
                "ok": False,
                "error": f"HTTP {e.response.status_code}: {e.response.text}"
            }
        except httpx.RequestError as e:
            return {
                "ok": False,
                "error": f"Request failed: {str(e)}"
            }
        except Exception as e:
            return {
                "ok": False,
                "error": str(e)
            }


async def call_api_get(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Make a GET request to the Manhattan API.
    
    Args:
        endpoint: API endpoint
        params: Query parameters
        
    Returns:
        Response data as dictionary
    """
    config = get_config()
    url = f"{config.api_url}/{endpoint}"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}"
    }
    
    async with httpx.AsyncClient(timeout=config.timeout) as client:
        try:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {
                "ok": False,
                "error": f"HTTP {e.response.status_code}: {e.response.text}"
            }
        except httpx.RequestError as e:
            return {
                "ok": False,
                "error": f"Request failed: {str(e)}"
            }
        except Exception as e:
            return {
                "ok": False,
                "error": str(e)
            }
