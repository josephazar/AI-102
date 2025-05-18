
import json
import os
from typing import Dict, Any

CONFIG_FILE = "config.json"

def load_config() -> Dict[str, Any]:
    """Load configuration from the config file."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to the config file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

def update_config(key: str, value: Any) -> None:
    """Update a specific configuration value."""
    config = load_config()
    config[key] = value
    save_config(config)

def get_config(key: str, default=None) -> Any:
    """Get a specific configuration value."""
    config = load_config()
    return config.get(key, default)