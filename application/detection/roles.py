"""
Role Mapping Layer for YOLO detection classes.

Maps raw YOLO class names to functional roles for anchor tracking.
Supports JSON configuration with fallback to defaults.
"""

import json
import logging
import os
from typing import Dict, Optional, Set

# Role constants
ROLE_PRIMARY_ANCHOR = "primary_anchor"
ROLE_SECONDARY_ANCHOR = "secondary_anchor"
ROLE_STROKER = "stroker"
ROLE_TARGET = "target"
ROLE_IGNORE = "ignore"

# Default role mappings
DEFAULT_ROLE_MAPPINGS = {
    "penis": ROLE_PRIMARY_ANCHOR,
    "glans": ROLE_SECONDARY_ANCHOR,
    "hand": ROLE_STROKER,
    "finger": ROLE_STROKER,
    "pussy": ROLE_TARGET,
    "butt": ROLE_TARGET,
    "face": ROLE_TARGET,
    "mouth": ROLE_TARGET,
    "breast": ROLE_TARGET,
    "foot": ROLE_STROKER,
    "dildo": ROLE_STROKER,
    "toy": ROLE_STROKER,
}


class RoleMapper:
    """
    Maps YOLO detection class names to functional roles.
    
    Allows for flexible configuration via JSON file, with sensible defaults
    to support future model evolution without core logic changes.
    """
    
    def __init__(self, config_path: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the role mapper.
        
        Args:
            config_path: Path to JSON configuration file. If None or file doesn't exist,
                        uses default mappings.
            logger: Optional logger instance.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.role_mappings: Dict[str, str] = {}
        
        # Try to load from config file
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_mappings = json.load(f)
                    self.role_mappings = loaded_mappings.get('role_mappings', {})
                    self.logger.info(f"Loaded role mappings from {config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load role mappings from {config_path}: {e}")
                self.role_mappings = DEFAULT_ROLE_MAPPINGS.copy()
        else:
            # Use defaults
            self.role_mappings = DEFAULT_ROLE_MAPPINGS.copy()
            if config_path:
                self.logger.debug(f"Config file not found at {config_path}, using defaults")
    
    def get_role(self, class_name: str) -> str:
        """
        Get the functional role for a YOLO class name.
        
        Args:
            class_name: YOLO detection class name
            
        Returns:
            Role string (e.g., ROLE_PRIMARY_ANCHOR, ROLE_STROKER)
            Returns ROLE_IGNORE if class is not mapped.
        """
        # Normalize class name to lowercase for case-insensitive matching
        normalized_name = class_name.lower().strip()
        return self.role_mappings.get(normalized_name, ROLE_IGNORE)
    
    def is_primary_anchor(self, class_name: str) -> bool:
        """Check if class name maps to primary anchor role."""
        return self.get_role(class_name) == ROLE_PRIMARY_ANCHOR
    
    def is_secondary_anchor(self, class_name: str) -> bool:
        """Check if class name maps to secondary anchor role."""
        return self.get_role(class_name) == ROLE_SECONDARY_ANCHOR
    
    def is_stroker(self, class_name: str) -> bool:
        """Check if class name maps to stroker role."""
        return self.get_role(class_name) == ROLE_STROKER
    
    def is_target(self, class_name: str) -> bool:
        """Check if class name maps to target role."""
        return self.get_role(class_name) == ROLE_TARGET
    
    def get_primary_anchor_classes(self) -> Set[str]:
        """Get set of all class names mapped to primary anchor role."""
        return {cls for cls, role in self.role_mappings.items() 
                if role == ROLE_PRIMARY_ANCHOR}
    
    def get_stroker_classes(self) -> Set[str]:
        """Get set of all class names mapped to stroker role."""
        return {cls for cls, role in self.role_mappings.items() 
                if role == ROLE_STROKER}
    
    def save_config(self, config_path: str) -> bool:
        """
        Save current role mappings to JSON file.
        
        Args:
            config_path: Path where to save the configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config = {
                'role_mappings': self.role_mappings,
                'description': 'YOLO class name to functional role mappings'
            }
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            self.logger.info(f"Saved role mappings to {config_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save role mappings to {config_path}: {e}")
            return False
    
    def add_mapping(self, class_name: str, role: str) -> None:
        """
        Add or update a role mapping.
        
        Args:
            class_name: YOLO detection class name
            role: Role string (should be one of the ROLE_* constants)
        """
        normalized_name = class_name.lower().strip()
        self.role_mappings[normalized_name] = role
        self.logger.debug(f"Added mapping: {normalized_name} -> {role}")
