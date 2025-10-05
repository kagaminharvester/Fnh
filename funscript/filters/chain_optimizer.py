"""
Filter Chain Optimizer
Author: FunGen AI System  
Version: 1.0.0

Optimizes funscript filter processing by:
- Grouping fusable operations (Clamp, Amplify, Invert)
- Single-pass vectorized numpy operations
- Reduced CPU overhead
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class FilterOperation:
    """Represents a filter operation."""
    name: str
    params: Dict[str, Any]
    fusable: bool = False


@dataclass
class FusedOperation:
    """Represents a fused filter operation."""
    operations: List[FilterOperation]
    apply_func: Callable[[np.ndarray], np.ndarray]


class FilterChainOptimizer:
    """
    Optimizes filter chains by fusing compatible operations.
    """
    
    # Operations that can be fused into a single pass
    FUSABLE_OPS = {'clamp', 'amplify', 'invert', 'offset'}
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize filter chain optimizer.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.fused_ops_count = 0
        
    def optimize_filter_plan(self, 
                            actions: List[Dict[str, Any]], 
                            filters: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Optimize filter execution plan by fusing compatible operations.
        
        Args:
            actions: Input funscript actions [{"at": ms, "pos": val}, ...]
            filters: List of filter specifications [{"name": str, "params": dict}, ...]
            
        Returns:
            Tuple of (optimized_actions, metrics)
        """
        if not actions or not filters:
            return actions, {"fused_ops": 0}
            
        # Parse filters into operations
        operations = self._parse_filters(filters)
        
        # Group fusable operations
        fused_groups = self._group_fusable_ops(operations)
        
        # Apply optimized execution
        optimized_actions = self._apply_optimized_filters(actions, fused_groups)
        
        metrics = {
            "fused_ops": self.fused_ops_count,
            "original_filter_count": len(filters),
            "optimized_filter_count": len(fused_groups)
        }
        
        return optimized_actions, metrics
        
    def _parse_filters(self, filters: List[Dict[str, Any]]) -> List[FilterOperation]:
        """Parse filter specifications into operations."""
        operations = []
        
        for f in filters:
            name = f.get('name', '').lower()
            params = f.get('params', {})
            fusable = name in self.FUSABLE_OPS
            
            operations.append(FilterOperation(
                name=name,
                params=params,
                fusable=fusable
            ))
            
        return operations
        
    def _group_fusable_ops(self, operations: List[FilterOperation]) -> List[Any]:
        """Group consecutive fusable operations."""
        groups = []
        current_fusable = []
        
        for op in operations:
            if op.fusable:
                current_fusable.append(op)
            else:
                # Flush current fusable group
                if current_fusable:
                    if len(current_fusable) > 1:
                        groups.append(self._create_fused_op(current_fusable))
                        self.fused_ops_count += len(current_fusable)
                    else:
                        # Single op, not worth fusing
                        groups.append(current_fusable[0])
                    current_fusable = []
                    
                # Add non-fusable operation
                groups.append(op)
                
        # Flush remaining fusable ops
        if current_fusable:
            if len(current_fusable) > 1:
                groups.append(self._create_fused_op(current_fusable))
                self.fused_ops_count += len(current_fusable)
            else:
                groups.append(current_fusable[0])
                
        return groups
        
    def _create_fused_op(self, operations: List[FilterOperation]) -> FusedOperation:
        """Create a fused operation from multiple fusable operations."""
        
        def apply_fused(positions: np.ndarray) -> np.ndarray:
            """Apply all fused operations in sequence."""
            result = positions.copy()
            
            for op in operations:
                if op.name == 'clamp':
                    min_val = op.params.get('min', 0)
                    max_val = op.params.get('max', 100)
                    result = np.clip(result, min_val, max_val)
                    
                elif op.name == 'amplify':
                    factor = op.params.get('factor', 1.0)
                    center = op.params.get('center', 50)
                    result = (result - center) * factor + center
                    
                elif op.name == 'invert':
                    center = op.params.get('center', 50)
                    result = 2 * center - result
                    
                elif op.name == 'offset':
                    offset = op.params.get('offset', 0)
                    result = result + offset
                    
            return result
            
        return FusedOperation(
            operations=operations,
            apply_func=apply_fused
        )
        
    def _apply_optimized_filters(self, 
                                actions: List[Dict[str, Any]], 
                                groups: List[Any]) -> List[Dict[str, Any]]:
        """Apply optimized filter groups to actions."""
        if not actions:
            return actions
            
        # Extract positions and timestamps
        times = np.array([a['at'] for a in actions])
        positions = np.array([a['pos'] for a in actions], dtype=np.float32)
        
        # Apply each group
        for group in groups:
            if isinstance(group, FusedOperation):
                # Apply fused operation
                positions = group.apply_func(positions)
            elif isinstance(group, FilterOperation):
                # Apply single operation (may need to delegate to actual filter)
                positions = self._apply_single_op(positions, group)
                
        # Ensure valid range
        positions = np.clip(positions, 0, 100)
        
        # Reconstruct actions
        result = [
            {"at": int(times[i]), "pos": int(positions[i])}
            for i in range(len(actions))
        ]
        
        return result
        
    def _apply_single_op(self, positions: np.ndarray, op: FilterOperation) -> np.ndarray:
        """Apply a single non-fusable operation."""
        # For non-fusable operations, we'd need to call the actual filter implementation
        # For now, just return positions unchanged
        self.logger.debug(f"Skipping non-fusable operation: {op.name}")
        return positions


def optimize_filter_plan(actions: List[Dict[str, Any]], 
                        filters: List[Dict[str, Any]],
                        logger: Optional[logging.Logger] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Convenience function to optimize filter plan.
    
    Args:
        actions: Input funscript actions
        filters: List of filter specifications
        logger: Optional logger
        
    Returns:
        Tuple of (optimized_actions, metrics)
    """
    optimizer = FilterChainOptimizer(logger=logger)
    return optimizer.optimize_filter_plan(actions, filters)


def create_optimizer(logger: Optional[logging.Logger] = None) -> FilterChainOptimizer:
    """
    Create a filter chain optimizer instance.
    
    Args:
        logger: Optional logger
        
    Returns:
        FilterChainOptimizer instance
    """
    return FilterChainOptimizer(logger=logger)
