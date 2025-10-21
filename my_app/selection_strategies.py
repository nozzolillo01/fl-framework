"""Simple client selection strategies for Federated Learning.

All strategies follow the same signature:
    def select_xxx(available_nodes, fleet_manager, params) -> (selected_nodes, prob_map)

Where:
    - available_nodes: list of node IDs (integers)
    - fleet_manager: FleetManager instance for battery information
    - params: dict with configuration parameters
    
Returns:
    - selected_nodes: list of selected node IDs (integers)
    - prob_map: dict mapping node_id (integer) -> selection probability (for logging)
"""

from typing import Callable, Optional
from my_app.battery_simulator import FleetManager
import random
import numpy as np
from flwr.common import log
from logging import INFO

def select_random(available_nodes: list[int], fleet_manager: FleetManager, params: dict) -> tuple[list[int], dict[int, float]]:
    """Random selection without battery awareness.
    
    Selects clients uniformly at random based on sample_fraction.
    """
    if not available_nodes:
        return [], {}
    
    sample_fraction = params.get("selection-fraction")
    num_to_select = max(1, int(len(available_nodes) * sample_fraction))
    
    selected = random.sample(available_nodes, num_to_select)
    
    # Uniform probability for all clients
    prob_map = {node_id: 1.0 / len(available_nodes) for node_id in available_nodes}

    return selected, prob_map

def select_battery_aware(available_nodes: list[int], fleet_manager: FleetManager, params: dict) -> tuple[list[int], dict[int, float]]:
    """Battery-weighted client selection.
    
    Selects clients with probability proportional to battery^alpha.
    Only clients with battery >= min_battery_threshold are eligible for the selection.
    """
    if not available_nodes:
        return [], {}

    sample_fraction = params.get("selection-fraction")
    alpha = params.get("alpha")
    min_battery_threshold = params.get("min-battery-threshold")

    # Filter clients below battery threshold
    eligible_node_ids = fleet_manager.get_clients_above_threshold(available_nodes, min_battery_threshold)

    # Fallback: if no eligible clients, select randomly
    if not eligible_node_ids:
        selected = random.sample(available_nodes, max(1, int(len(available_nodes) * sample_fraction)))
        prob_map = {node_id: 1.0 / len(available_nodes) for node_id in available_nodes}
        return selected, prob_map
    
    # Calculate battery-based weights: weight_i = battery_i^alpha
    weights_map = fleet_manager.calculate_selection_weights(eligible_node_ids, alpha)
    weights = np.array(
        [weights_map.get(node_id, 0.0) for node_id in eligible_node_ids], 
        dtype=float
    )
    
    # Ensure valid weights
    if weights.sum() <= 0:
        weights = np.ones(len(eligible_node_ids), dtype=float)
    
    # Normalize to probabilities
    probabilities = weights / weights.sum()
    
    # Determine number of clients to select
    num_to_select = max(1, int(len(available_nodes) * sample_fraction))
    num_to_select = min(num_to_select, len(eligible_node_ids))
    
    # Weighted random sampling without replacement
    indices = np.random.choice(
        len(eligible_node_ids), 
        size=num_to_select, 
        replace=False, 
        p=probabilities
    )
    selected = [eligible_node_ids[i] for i in indices]
    
    # Build probability map for all available clients
    prob_map = {node_id: 0.0 for node_id in available_nodes}
    for node_id, prob in zip(eligible_node_ids, probabilities):
        prob_map[node_id] = float(prob)
    
    return selected, prob_map

def select_all_available(available_nodes: list[int], fleet_manager: FleetManager, params: dict) -> tuple[list[int], dict[int, float]]:
    """Select all available clients.
    
    No sampling - all clients participate in every round.
    Useful for experiments with full participation.
    """
    prob_map = {node_id: 1.0 for node_id in available_nodes}
    return available_nodes, prob_map


# Dictionary of available strategies
STRATEGIES = {
    "random": select_random,
    "battery_aware": select_battery_aware,
    "all_available": select_all_available,
}


def get_selection_strategy(name: str) -> Optional[Callable[[list[int], FleetManager, dict], tuple[list[int], dict[int, float]]]]:
    """Get a selection strategy function by name.
    
    Args:
        name: Name of the strategy ("random", "battery_aware", "all_available")
        
    Returns:
        Function with signature: (available_nodes, fleet_manager, params) -> (selected_nodes, prob_map)
        
    Raises:
        ValueError: If strategy name is not found
    """
    if name not in STRATEGIES:
        available = list(STRATEGIES.keys())
        raise ValueError(
            f"Unknown selection strategy '{name}'. "
            f"Available strategies: {available}"
        )
    return STRATEGIES[name]
