"""Battery simulation for federated learning clients."""

import random
from typing import Optional
from flwr.common import log
from logging import INFO


class BatterySimulator:
    """Simulates battery behavior for a client device (CLIENT-SIDE).
    
    Each client instantiates its own BatterySimulator to manage local battery state.
    The simulator handles consumption during training and energy harvesting.
    """

    DEVICE_CLASSES = {
        0: { # low_power_device
            "consumption_range": (0.015, 0.025),
            "harvesting_range": (0.0, 0.015),
        },
        1: { # mid_power_device
            "consumption_range": (0.025, 0.035),
            "harvesting_range": (0.0, 0.025),
        },
        2: { # high_power_device
            "consumption_range": (0.035, 0.045),
            "harvesting_range": (0.0, 0.035),
        },
    }

    def __init__(self, client_id: int, initial_battery: float = None, device_class: int = None):
        """Initialize battery simulator for a client device.
        
        Args:
            client_id: Unique identifier for this client
            initial_battery: Initial battery level (if restoring state), otherwise random
            device_class: Type of device (low/mid/high power), deterministic based on client_id if not specified
        """
        self.client_id = client_id
        
        # Assign device class deterministically based on client_id (so it's consistent across reinitializations)
        if device_class is None:
            # Use modulo to deterministically assign device class based on client_id
            self.device_class = int(client_id % 3)
        else:
            self.device_class = device_class
        
        # Initialize battery level
        if initial_battery is not None:
            self.battery_level = initial_battery
        else:
            # Use client_id as seed for reproducible random battery initialization
            random.seed(client_id)
            self.battery_level = random.uniform(0.1, 1.0)
            random.seed()  # Reset seed to avoid affecting other random calls
        
        self.total_consumption = 0.0

        cmin, cmax = self.DEVICE_CLASSES[self.device_class]["consumption_range"]
        hmin, hmax = self.DEVICE_CLASSES[self.device_class]["harvesting_range"]


    def consume(self, local_epochs: int) -> dict:
        """Simulate training: consume battery, then recharge via energy harvesting.
        
        Args:
            local_epochs: Number of training epochs performed
            
        Returns:
            dict with battery metrics:
                - battery_level: Current battery level after training and recharge
                - previous_battery_level: Battery level before training
                - consumed: Energy consumed during training
                - recharged: Energy harvested after training
                - training_completed: Whether training completed successfully
                - device_class: Device class identifier
        """

        cmin, cmax = self.DEVICE_CLASSES[self.device_class]["consumption_range"]

        needed = random.uniform(cmin, cmax) * local_epochs

        previous_level = self.battery_level

        if self.battery_level >= needed:
            self.battery_level -= needed
            consumed = needed
            training_completed = True
        else:
            consumed = self.battery_level 
            self.battery_level = 0.0
            training_completed = False
            
        self.total_consumption += consumed
        
        return {
            "battery_level": self.battery_level,
            "previous_battery_level": previous_level,
            "consumed": consumed,
            "training_completed": training_completed,
        }

    def recharge(self) -> dict:
        """Simulate energy harvesting (passive recharge).
        
        This is called at the beginning of each round for ALL clients,
        whether they are selected for training or not.
        
        Returns:
            dict with recharge metrics:
                - battery_level: Current battery level after recharge
                - previous_battery_level: Battery level before recharge
                - recharged: Energy harvested
                - device_class: Device class identifier
        """
        hmin, hmax = self.DEVICE_CLASSES[self.device_class]["harvesting_range"]
        
        previous_level = self.battery_level
        
        # Simulate energy harvesting (using 1 time unit as base)
        harvested = random.uniform(hmin, hmax)
        
        # Add harvested energy, capped at 1.0
        self.battery_level = min(1.0, self.battery_level + harvested)
        
        return {
            "battery_level": self.battery_level,
            "previous_battery_level": previous_level,
            "recharged": harvested,
        }

class FleetManager:
    """Manages fleet-level metrics and statistics (SERVER-SIDE).
    
    The server does NOT simulate batteries - it only tracks reported metrics from clients.
    Each client manages its own battery locally and reports status to the server.
    """
    
    def __init__(self):
        """Initialize fleet manager."""
        self.all_client_battery_levels: dict[int, float] = {}
        self.all_client_device_classes: dict[int, int] = {} 
        self.all_client_participation_count: dict[int, int] = {}
        self.total_consumption_cumulative: float = 0.0
        self.round_consumed: dict[int, float] = {}
        self.round_recharged: dict[int, float] = {}
        self.all_previous_battery_levels: dict[int, float] = {}
        
        # Track client initialization to detect inconsistencies
        self.client_initialized: dict[int, bool] = {}

    def update_client_info(self, client_id: int, client_report: dict) -> None:
        """Update battery info for a specific client based on its report.
        
        Called:
        - Before round 1: only battery_level and device_class
        - After each round N: current_battery, previous_battery, consumed, recharged, battery_level
        
        Args:
            client_id: Unique identifier for the client
            client_report: Dict with reported battery metrics from the client
        """
        # Update device class
        self.all_client_device_classes[client_id] = client_report.get("device_class", -1)
        self.all_client_battery_levels[client_id] = client_report.get("battery_level", 0.0)
        
        # If this is round completion report (contains consumption info)
        if "consumed" in client_report:
            # Update previous level
            self.all_previous_battery_levels[client_id] = client_report.get("previous_battery_level", 0.0)
            
            # Update consumption
            consumed = client_report.get("consumed", 0.0)
            self.total_consumption_cumulative += consumed
            self.round_consumed[client_id] = consumed
            
            # Update recharged amount
            recharged = client_report.get("recharged", 0.0)
            self.round_recharged[client_id] = recharged

    def update_participation(self, client_ids: list[int]) -> None:
        """Track client participation counts."""
        for client_id in client_ids:
            old_count = self.all_client_participation_count.get(client_id, 0)
            self.all_client_participation_count[client_id] = old_count + 1

    def get_round_metrics(self, selected_clients: list[int], responded_clients: list[int], total_clients: list[int]) -> dict[str, float]:
        """Calculate fleet metrics for the current round.
        
        Args:
            selected_clients: Clients selected for this round
            responded_clients: Clients that successfully responded
            
        Returns:
            Dict with fleet-level metrics
        """        

        if len(self.all_previous_battery_levels) == len(total_clients):
            all_previous_battery_levels = list(self.all_previous_battery_levels.values())

        battery_min = min(all_previous_battery_levels) if all_previous_battery_levels else 0.0
        battery_max = max(all_previous_battery_levels) if all_previous_battery_levels else 0.0
        battery_avg = sum(all_previous_battery_levels) / len(all_previous_battery_levels) if all_previous_battery_levels else 0.0

        # Fairness index (Jain's fairness)
        # Calculate over ALL clients (total_clients), including those that never responded (count=0)
        counts = []
        for client in total_clients:
            if client in self.all_client_participation_count:
                counts.append(self.all_client_participation_count[client])
            else:
                counts.append(0)

        sum_x = sum(counts)
        sum_x2 = sum(c * c for c in counts)
        fairness_jain = 0.0
        if sum_x2 > 0:
            fairness_jain = (sum_x * sum_x) / (len(counts) * sum_x2)

        return {
            "selected_clients": float(len(selected_clients)),
            "dead_clients": float(len(selected_clients) - len(responded_clients)),
            "total_consumption": self.total_consumption_cumulative,
            "battery_min": battery_min,
            "battery_max": battery_max,
            "battery_avg": battery_avg,
            "fairness_index_jain": fairness_jain,
        }

    def get_client_details(
        self, 
        all_clients: list[int],
        selected_clients: list[int],
        responded_clients: list[int],
        prob_map: dict[int, float],
        min_threshold: float
    ) -> list[dict]:
        """Get detailed client information for logging.
        
        Args:
            all_clients: All clients in the system
            selected_clients: Clients selected for this round
            responded_clients: Clients that successfully responded (subset of selected_clients)
            prob_map: Selection probabilities for each client
            min_threshold: Minimum battery threshold for participation
            
        Returns:
            List of dicts with detailed client information for ALL clients
        """
        client_details = []

        # Process ALL clients (not just selected ones)
        for client_id in all_clients:
            # Get battery status
            current_battery = self.all_client_battery_levels.get(client_id, 0.0)
            previous_battery = self.all_previous_battery_levels.get(client_id, current_battery)
            consumed = self.round_consumed.get(client_id, 0.0)
            recharged = self.round_recharged.get(client_id, 0.0)
            participation_count = self.all_client_participation_count.get(client_id, 0)

            # Get device class name
            device_class_name = "unknown"
            device_class_code = self.all_client_device_classes.get(client_id, -1)
            if device_class_code == 0:
                device_class_name = "low_power_device"
            elif device_class_code == 1:
                device_class_name = "mid_power_device"
            elif device_class_code == 2:
                device_class_name = "high_power_device"

            # Determine client status
            was_selected = client_id in selected_clients
            did_respond = client_id in responded_clients
            
            # Determine completion status
            if not was_selected:
                completion_status = "not_selected"
            elif did_respond:
                completion_status = "completed"
            else:
                completion_status = "failed"

            client_details.append({
                "client_id": client_id,
                "device_class": device_class_name,
                "battery_level": current_battery,
                "previous_battery_level": previous_battery,
                "consumed": consumed,  # Only selected clients consume
                "recharged": recharged,  # ALL clients recharge during query
                "prob_selection": prob_map.get(client_id, 0.0),
                "was_selected": was_selected,
                "completion_status": completion_status,
                "was_above_threshold": previous_battery >= min_threshold,
                "participation_count": participation_count,
            })
        
        return client_details

    def calculate_selection_weights(self, client_ids: list[int], alpha: float = 2.0) -> dict[int, float]:
        """Calculate selection weights (battery^alpha) based on reported levels."""
        weights = {}
        for client_id in client_ids:
            battery_level = self.all_client_battery_levels.get(client_id, 0.0)
            weights[client_id] = battery_level ** alpha
        return weights

    def get_clients_above_threshold(self, client_ids: list[int], min_threshold: float = 0.0) -> list[int]:
        """Filter clients by battery threshold based on reported levels."""
        eligible = [
            cid for cid in client_ids
            if self.all_client_battery_levels.get(cid, 0.0) >= min_threshold
        ]
        return eligible

