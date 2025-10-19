"""Battery simulation for federated learning clients."""

import random
from typing import Optional


class BatterySimulator:
    """Simulates battery behavior for a client device (CLIENT-SIDE).
    
    Each client instantiates its own BatterySimulator to manage local battery state.
    The simulator handles consumption during training and energy harvesting.
    """

    DEVICE_CLASSES = {
        0: { # low_power_device
            "consumption_range": (0.015, 0.025),
            "harvesting_range": (0.0, 0.010),
        },
        1: { # mid_power_device
            "consumption_range": (0.030, 0.040),
            "harvesting_range": (0.0, 0.025),
        },
        2: { # high_power_device
            "consumption_range": (0.050, 0.070),
            "harvesting_range": (0.0, 0.045),
        },
    }

    def __init__(self, client_id: int, device_class: Optional[str] = None):
        """Initialize battery simulator for a client device.
        
        Args:
            client_id: Unique identifier for this client
            device_class: Type of device (low/mid/high power), random if not specified
        """
        self.client_id = client_id
        self.battery_level = random.uniform(0.1, 1.0)
        self.total_consumption = 0.0

        if device_class not in self.DEVICE_CLASSES:
            device_class = random.choice(list(self.DEVICE_CLASSES.keys()))
        self.device_class = device_class
        cmin, cmax = self.DEVICE_CLASSES[self.device_class]["consumption_range"]
        hmin, hmax = self.DEVICE_CLASSES[self.device_class]["harvesting_range"]

        self.consumption_per_epoch = random.uniform(cmin, cmax)
        self.harvesting_per_epoch = random.uniform(hmin, hmax)

    def can_train(self, local_epochs: int) -> bool:
        """Check if client has enough battery to complete training.
        
        Args:
            local_epochs: Number of training epochs
            
        Returns:
            True if battery is sufficient, False otherwise
        """
        epochs = max(1, int(local_epochs))
        needed = self.consumption_per_epoch * epochs
        return self.battery_level >= needed

    def update(self, local_epochs: int) -> dict:
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
        epochs = max(1, int(local_epochs))
        previous_level = self.battery_level
        
        # 1. Consume battery for training
        needed = self.consumption_per_epoch * epochs

        if self.battery_level >= needed:
            self.battery_level -= needed
            consumed = needed
        else:
            # Client dies during training
            self.battery_level = 0.0
            consumed = self.battery_level
            
        self.total_consumption += consumed
        
        # 2. Recharge (energy harvesting)
        harvested = self.harvesting_per_epoch * epochs
        self.battery_level = min(1.0, self.battery_level + harvested)
        
        return {
            "device_class": self.device_class,
            "battery_level": self.battery_level,
            "previous_battery_level": previous_level,
            "consumed": consumed,
            "recharged": harvested,
        }



class FleetManager:
    """Manages fleet-level metrics and statistics (SERVER-SIDE).
    
    The server does NOT simulate batteries - it only tracks reported metrics from clients.
    Each client manages its own battery locally and reports status to the server.
    """
    
    def __init__(self):
        """Initialize fleet manager."""
        self.client_battery_levels: dict[int, float] = {}  # Latest reported levels
        self.client_device_classes: dict[int, str] = {}
        self.client_participation_count: dict[int, int] = {}
        self.total_consumption_cumulative: float = 0.0
        
        # Per-round tracking
        self.round_consumed: dict[int, float] = {}
        self.round_recharged: dict[int, float] = {}
        self.round_previous_levels: dict[int, float] = {}

    def update_client_status(self, client_id: int, battery_metrics: dict) -> None:
        """Update client status based on reported metrics from ClientApp."""
        self.client_device_classes[client_id] = battery_metrics.get("device_class", "unknown")
        self.client_battery_levels[client_id] = battery_metrics.get("battery_level", 0.0)
        self.round_consumed[client_id] = battery_metrics.get("consumed", 0.0)
        self.round_recharged[client_id] = battery_metrics.get("recharged", 0.0)
        self.round_previous_levels[client_id] = battery_metrics.get("previous_battery_level", 0.0)        
        # Update cumulative consumption
        self.total_consumption_cumulative += battery_metrics.get("consumed", 0.0)

    def update_participation(self, client_ids: list[int]) -> None:
        """Track client participation counts."""
        for client_id in client_ids:
            self.client_participation_count[client_id] = \
                self.client_participation_count.get(client_id, 0) + 1

    def get_battery_level(self, client_id: int) -> float:
        """Get reported battery level for a client (default 0.5 if not yet reported)."""
        return self.client_battery_levels.get(client_id, 0.5)

    def get_clients_above_threshold(self, client_ids: list[int], min_threshold: float = 0.0) -> list[int]:
        """Filter clients by battery threshold based on reported levels."""
        return [
            cid for cid in client_ids
            if self.get_battery_level(cid) >= min_threshold
        ]

    def calculate_selection_weights(self, client_ids: list[int], alpha: float = 2.0) -> dict[int, float]:
        """Calculate selection weights (battery^alpha) based on reported levels."""
        weights = {}
        for client_id in client_ids:
            battery_level = self.get_battery_level(client_id)
            weights[client_id] = battery_level ** alpha
        return weights

    def get_round_metrics(self, selected_clients: list[int], responded_clients: list[int], total_clients: int) -> dict[str, float]:
        """Calculate fleet metrics for the current round.
        
        Args:
            selected_clients: Clients selected for this round
            responded_clients: Clients that successfully responded
            
        Returns:
            Dict with fleet-level metrics
        """
        # Battery stats for all clients alive this round
        all_battery_levels = [
            self.client_battery_levels.get(cid, 0.0)
            for cid in responded_clients
            if cid in self.client_battery_levels
        ]

        battery_min = min(all_battery_levels) if all_battery_levels else 0.0
        battery_max = max(all_battery_levels) if all_battery_levels else 0.0
        battery_avg = sum(all_battery_levels) / len(all_battery_levels) if all_battery_levels else 0.0
        
        # Fairness index (Jain's fairness)
        # Calcola su TUTTI i client (total_clients), inclusi quelli che non hanno mai risposto (count=0)
        counts = []
        for i in range(total_clients):
            # Se il client ha risposto almeno una volta, usa il suo count, altrimenti 0
            if i in self.client_participation_count:
                counts.append(self.client_participation_count[i])
            else:
                counts.append(0)
        
        sum_x = sum(counts)
        sum_x2 = sum(c * c for c in counts)
        fairness_jain = 0.0
        if total_clients > 0 and sum_x2 > 0:
            fairness_jain = (sum_x * sum_x) / (total_clients * sum_x2)
        
        return {
            "selected_clients": float(len(selected_clients)),
            "responded_clients": float(len(responded_clients)),
            "total_consumption": self.total_consumption_cumulative,
            "battery_min": battery_min,
            "battery_max": battery_max,
            "battery_avg": battery_avg,
            "fairness_index_jain": fairness_jain,
        }

    def get_client_details(
        self, 
        selected_clients: list[int],
        responded_clients: list[int],
        prob_map: dict[int, float],
        min_threshold: float
    ) -> list[dict]:
        """Get detailed client information for logging.
        
        Args:
            selected_clients: Clients selected for this round
            responded_clients: Clients that successfully responded (subset of selected_clients)
            prob_map: Selection probabilities for each client
            min_threshold: Minimum battery threshold for participation
            
        Returns:
            List of dicts with detailed client information (only for clients we have data for)
        """
        client_details = []

        for client_id in responded_clients:
            current_battery = self.get_battery_level(client_id)
            previous_battery = self.round_previous_levels.get(client_id, current_battery)
            device_class = "unknown"
            device_class_code = self.client_device_classes.get(client_id, "unknown")
            if device_class_code == 0:
                device_class = "low_power_device"
            elif device_class_code == 1:
                device_class = "mid_power_device"
            elif device_class_code == 2:
                device_class = "high_power_device"

            client_details.append({
                "client_id": client_id,
                "device_class": device_class,
                "current_battery_level": current_battery,
                "previous_battery_level": previous_battery,
                "consumed_battery": self.round_consumed.get(client_id, 0.0),
                "recharged_battery": self.round_recharged.get(client_id, 0.0),
                "prob_selection": prob_map.get(client_id, 0.0),
                "has_completed_the_current_round": "yes",
                "is_above_threshold": current_battery >= min_threshold,
                "participation_count": self.client_participation_count.get(client_id, 0),
            })

        # Add details for clients that did not respond

        dead_clients = [c for c in selected_clients if c not in responded_clients]

        for client_id in dead_clients:

            client_details.append({
                "client_id": client_id,
                "device_class": "unknown",
                "current_battery_level": float(0.0),
                "previous_battery_level": float(0.0),
                "consumed_battery": float(0.0),
                "recharged_battery": float(0.0),
                "prob_selection": float(0.0),
                "has_completed_the_current_round": "no",
                "is_above_threshold": int(0),
                "participation_count": int(0),
            })
        
        return client_details

    def reset_round_tracking(self) -> None:
        """Reset per-round tracking data (call at start of each round)."""
        self.round_consumed.clear()
        self.round_recharged.clear()
        self.round_previous_levels.clear()

