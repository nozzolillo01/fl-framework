"""Battery simulation for federated learning clients."""

import random
from typing import Optional


class BatterySimulator:
    """Simulates battery behavior for a client device."""

    DEVICE_CLASSES = {
        "low_power_device": {
            "consumption_range": (0.015, 0.025),
            "harvesting_range": (0.0, 0.010),
        },
        "mid_power_device": {
            "consumption_range": (0.030, 0.040),
            "harvesting_range": (0.0, 0.025),
        },
        "high_power_device": {
            "consumption_range": (0.050, 0.070),
            "harvesting_range": (0.0, 0.045),
        },
    }

    def __init__(self, client_id: int, device_class: Optional[str] = None):
        """Initialize battery simulator."""
        self.client_id = client_id
        self.battery_level = random.uniform(0.1, 1.0)
        self.total_consumption = 0.0

        if device_class not in self.DEVICE_CLASSES:
            device_class = random.choice(list(self.DEVICE_CLASSES.keys()))
        self.device_class = device_class
        cmin, cmax = self.DEVICE_CLASSES[self.device_class]["consumption_range"]
        hmin, hmax = self.DEVICE_CLASSES[self.device_class]["harvesting_range"]

        self.consumption_for_epochs = random.uniform(cmin, cmax)
        self.harvesting_capability = random.uniform(hmin, hmax)

    def recharge(self, local_epochs: int = 1) -> float:
        """Recharge battery through energy harvesting."""
        previous_level = self.battery_level
        epochs = max(1, int(local_epochs))
        harvested = self.harvesting_capability * epochs
        self.battery_level = min(1.0, previous_level + harvested)
        effective_harvested = self.battery_level - previous_level
        return effective_harvested

    def consume(self, local_epochs: int) -> bool:
        """Consume battery for training."""
        epochs = max(1, int(local_epochs))
        needed = self.consumption_for_epochs * epochs
        
        if self.battery_level >= needed:
            # Sufficient battery: complete training
            self.battery_level = max(0.0, self.battery_level - needed)
            self.total_consumption += needed
            return True
        else:
            # Insufficient battery: consume all remaining and fail
            consumed = self.battery_level
            self.battery_level = 0.0
            self.total_consumption += consumed
            return False

    def is_bigger_than_threshold(self, min_threshold: float = 0.0) -> bool:
        """Check if client has sufficient battery."""
        return self.battery_level >= min_threshold 

    def is_enough_for_training(self, local_epochs: int) -> bool:
        """Check if battery is sufficient for training."""
        epochs = max(1, int(local_epochs))
        needed = self.consumption_for_epochs * epochs
        return self.battery_level >= needed


class FleetManager:
    """Manages a fleet of client devices with battery simulation."""
    
    def __init__(self):
        """Initialize fleet manager."""
        self.clients: dict[int, BatterySimulator] = {}
        self.client_participation_count: dict[int, int] = {}
        self.client_recharged_battery: dict[int, float] = {}
        self.client_consumed_battery: dict[int, float] = {}
        self.total_consumption_cumulative: float = 0.0
    
    def add_client(self, client_id: int) -> BatterySimulator:
        """Add a new client."""
        if client_id not in self.clients:
            self.clients[client_id] = BatterySimulator(client_id)
        return self.clients[client_id]

    def get_battery_level(self, client_id: int) -> float:
        """Get battery level."""
        if client_id not in self.clients:
            self.add_client(client_id)
        return self.clients[client_id].battery_level

    def get_dead_clients(self, selected_clients: list[int], local_epochs: int) -> list[int]:
        """Get clients that ran out of battery."""
        return [
            cid for cid in selected_clients 
            if not self.clients[cid].is_enough_for_training(local_epochs)
        ]

    def get_clients_above_threshold(self, client_ids: list[int], min_threshold: float = 0.0) -> list[int]:
        """Filter clients by battery threshold."""
        for client_id in client_ids:
            if client_id not in self.clients:
                self.add_client(client_id)
        
        return [
            cid for cid in client_ids
            if self.clients[cid].is_bigger_than_threshold(min_threshold)
        ]

    def calculate_selection_weights(self, client_ids: list[int], alpha: float = 2.0) -> dict[int, float]:
        """Calculate selection weights (battery^alpha)."""
        weights = {}
        for client_id in client_ids:
            if client_id not in self.clients:
                self.add_client(client_id)
            battery_level = self.clients[client_id].battery_level
            weights[client_id] = battery_level ** alpha
        return weights

    def update_round(self, selected_clients: list[int], all_clients: list[int], local_epochs: int) -> None:
        """Update battery levels after training round."""
        round_consumption = 0.0
        
        for client_id in all_clients:
            if client_id not in self.clients:
                self.add_client(client_id)

            self.client_consumed_battery[client_id] = 0.0
            
            if client_id in selected_clients:
                previous_level = self.clients[client_id].battery_level
                self.clients[client_id].consume(local_epochs)
                consumed = previous_level - self.clients[client_id].battery_level
                self.client_consumed_battery[client_id] = consumed
                round_consumption += consumed
                self.client_participation_count[client_id] = self.client_participation_count.get(client_id, 0) + 1

            recharged = self.clients[client_id].recharge(local_epochs)
            self.client_recharged_battery[client_id] = recharged
        
        # Accumulate total consumption
        self.total_consumption_cumulative += round_consumption    

    def get_round_metrics(self, selected_clients: list[int], all_clients: list[int], min_threshold: float, local_epochs: int) -> dict[str, float]:
        """Calculate comprehensive metrics for the current round."""
        # Basic counts
        dead_clients = self.get_dead_clients(selected_clients, local_epochs)
        clients_above_threshold = self.get_clients_above_threshold(all_clients, min_threshold)
        
        # Battery stats for selected clients (BEFORE consumption)
        selected_battery_levels = [self.clients[cid].battery_level for cid in selected_clients if cid in self.clients]
        selected_battery_min = min(selected_battery_levels) if selected_battery_levels else 0.0
        selected_battery_max = max(selected_battery_levels) if selected_battery_levels else 0.0
        selected_battery_avg = sum(selected_battery_levels) / len(selected_battery_levels) if selected_battery_levels else 0.0
        
        # Battery stats for all clients
        all_battery_levels = [self.clients[cid].battery_level for cid in all_clients if cid in self.clients]
        battery_min = min(all_battery_levels) if all_battery_levels else 0.0
        battery_max = max(all_battery_levels) if all_battery_levels else 0.0
        battery_avg = sum(all_battery_levels) / len(all_battery_levels) if all_battery_levels else 0.0
        
        # Fairness index (Jain's fairness)
        total_clients = len(all_clients)
        counts = [self.client_participation_count.get(cid, 0) for cid in all_clients]
        sum_x = sum(counts)
        sum_x2 = sum(c * c for c in counts)
        fairness_jain = 0.0
        if total_clients > 0 and sum_x2 > 0:
            fairness_jain = (sum_x * sum_x) / (total_clients * sum_x2)
        
        # Active clients (selected - dead)
        active_clients = [cid for cid in selected_clients if cid not in dead_clients]
        
        return {
            "selected_clients": float(len(selected_clients)),
            "active_clients": float(len(active_clients)),
            "dead_clients": float(len(dead_clients)),
            "clients_above_threshold": float(len(clients_above_threshold)),
            "total_consumption": self.total_consumption_cumulative,
            "selected_battery_min": selected_battery_min,
            "selected_battery_max": selected_battery_max,
            "selected_battery_avg": selected_battery_avg,
            "battery_min": battery_min,
            "battery_max": battery_max,
            "battery_avg": battery_avg,
            "fairness_index_jain": fairness_jain,
        }
    
    def get_client_details(self, all_clients: list[int], selected_clients: list[int], dead_clients: list[int], min_threshold: float, prob_map: dict[int, float], previous_battery_levels: dict[int, float]) -> list[dict]:
        """Get detailed information for each client for W&B table logging."""
        client_details = []
        
        for client_id in all_clients:
            if client_id not in self.clients:
                continue
                
            client = self.clients[client_id]
            
            # Get previous battery level (before consumption/recharge)
            previous_battery = previous_battery_levels.get(client_id)
            
            # Get consumption and recharge for this round
            consumed = self.client_consumed_battery.get(client_id)
            recharged = self.client_recharged_battery.get(client_id)
            
            client_details.append({
                "client_id": client_id,
                "device_class": client.device_class,
                "current_battery_level": client.battery_level,
                "previous_battery_level": previous_battery,
                "consumed_battery": consumed,
                "recharged_battery": recharged,
                "prob_selection": prob_map.get(client_id),
                "selected": client_id in selected_clients,
                "is_above_threshold": client.is_bigger_than_threshold(min_threshold),
                "is_dead_during_this_round": client_id in dead_clients,
            })
        
        return client_details
