"""Weights & Biases logging utilities for detailed client tracking."""

import time
import wandb

PROJECT_NAME = "fl-framework"

def wandb_init(strategy_name: str, num_supernodes: int, num_server_rounds: int, selection_fraction: float, local_epochs: int, lr: float, alpha: float, min_battery_threshold: float) -> None:

    current_time = time.strftime("%d/%m/%Y_%H:%M:%S")
    run_name = f"[{strategy_name}] - {current_time}"
    wandb.init(
     project=PROJECT_NAME,
     name=run_name,
     config={
        "num-supernodes": num_supernodes,
        "strategy": strategy_name,
        "num-server-rounds": num_server_rounds,
        "selection-fraction": selection_fraction,
        "local-epochs": local_epochs,
        "lr": lr,
        "alpha": alpha,
        "min-battery-threshold": min_battery_threshold,
    })

def log_metrics(server_round: int, metrics: dict) -> None:
    """Log any metrics to W&B.
    
    Args:
        server_round: Current server round number
        metrics: Dictionary of metrics to log (training, evaluation, fleet, etc.)
    """
    try:
        wandb.log(metrics, step=server_round)
    except Exception:
        pass

def log_client_details_table(server_round: int, client_details: list) -> None:
    """Create and log a table with ALL clients for each round to W&B."""
    if not client_details:
        return
    
    try:
        columns = [
            "round",
            "client_id",
            "device_class",
            "current_battery_level",
            "previous_battery_level",
            "consumed_battery",
            "recharged_battery",
            "prob_selection",
            "was_selected",
            "completion_status",
            "was_above_threshold",
            "participation_count"
        ]
        
        
        # Costruisci tutte le righe prima
        rows = []
        for client in client_details:
            row = [
                server_round,
                str(client["client_id"]),
                client["device_class"],
                client["current_battery_level"],
                client["previous_battery_level"],
                client["consumed_battery"],
                client["recharged_battery"],
                client["prob_selection"],
                client["was_selected"],
                client["completion_status"],
                client["was_above_threshold"],
                client["participation_count"]
            ]
            rows.append(row)
        

        table = wandb.Table(columns=columns, data=rows)
        wandb.log({f"all_clients_details_round_{server_round}": table}, step=server_round)
        
    except Exception as e:
        print(f"Warning: Failed to log client details table for round {server_round}: {e}")