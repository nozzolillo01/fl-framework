"""Weights & Biases logging utilities for detailed client tracking."""

import time
import wandb
import logging
import random

import numpy as np


PROJECT_NAME = "fl-client-selection-framework"


def wandb_init(strategy_name: str, num_supernodes: int, num_server_rounds: int, selection_fraction: float, local_epochs: int, lr: float, alpha: float, min_battery_threshold: float) -> None:

    current_time = time.strftime("%d/%m/%Y_%H:%M:%S")
    run_name = f"[{strategy_name}] - {current_time}"
    wandb.init(project=PROJECT_NAME, name=run_name, config={
        "num-supernodes": num_supernodes,
        "strategy": strategy_name,
        "num-server-rounds": num_server_rounds,
        "selection-fraction": selection_fraction,
        "local-epochs": local_epochs,
        "lr": lr,
        "alpha": alpha,
        "min-battery-threshold": min_battery_threshold,
    })

def log_training_metrics(server_round: int, train_metrics: dict) -> None:
    """Log training metrics to W&B.
    
    Args:
        server_round: Current server round number
        train_metrics: Training metrics from aggregate_train (loss, accuracy, etc.)
    """
    try:
        wandb.log(train_metrics, step=server_round)
    except Exception:
        pass

def log_evaluation_metrics(server_round: int, eval_metrics: dict) -> None:
    """Log evaluation metrics to W&B.

    Args:
        server_round: Current server round number
        eval_metrics: Evaluation metrics (eval_loss, eval_accuracy, etc.)
    """
    try:
        wandb.log(eval_metrics, step=server_round)
    except Exception:
        pass

def log_centralized_metrics(server_round: int, centralized_metrics: dict) -> None:
    """Log centralized/server-side evaluation metrics to W&B.

    Args:
        server_round: Current server round number
        centralized_metrics: Centralized evaluation metrics (centralized_accuracy, centralized_loss, etc.)
    """
    try:
        wandb.log(centralized_metrics, step=server_round)
    except Exception:
        pass

def log_fleet_metrics(server_round: int, fleet_metrics: dict) -> None:
    """Log fleet metrics to W&B.

    Args:
        server_round: Current server round number
        fleet_metrics: Fleet metrics (selected_clients, dead_clients, battery stats, etc.)
    """
    try:
        wandb.log(fleet_metrics, step=server_round)
    except Exception:
        pass

def log_client_details_table(server_round: int, client_details: list) -> None:
    """Crea e registra una nuova tabella semplificata per ogni round in W&B."""
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
            "selected",
            "is_above_threshold",
            "is_dead_during_this_round"
        ]
        table = wandb.Table(columns=columns)

        
        for client in client_details:

            row = [
                int(server_round),
                str(client["client_id"]),
                str(client["device_class"]),
                round(float(client["current_battery_level"]), 4),
                round(float(client["previous_battery_level"]), 4),
                round(float(client["consumed_battery"]), 4),
                round(float(client["recharged_battery"]), 4),
                round(float(client["prob_selection"]), 4),
                int(client["selected"]),
                int(client["is_above_threshold"]),
                int(client["is_dead_during_this_round"])
            ]

            table.add_data(*row)

        wandb.log({f"simplified_table_round_{server_round}": table}, step=server_round)

    except Exception:
        pass

