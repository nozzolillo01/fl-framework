"""Weights & Biases logging utilities for detailed client tracking."""

import csv
import os
import time
from pathlib import Path
import wandb

PROJECT_NAME = "exp_1"

# Global variables to track CSV files
_metrics_csv_path = None
_clients_csv_path = None
_metrics_csv_writer = None
_clients_csv_writer = None
_metrics_csv_file = None
_clients_csv_file = None

def wandb_init(strategy_name: str, num_supernodes: int, num_server_rounds: int, selection_fraction: float, local_epochs: int, lr: float, alpha: float, min_battery_threshold: float) -> None:
    global _metrics_csv_path, _clients_csv_path, _metrics_csv_writer, _clients_csv_writer
    global _metrics_csv_file, _clients_csv_file

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
    
    # Setup local CSV logging
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_dir = Path("csv_logs")
    csv_dir.mkdir(exist_ok=True)
    
    # Initialize metrics CSV
    _metrics_csv_path = csv_dir / f"metrics_{strategy_name}_{timestamp}.csv"
    _metrics_csv_file = open(_metrics_csv_path, 'w', newline='')
    _metrics_csv_writer = csv.writer(_metrics_csv_file)
    _metrics_csv_writer.writerow(["round", "metric_name", "value"])
    
    # Initialize clients CSV
    _clients_csv_path = csv_dir / f"clients_{strategy_name}_{timestamp}.csv"
    _clients_csv_file = open(_clients_csv_path, 'w', newline='')
    _clients_csv_writer = csv.writer(_clients_csv_file)
    _clients_csv_writer.writerow([
        "round", "client_id", "device_class", "battery_level", 
        "previous_battery_level", "consumed", "recharged", 
        "prob_selection", "was_selected", "completion_status",
        "was_above_threshold", "participation_count"
    ])
    
    print(f"📊 CSV logs will be saved to:")
    print(f"   - Metrics: {_metrics_csv_path}")
    print(f"   - Clients: {_clients_csv_path}")

def log_metrics(server_round: int, metrics: dict) -> None:
    """Log any metrics to W&B and save to local CSV.
    
    Args:
        server_round: Current server round number
        metrics: Dictionary of metrics to log (training, evaluation, fleet, etc.)
    """
    global _metrics_csv_writer, _metrics_csv_file
    
    try:
        wandb.log(metrics, step=server_round)
    except Exception:
        pass
    
    # Save to local CSV
    if _metrics_csv_writer is not None:
        try:
            for metric_name, value in metrics.items():
                _metrics_csv_writer.writerow([server_round, metric_name, value])
            _metrics_csv_file.flush()  # Flush to ensure data is written
        except Exception as e:
            print(f"Warning: Failed to write metrics to CSV: {e}")

def log_client_details_table(server_round: int, client_details: list) -> None:
    """Create and log a table with ALL clients for each round to W&B and save to local CSV."""
    global _clients_csv_writer, _clients_csv_file
    
    if not client_details:
        return
    
    try:
        columns = [
            "round",
            "client_id",
            "device_class",
            "battery_level",
            "previous_battery_level",
            "consumed",
            "recharged",
            "prob_selection",
            "was_selected",
            "completion_status",
            "was_above_threshold",
            "participation_count"
        ]
        
        
        # Build all rows first
        rows = []
        for client in client_details:
            row = [
                server_round,
                str(client["client_id"]),
                client["device_class"],
                client["battery_level"],
                client["previous_battery_level"],
                client["consumed"],
                client["recharged"],
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
    
    # Save to local CSV
    if _clients_csv_writer is not None:
        try:
            for client in client_details:
                _clients_csv_writer.writerow([
                    server_round,
                    str(client["client_id"]),
                    client["device_class"],
                    client["battery_level"],
                    client["previous_battery_level"],
                    client["consumed"],
                    client["recharged"],
                    client["prob_selection"],
                    client["was_selected"],
                    client["completion_status"],
                    client["was_above_threshold"],
                    client["participation_count"]
                ])
            _clients_csv_file.flush()  # Flush to ensure data is written
        except Exception as e:
            print(f"Warning: Failed to write client details to CSV: {e}")


def close_csv_files() -> None:
    """Close all open CSV files. Call this at the end of training."""
    global _metrics_csv_file, _clients_csv_file
    
    if _metrics_csv_file is not None:
        try:
            _metrics_csv_file.close()
            print(f"✅ Metrics CSV file closed successfully")
        except Exception as e:
            print(f"Warning: Failed to close metrics CSV file: {e}")
    
    if _clients_csv_file is not None:
        try:
            _clients_csv_file.close()
            print(f"✅ Clients CSV file closed successfully")
        except Exception as e:
            print(f"Warning: Failed to close clients CSV file: {e}")