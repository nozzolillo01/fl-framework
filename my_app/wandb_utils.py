"""Weights & Biases logging utilities for detailed client tracking."""

import csv
import os
import time
from pathlib import Path
import wandb

PROJECT_NAME = "exp_2"

# Global variables for DOE output CSV
_output_csv_path = "output2.csv"
_output_csv_file = None
_output_csv_writer = None
_current_experiment_config = {}
_current_round_metrics = {}

def wandb_init(strategy_name: str, num_supernodes: int, num_server_rounds: int, selection_fraction: float, local_epochs: int, lr: float, alpha: float, min_battery_threshold: float) -> None:
    global _output_csv_file, _output_csv_writer, _current_experiment_config

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
    
    # Store current experiment configuration (factors for DOE)
    # Note: num_server_rounds stored separately (not a DoE factor, just for checking final round)
    _current_experiment_config = {
        "strategy": strategy_name,
        "num_supernodes": num_supernodes,
        "num_server_rounds": num_server_rounds,  # Keep for internal use
        "selection_fraction": selection_fraction,
        "local_epochs": local_epochs,
        "learning_rate": lr,
        "alpha": alpha if alpha is not None else "",
        "min_battery_threshold": min_battery_threshold if min_battery_threshold is not None else "",
    }
    
    # Initialize or append to output.csv
    file_exists = Path(_output_csv_path).exists()
    _output_csv_file = open(_output_csv_path, 'a', newline='')
    _output_csv_writer = csv.writer(_output_csv_file)
    
    # Write header only if file is new
    if not file_exists:
        _output_csv_writer.writerow([
            # Factors (independent variables)
            "strategy", "num_supernodes", "selection_fraction",
            "local_epochs", "learning_rate", "alpha", "min_battery_threshold",
            # Snapshot info
            "round", "is_final",
            # Responses (dependent variables)
            "selected_clients", "dead_clients", "total_consumption",
            "battery_min", "battery_max", "battery_avg", "fairness_index_jain",
            "train_loss", "eval_loss", "centralized_loss",
            "train_accuracy", "eval_accuracy", "centralized_accuracy"
        ])
    
    print(f"📊 DOE output will be appended to: {_output_csv_path}")
    print(f"   (Snapshots saved every 10 rounds + final round)")

def log_metrics(server_round: int, metrics: dict) -> None:
    """Log any metrics to W&B and accumulate for DOE output.
    
    Args:
        server_round: Current server round number
        metrics: Dictionary of metrics to log (training, evaluation, fleet, etc.)
    """
    global _current_round_metrics, _output_csv_writer, _output_csv_file, _current_experiment_config
    
    try:
        wandb.log(metrics, step=server_round)
    except Exception:
        pass
    
    # Accumulate metrics for current round
    _current_round_metrics.update(metrics)
    _current_round_metrics["round"] = server_round

def log_client_details_table(server_round: int, client_details: list) -> None:
    """Create and log a table with ALL clients for each round to W&B.
    Also write aggregated DOE data to output.csv ONLY for the final round.
    """
    global _output_csv_writer, _output_csv_file, _current_experiment_config, _current_round_metrics
    
    # Log to wandb only if there are client details
    if client_details:
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
    
    # Write aggregated DOE data to output.csv periodically or at final round
    # Save every SNAPSHOT_INTERVAL rounds + final round
    SNAPSHOT_INTERVAL = 10  # Save every 10 rounds
    num_server_rounds = _current_experiment_config.get("num_server_rounds", 0)
    is_final_round = (server_round == num_server_rounds)
    is_snapshot_round = (server_round % SNAPSHOT_INTERVAL == 0)
    should_save = is_final_round or is_snapshot_round
    
    if _output_csv_writer is not None and should_save:
        try:
            row = [
                # Factors (num_server_rounds removed - not a DoE factor)
                _current_experiment_config.get("strategy", ""),
                _current_experiment_config.get("num_supernodes", ""),
                _current_experiment_config.get("selection_fraction", ""),
                _current_experiment_config.get("local_epochs", ""),
                _current_experiment_config.get("learning_rate", ""),
                _current_experiment_config.get("alpha", ""),
                _current_experiment_config.get("min_battery_threshold", ""),
                # Snapshot info
                server_round,
                is_final_round,
                # Responses
                _current_round_metrics.get("selected_clients", ""),
                _current_round_metrics.get("dead_clients", ""),
                _current_round_metrics.get("total_consumption", ""),
                _current_round_metrics.get("battery_min", ""),
                _current_round_metrics.get("battery_max", ""),
                _current_round_metrics.get("battery_avg", ""),
                _current_round_metrics.get("fairness_index_jain", ""),
                _current_round_metrics.get("train_loss", ""),
                _current_round_metrics.get("eval_loss", ""),
                _current_round_metrics.get("centralized_loss", ""),
                _current_round_metrics.get("train_accuracy", ""),
                _current_round_metrics.get("eval_accuracy", ""),
                _current_round_metrics.get("centralized_accuracy", ""),
            ]
            _output_csv_writer.writerow(row)
            _output_csv_file.flush()
            
            status = "final" if is_final_round else "snapshot"
            print(f"✅ DOE data written to output.csv (round {server_round}, {status})")
            
        except Exception as e:
            print(f"Warning: Failed to write DOE data to output.csv: {e}")
    
    # Clear metrics for next round
    _current_round_metrics.clear()


def close_csv_files() -> None:
    """Close output.csv file. Call this at the end of training."""
    global _output_csv_file
    
    if _output_csv_file is not None:
        try:
            _output_csv_file.close()
            print(f"✅ DOE output CSV file closed successfully: {_output_csv_path}")
        except Exception as e:
            print(f"Warning: Failed to close output CSV file: {e}")