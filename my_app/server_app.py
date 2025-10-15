"""my-app: A Flower / PyTorch app."""

import torch
from datetime import datetime
from pathlib import Path
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from my_app.task import Net, central_evaluate
from my_app.battery_simulator import FleetManager
from my_app.strategy import CustomFedAvg

# Create ServerApp
app = ServerApp()

# Global FleetManager instance
fleet_manager = FleetManager()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    selection_fraction: float = context.run_config["selection-fraction"]  
    num_rounds: int = context.run_config["num-server-rounds"]  
    lr: float = context.run_config["lr"]  
    local_epochs: int = context.run_config["local-epochs"]
    
    # Read custom strategy parameters
    selection_strategy: str = context.run_config.get("selection-strategy", "random") 
    alpha: float = context.run_config.get("alpha", 2.0) 
    min_battery_threshold: float = context.run_config.get("min-battery-threshold", 0.2) 

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())  

    # Initialize Custom FedAvg strategy with selection parameters
    selection_params = {
        "selection-fraction": selection_fraction,
        "alpha": alpha,
        "min-battery-threshold": min_battery_threshold,
    }
    
    strategy = CustomFedAvg(
        fleet_manager=fleet_manager,
        selection_strategy=selection_strategy,
        selection_params=selection_params,
        fraction_train=selection_fraction,
        fraction_evaluate=selection_fraction,
    )


    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr, "local-epochs": local_epochs}),
        num_rounds=num_rounds,
        evaluate_fn=central_evaluate,
    )

    # Save final model to disk
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")

