"""Battery-aware federated learning strategy."""

import time
import warnings
from collections.abc import Iterable
from logging import INFO
from typing import Callable, Optional

from flwr.common import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict, log
from flwr.server import Grid
from flwr.serverapp.strategy import FedAvg, Result

from my_app.selection_strategies import get_selection_strategy
from my_app.wandb_utils import wandb_init, log_metrics, log_client_details_table


class CustomFedAvg(FedAvg):
    """Custom FedAvg with configurable client selection strategies.

    Extends FedAvg to support custom client selection strategies defined in selection_strategies.py.
    """

    def __init__(self, fleet_manager, selection_strategy: str = "random", selection_params: Optional[dict] = None, *args, **kwargs):

        """Initialize strategy with fleet manager and selection strategy.

        Args:
            fleet_manager: FleetManager instance to track batteries
            selection_strategy: Name of selection strategy ("random", "battery_aware", "all_available")
            selection_params: Parameters for the selection strategy (e.g., alpha, min_battery_threshold)
            *args, **kwargs: Arguments for FedAvg parent class
        """

        super().__init__(*args, **kwargs)
        self.fleet_manager = fleet_manager
        self.selection_strategy_name = selection_strategy
        self.selection_strategy_fn = get_selection_strategy(selection_strategy)
        self.selection_params = selection_params or {}

        self.active_node_ids: list = [] # Store selected active clients for evaluation
        self.round_fleet_metrics: Optional[dict] = None # Store fleet metrics for W&B
        self.round_client_details: list = [] # Store client details for W&B
        




    def configure_train(self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid) -> Iterable[Message]:
        """Configure the next round of federated training with custom client selection."""

        # Get all available nodes
        all_node_ids = list(grid.get_node_ids())
        
        # Check minimum available nodes
        if len(all_node_ids) < self.min_available_nodes:
            return []
        
        # Ensure all clients are in fleet_manager
        for node_id in all_node_ids:
            if node_id not in self.fleet_manager.clients:
                self.fleet_manager.add_client(node_id)
        
        # Save battery levels BEFORE update
        previous_battery_levels = {
            node_id: self.fleet_manager.get_battery_level(node_id)
            for node_id in all_node_ids
        }
        
        # Apply custom selection strategy
        selected_node_ids, prob_map = self.selection_strategy_fn(
            all_node_ids,
            self.fleet_manager,
            self.selection_params
        )
        
        # Get local epochs for battery calculations
        local_epochs = config.get("local-epochs")


        # Identify dead clients (insufficient battery for training)
        dead_clients = self.fleet_manager.get_dead_clients(selected_node_ids, local_epochs)
        
        # Exclude dead clients from actual training
        active_node_ids = [cid for cid in selected_node_ids if cid not in dead_clients]

        # Persist the selection (excluding dead clients) for evaluate
        self.active_node_ids = list(active_node_ids)
        
        
        # Calculate metrics BEFORE update
        min_threshold = self.selection_params.get("min-battery-threshold")
        self.round_fleet_metrics = self.fleet_manager.get_round_metrics(
            selected_clients=selected_node_ids,
            all_clients=all_node_ids,
            min_threshold=min_threshold,
            local_epochs=local_epochs
        )

        # Update fleet manager: all selected clients (including dead) consume/recharge
        self.fleet_manager.update_round(selected_node_ids, all_node_ids, local_epochs)
        
        # Build client details for W&B table
        self.round_client_details = self.fleet_manager.get_client_details(
            all_clients=all_node_ids,
            selected_clients=selected_node_ids,
            dead_clients=dead_clients,
            min_threshold=min_threshold,
            prob_map=prob_map,
            previous_battery_levels=previous_battery_levels
        )
        
        
        # Log the selection
        log(
            INFO,
            "configure_train: Strategy [%s] sampled %s nodes (out of %s), %s active, %s dead",
            self.selection_strategy_name,
            len(selected_node_ids),
            len(all_node_ids),
            len(active_node_ids),
            len(dead_clients),
        )
    
        # Always inject current server round
        config["server-round"] = server_round

        # Construct messages only for ACTIVE clients (excluding dead)
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, active_node_ids, "train")


    def aggregate_train(self, server_round: int, replies: Iterable[Message], ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate training results."""
        return super().aggregate_train(server_round, replies)


    def configure_evaluate(self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid) -> Iterable[Message]:
        """Configure evaluation to use the same clients selected for train."""

        # Use the same clients selected for training (excluding dead)
        selected_node_ids = list(self.active_node_ids)

        # Inject server round
        config["server-round"] = server_round

        record = RecordDict({self.arrayrecord_key: arrays, self.configrecord_key: config})
        return self._construct_messages(record, selected_node_ids, "evaluate")


    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate evaluation results and clear persisted selection for the round."""
        result = super().aggregate_evaluate(server_round, replies)

        # Clear stored selection after evaluation aggregation
        self.active_node_ids = []


        return result

    def start(self, grid: Grid, initial_arrays: ArrayRecord, num_rounds: int = 3, timeout: float = 3600, train_config: Optional[ConfigRecord] = None, evaluate_config: Optional[ConfigRecord] = None, evaluate_fn: Optional[Callable[[int, ArrayRecord], Optional[MetricRecord]]] = None) -> Result:
        """Execute the federated learning strategy"""

        # Suppress wandb deprecation warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="wandb")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.protobuf")

        # Init W&B
        wandb_init(strategy_name=self.selection_strategy_name, num_supernodes=len(grid.get_node_ids()), num_server_rounds=num_rounds, selection_fraction=self.fraction_train, local_epochs=train_config.get("local-epochs") if train_config else None, lr=train_config.get("lr") if train_config else None, alpha=self.selection_params.get("alpha"), min_battery_threshold=self.selection_params.get("min-battery-threshold"))

        log(INFO, "")
        log(INFO, "Starting %s with [%s] client selection strategy...", self.__class__.__name__, self.selection_strategy_name)
        log(INFO, "")
        #log numero di round
        log(INFO, "Number of server rounds: %s", num_rounds)
        #log numero di local epochs
        log(INFO, "Local epochs: %s", train_config.get("local-epochs"))
        #log numero di client
        log(INFO, "Total clients: %s", len(grid.get_node_ids()))
        #log fraction
        log(INFO, "Fraction of client selected: %.2f", self.fraction_train)
        log(INFO, "")

        # Initialize if None
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()
        # Evaluate starting global parameters
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # -----------------------------------------------------------------
            # --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure training round
            # Send messages and wait for replies
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    arrays,
                    train_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate train
            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round,
                train_replies,
            )

            # Use pre-calculated fleet metrics (from configure_train, before battery update)
            fleet_metrics = self.round_fleet_metrics if self.round_fleet_metrics else {}
            client_details = self.round_client_details if self.round_client_details else []

            # Log training metrics and append to history
            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

                # Log to W&B
                log_metrics(server_round=current_round, metrics={**dict(agg_train_metrics), **fleet_metrics})
                log_client_details_table(server_round=current_round, client_details=client_details)

            # -----------------------------------------------------------------
            # --- EVALUATION (CLIENTAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure evaluation round
            # Send messages and wait for replies
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    arrays,
                    evaluate_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate evaluate
            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                evaluate_replies,
            )

            # Log evaluation metrics and append to history
            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics
                # Log to W&B
                log_metrics(server_round=current_round, metrics=dict(agg_evaluate_metrics))

            # -----------------------------------------------------------------
            # --- EVALUATION (SERVERAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Centralized evaluation
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res
                    # Log to W&B
                    log_metrics(server_round=current_round, metrics=dict(res))
            log(INFO, "")

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")

        return result
    