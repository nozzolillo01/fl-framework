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

        self.all_node_ids: list = [] # Store all available clients
        self.selected_node_ids: list = [] # Store selected clients
        self.alive_node_ids: list = [] # Store clients that completed training successfully
        self.round_fleet_metrics: Optional[dict] = None # Store fleet metrics for W&B
        
        self.round_client_details: list = [] # Store client details for W&B
        self.round_prob_map: dict[int, float] = {} # Store selection probabilities for current round
        
    def query_all_batteries(self, grid: Grid, timeout: float = 60) -> None:
        """Query battery status from all clients and update FleetManager.

        This is called at the start of each round in configure_train().

        Clients will apply idle_recharge() before responding (except first time).
        
        Args:
            grid: Grid instance to get client nodes
            timeout: Timeout for battery query
        """
        all_node_ids = list(grid.get_node_ids())
        
        log(INFO, f"Querying battery status from all {len(all_node_ids)} clients...")
        
        # Send battery status query to all clients
        query_config = ConfigRecord()
        query_record = RecordDict({"config": query_config})

        query_messages = self._construct_messages(query_record, all_node_ids, "query")
        
        battery_replies = grid.send_and_receive(
            messages=query_messages,
            timeout=timeout,
        )
        
        # Update fleet manager with battery levels (after recharge)
        for reply in battery_replies:
            if not reply.has_error():
                client_id = reply.metadata.src_node_id
                metrics = dict(reply.content.get("metrics", {}))
                battery_metrics = {
                    "device_class": metrics.get("device_class", 0),
                    "battery_level": metrics.get("battery_level", 0.0),
                    "recharged": metrics.get("battery_recharged", 0.0),
                    "previous_battery_level": metrics.get("previous_battery_level", 0.0)
                    }
                self.fleet_manager.update_pre_training(client_id, battery_metrics)
    
    def configure_train(self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid) -> Iterable[Message]:
        """Configure the next round of federated training with custom client selection."""

        # Get all available nodes
        all_node_ids = list(grid.get_node_ids())

        self.all_node_ids = all_node_ids
        
        # Query battery status from ALL clients (to get updated levels)
        self.query_all_batteries(grid)
        
        # Apply custom selection strategy with updated battery levels
        selected_node_ids, prob_map = self.selection_strategy_fn(
            all_node_ids,
            self.fleet_manager,
            self.selection_params
        )
        
        # Store active clients and probabilities for later use
        self.selected_node_ids = list(selected_node_ids)
        self.round_prob_map = prob_map
        
        # Log the selection
        log(
            INFO,
            "configure_train: Strategy [%s] sampled %s nodes (out of %s)",
            self.selection_strategy_name,
            len(selected_node_ids),
            len(all_node_ids),
        )

        # Inject current server round
        config["server-round"] = server_round

        # Construct messages for selected clients
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, selected_node_ids, "train")

    def aggregate_train(self, server_round: int, replies: Iterable[Message]) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate training results and update fleet manager with battery metrics from clients."""
        
        # Calculate failed clients by checking for errors in replies
        responded_clients = []

        # Extract battery metrics from successful client replies
        for reply in replies:
            client_id = reply.metadata.src_node_id
            
            # Check if reply contains an error (battery exhaustion)
            if reply.has_error():
                continue

            # Client responded successfully
            responded_clients.append(client_id)

            metrics_dict = dict(reply.content.get("metrics", {}))
            
            # Update fleet manager with reported battery status (after training consumption)
            battery_metrics = {
                "device_class": metrics_dict.pop("device_class", "unknown"),
                "battery_level": metrics_dict.pop("battery_level", 0.0),
                "consumed": metrics_dict.pop("battery_consumed", 0.0),
                "previous_battery_level": metrics_dict.pop("previous_battery_level", 0.0),
                }
            # Note: recharged is already tracked in update_pre_training (during query phase)
            
            #Update fleet manager with reported battery status for the client who responded successfully after training
            self.fleet_manager.update_after_training(client_id, battery_metrics)

            # Create cleaned reply (without battery metrics)
            reply.content["metrics"] = MetricRecord(metrics_dict)

        self.alive_node_ids = responded_clients
        
        # Update participation tracking only for clients that completed training
        if responded_clients:
            self.fleet_manager.update_participation(responded_clients)
        
        # Calculate fleet metrics for logging
        min_threshold = self.selection_params.get("min-battery-threshold", 0.0)
        
        self.round_fleet_metrics = self.fleet_manager.get_round_metrics(
            selected_clients=self.selected_node_ids,
            responded_clients=self.alive_node_ids,
            total_clients=self.all_node_ids
        )
        
        # Build client details for W&B table (include ALL clients)
        self.round_client_details = self.fleet_manager.get_client_details(
            all_clients=self.all_node_ids,
            selected_clients=self.selected_node_ids,
            responded_clients=self.alive_node_ids,
            prob_map=self.round_prob_map,
            min_threshold=min_threshold
        )

        # Call parent aggregation
        if replies:
            return super().aggregate_train(server_round, replies)
        else:
            # No valid training results to aggregate (all clients failed)
            log(INFO, "No valid training results to aggregate (all clients failed)")
            return None, None

    def configure_evaluate(self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid) -> Iterable[Message]:
        """Configure evaluation to use the same clients selected for train."""

        # Use the same clients selected for training (excluding dead)
        selected_for_eval_node_ids = list(self.alive_node_ids)

        # Inject server round
        config["server-round"] = server_round

        record = RecordDict({self.arrayrecord_key: arrays, self.configrecord_key: config})
        return self._construct_messages(record, selected_for_eval_node_ids, "evaluate")

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate evaluation results and clear persisted selection for the round."""


        result = super().aggregate_evaluate(server_round, replies)

        # Clear storage
        self.selected_node_ids = []
        self.alive_node_ids = []

        return result

   
    def start(self,
              grid: Grid,
              initial_arrays: ArrayRecord,
              num_rounds: int,
              timeout: float = 3600,
              train_config: Optional[ConfigRecord] = None,
              evaluate_config: Optional[ConfigRecord] = None,
              evaluate_fn: Optional[Callable[[int, ArrayRecord], Optional[MetricRecord]]] = None) -> Result:
        """Execute the federated learning strategy"""

        # Suppress wandb deprecation warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="wandb")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.protobuf")

        # Init W&B
        wandb_init(
            strategy_name=self.selection_strategy_name,
            num_supernodes=len(grid.get_node_ids()),
            num_server_rounds=num_rounds,
            selection_fraction=self.selection_params.get("selection-fraction"),
            local_epochs=train_config.get("local-epochs") if train_config else None,
            lr=train_config.get("lr") if train_config else None,
            alpha=self.selection_params.get("alpha"),
            min_battery_threshold=self.selection_params.get("min-battery-threshold"))

        log(INFO, "")
        log(INFO, "Starting %s with [%s] client selection strategy...", self.__class__.__name__, self.selection_strategy_name)
        log(INFO, "")
        #log numero di round
        log(INFO, "Number of server rounds: %s", num_rounds)
        #log numero di local epochs
        log(INFO, "Local epochs: %s", train_config.get("local-epochs"))
        #log numero di client
        log(INFO, "Total clients: %s", len(grid.get_node_ids()))
        #log lr
        log(INFO, "Learning rate: %s", train_config.get("lr"))
        #log fraction
        log(INFO, "Fraction of clients selected: %.2f", self.selection_params.get("selection-fraction", 0.0))

        #log parametri selezione
        if self.selection_strategy_name == "battery_aware":
            
            log(INFO, "Alpha: %.2f", self.selection_params.get("alpha"))
            log(INFO, "Min battery threshold: %.2f", self.selection_params.get("min-battery-threshold"))
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

            # Use fleet metrics calculated in aggregate_train (after receiving client reports)
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
    