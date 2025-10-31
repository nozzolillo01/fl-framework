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
from my_app.wandb_utils import wandb_init, log_metrics, log_client_details_table, close_csv_files


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
    
    def configure_train(self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid) -> Iterable[Message]:
        """Configure the next round of federated training with custom client selection.
        
        Uses battery levels collected from previous query (battery_for_selection).
        """
        # Get all available nodes
        all_node_ids = list(grid.get_node_ids())
        self.all_node_ids = all_node_ids
        
        # Apply custom selection strategy using battery levels from previous query
        selected_node_ids, prob_map = self.selection_strategy_fn(
            all_node_ids,
            self.fleet_manager,
            self.selection_params
        )
        
        # Store selected clients and probabilities for later logging
        self.selected_node_ids = list(selected_node_ids)
        self.round_prob_map = prob_map
        
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
        """Aggregate training results from successful clients.
        
        No battery info here - that comes later in the query phase.
        """
        responded_clients = []

        # Track which clients responded successfully
        for reply in replies:
            client_id = reply.metadata.src_node_id
            
            # Check if reply contains an error (battery exhaustion)
            if reply.has_error():
                log(INFO, f"Client {client_id} failed training (battery exhausted)")
                continue

            # Client responded successfully
            responded_clients.append(client_id)

        self.alive_node_ids = responded_clients
        
        # Update participation tracking only for clients that completed training
        if responded_clients:
            self.fleet_manager.update_participation(responded_clients)

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
        """Aggregate evaluation results.
        
        Note: Do NOT clear selected_node_ids here - they are needed for post-round logging.
        """
        result = super().aggregate_evaluate(server_round, replies)
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

        #ask for battery status at the beginning of the first round
        all_node_ids = list(grid.get_node_ids())
        # Send battery status query to all clients
        query_config = ConfigRecord()
        query_record = RecordDict({"config": query_config})

        query_messages = self._construct_messages(query_record, all_node_ids, "query")

        battery_replies = grid.send_and_receive(
            messages=query_messages,
            timeout=timeout,
        )

        for reply in battery_replies:
            if not reply.has_error():
                # Process valid battery status replies
                client_id = reply.metadata.src_node_id
                metrics = dict(reply.content.get("metrics", {}))
                battery_metrics = {
                    "device_class": metrics.get("device_class", 0),
                    "battery_level": metrics.get("battery_level", 0.0),
                    }
                self.fleet_manager.update_client_info(client_id, battery_metrics)
                
        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # -----------------------------------------------------------------
            # --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure training round (uses battery_for_selection from previous query)
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

            # Log training metrics
            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            # -----------------------------------------------------------------
            # --- EVALUATION (CLIENTAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    arrays,
                    evaluate_config,
                    grid,
                ),
                timeout=timeout,
            )

            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                evaluate_replies,
            )

            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics

            # -----------------------------------------------------------------
            # --- EVALUATION (SERVERAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res

            # -----------------------------------------------------------------
            # --- QUERY BATTERY STATUS AFTER ROUND (ALL CLIENTS) --------------
            # -----------------------------------------------------------------

            all_node_ids = list(grid.get_node_ids())
            log(INFO, f"Querying battery status from all {len(all_node_ids)} clients after round {current_round}...")
            
            query_config = ConfigRecord()
            query_record = RecordDict({"config": query_config})
            query_messages = self._construct_messages(query_record, all_node_ids, "query")
            battery_replies = grid.send_and_receive(
                messages=query_messages,
                timeout=timeout,
            )

            # Update fleet manager and collect round info for logging
            for reply in battery_replies:
                if not reply.has_error():
                    client_id = reply.metadata.src_node_id
                    metrics = dict(reply.content.get("metrics", {}))
                    
                    # Extract completed round info
                    battery_metrics = {
                        "device_class": metrics.get("device_class", 0),
                        "battery_level": metrics.get("battery_level", 0.0),
                        "previous_battery_level": metrics.get("previous_battery_level", 0.0),
                        "consumed": metrics.get("consumed", 0.0),
                        "recharged": metrics.get("recharged", 0.0),
                    }
                    self.fleet_manager.update_client_info(client_id, battery_metrics)

            # Calculate fleet metrics for completed round
            fleet_metrics = self.fleet_manager.get_round_metrics(
                selected_clients=self.selected_node_ids,
                responded_clients=self.alive_node_ids,
                total_clients=all_node_ids
            )
            
            # Get client details for W&B table
            min_threshold = self.selection_params.get("min-battery-threshold", 0.0)
            client_details = self.fleet_manager.get_client_details(
                all_clients=all_node_ids,
                selected_clients=self.selected_node_ids,
                responded_clients=self.alive_node_ids,
                prob_map=self.round_prob_map,
                min_threshold=min_threshold
            )

            # Log everything to W&B
            combined_metrics = {}
            if agg_train_metrics:
                combined_metrics.update(dict(agg_train_metrics))
            if agg_evaluate_metrics:
                combined_metrics.update(dict(agg_evaluate_metrics))
            if res:
                combined_metrics.update(dict(res))
            combined_metrics.update(fleet_metrics)
            
            log_metrics(server_round=current_round, metrics=combined_metrics)
            log_client_details_table(server_round=current_round, client_details=client_details)
            
            # Clear selection tracking for next round
            self.selected_node_ids = []
            self.alive_node_ids = []
            self.round_prob_map = {}
            
            log(INFO, "")

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")

        # Close CSV files
        close_csv_files()

        return result
    