"""my-app: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict, Error
from flwr.clientapp import ClientApp
from flwr.common import log
from logging import INFO

from my_app.task import Net, load_data
from my_app.task import test as test_fn
from my_app.task import train as train_fn
from my_app.battery_simulator import BatterySimulator

# Flower ClientApp
app = ClientApp()

@app.query()
def get_battery_status(msg: Message, context: Context):
    """Report battery status - handles both initial query and post-round queries.
    
    FIRST call (before round 1): Returns only device_class and battery_level
    SUBSEQUENT calls (after each round): Returns complete round history
    """
    node_id = context.node_id

    # Check if this is the first query (initialization)
    if "battery_sim_state" not in context.state:
        # FIRST QUERY - Initialize battery simulator
        battery_sim = BatterySimulator(client_id=node_id)
        
        # Apply initial recharge
        recharge_info = battery_sim.recharge()
        initial_recharged = recharge_info.get("recharged", 0.0)
        
        # Save state for first round
        context.state["battery_sim_state"] = ConfigRecord({
            "battery_level": battery_sim.battery_level,
            "device_class": battery_sim.device_class,
            "round_start_battery": battery_sim.battery_level,
            "round_consumed": 0.0,
            "round_recharged": initial_recharged,
        })
        
        # Return only initial info
        metrics = MetricRecord({
            "device_class": battery_sim.device_class,
            "battery_level": battery_sim.battery_level,
        })
        
        return Message(
            content=RecordDict({"metrics": metrics}),
            reply_to=msg
        )
    
    # SUBSEQUENT QUERIES - Return complete round history
    state = context.state["battery_sim_state"]
    battery_sim = BatterySimulator(client_id=node_id)
    battery_sim.battery_level = state["battery_level"]
    battery_sim.device_class = state["device_class"]
    
    # Get round history from state
    round_start_battery = state.get("round_start_battery", battery_sim.battery_level)
    round_consumed = state.get("round_consumed", 0.0)
    round_recharged = state.get("round_recharged", 0.0)
    
    # Current battery is at end of round (after consumption, before next recharge)
    batteria_attuale = battery_sim.battery_level
    batteria_passata = round_start_battery
    delta_scarica = round_consumed
    delta_ricarica = round_recharged
    
    # Apply recharge for NEXT round
    recharge_info = battery_sim.recharge()
    next_round_recharged = recharge_info.get("recharged", 0.0)
    
    # Save state for next round
    context.state["battery_sim_state"] = ConfigRecord({
        "battery_level": battery_sim.battery_level,  # After recharge
        "device_class": battery_sim.device_class,
        "round_start_battery": battery_sim.battery_level,  # This will be start of next round
        "round_consumed": 0.0,  # Will be updated in train() if selected
        "round_recharged": next_round_recharged,  # Recharge we just applied
    })
    
    # Return completed round info
    metrics = MetricRecord({
        "device_class": battery_sim.device_class,
        "batteria_attuale": batteria_attuale,
        "batteria_passata": batteria_passata,
        "delta_scarica": delta_scarica,
        "delta_ricarica": delta_ricarica,
        "battery_for_selection": battery_sim.battery_level,  # After recharge, for next round
    })
    
    return Message(
        content=RecordDict({"metrics": metrics}),
        reply_to=msg
    )

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data with battery simulation."""
    node_id = context.node_id
    local_epochs = context.run_config["local-epochs"]

    # Restore battery simulator state
    state = context.state["battery_sim_state"]
    battery_sim = BatterySimulator(client_id=node_id)
    battery_sim.battery_level = state["battery_level"]
    battery_sim.device_class = state["device_class"]
    
    # Consume battery for training
    round_info = battery_sim.consume(local_epochs)
    consumed = round_info.get("consumed", 0.0)

    # Update state with consumption info (will be reported in next query)
    context.state["battery_sim_state"] = ConfigRecord({
        "battery_level": battery_sim.battery_level,  # After consumption
        "device_class": battery_sim.device_class,
        "round_start_battery": state.get("round_start_battery", battery_sim.battery_level),
        "round_consumed": consumed,  # Save consumption for next query
        "round_recharged": state.get("round_recharged", 0.0),
    })

    if not round_info["training_completed"]:
        # Battery is insufficient, return error
        error = Error(code=1, reason=f"ran out of battery")
        return Message(error=error, reply_to=msg)

    # Battery is sufficient - proceed with actual training
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _ = load_data(partition_id, num_partitions)
    
    # Train the model
    train_loss, train_accuracy = train_fn(
        model, trainloader, local_epochs, msg.content["config"]["lr"], device
    )
    
    # Return updated model and metrics (NO battery info here, it's in query)
    model_record = ArrayRecord(model.state_dict())
    metrics = MetricRecord({
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "num-examples": len(trainloader.dataset),
    })
    
    return Message(
        content=RecordDict({"arrays": model_record, "metrics": metrics}),
        reply_to=msg
    )

@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader = load_data(partition_id, num_partitions)  

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_accuracy": eval_acc,
        "num-examples": len(valloader.dataset) 
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)