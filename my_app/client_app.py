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
        # This query is BEFORE round 1, just to provide initial battery level

        battery_sim = BatterySimulator(client_id=node_id)
        
        # Save state for round 1
        context.state["battery_sim_state"] = ConfigRecord({
            "battery_level": battery_sim.battery_level,
            "device_class": battery_sim.device_class,
        })
        
        # Return only initial info for round 1 selection
        metrics = MetricRecord({
            "battery_level": battery_sim.battery_level,
            "device_class": battery_sim.device_class,
        })
        
        return Message(
            content=RecordDict({"metrics": metrics}),
            reply_to=msg
        )
    
    # SUBSEQUENT QUERIES - Restore simulator from saved state
    # This query reports the COMPLETED round's metrics
    state = context.state["battery_sim_state"]
    
    # Get info about the round
    if "consumed" in state:
        # If the client has been selected for this round
        previous_battery_level = state.get("previous_battery_level", 0.0)
        consumed = state.get("consumed", 0.0)
        battery_level = state.get("battery_level", 0.0)
    else:
        previous_battery_level = state.get("battery_level", 0.0)
        consumed = 0.0
        battery_level = state.get("battery_level", 0.0)

    # Restore battery simulator with the current battery level
    device_class = state.get("device_class", node_id % 3)
    battery_sim = BatterySimulator(
        client_id=node_id, 
        initial_battery=battery_level,
        device_class=device_class
    )

    # Apply recharge for all the clients
    recharge_info = battery_sim.recharge()

    recharged = recharge_info.get("recharged", 0.0)
    battery_level = recharge_info.get("battery_level", 0.0)
    
    
    # Save state for next round reporting
    context.state["battery_sim_state"] = ConfigRecord({
        "battery_level": battery_level,
        "recharged": recharged,
        "device_class": battery_sim.device_class,
    })
    
    # Return completed round info
    # Formula verified: current_battery = previous_battery + recharged - consumed
    metrics = MetricRecord({
        "device_class": battery_sim.device_class,
        "battery_level": battery_level,                     # Battery at end of round (after consume and recharge)
        "previous_battery_level": previous_battery_level,   # Battery at start of round
        "consumed": consumed,                               # Consumed during this round
        "recharged": recharged,                             # Recharged during this round
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

    # Restore battery simulator from saved state
    state = context.state["battery_sim_state"]
    current_battery = state.get("battery_level", 0.0)
    device_class = state.get("device_class", node_id % 3)
    
    battery_sim = BatterySimulator(
        client_id=node_id,
        initial_battery=current_battery,
        device_class=device_class
    )
    
    # Consume battery for training
    round_info = battery_sim.consume(local_epochs)

    previous_battery_level = round_info.get("previous_battery_level")
    consumed = round_info.get("consumed")
    battery_level = round_info.get("battery_level")

    # Update state with the level before training, the consumed, and current level
    context.state["battery_sim_state"] = ConfigRecord({
        "previous_battery_level": previous_battery_level,
        "consumed": consumed,
        "battery_level": battery_level,
        "device_class": device_class,
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