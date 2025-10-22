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
    """Report current battery status."""
    node_id = context.node_id

    # Initialize battery simulator for this client if not already present in context state
    if "battery_sim_state" not in context.state:
        # First initialization - create with deterministic device class
        battery_sim = BatterySimulator(client_id=node_id)

        # Save initial battery state to context
        context.state["battery_sim_state"] = ConfigRecord({
            "battery_level": battery_sim.battery_level,
            "device_class": battery_sim.device_class,
        })
    else:
        # Restore from saved state
        state = context.state["battery_sim_state"]
        battery_sim = BatterySimulator(client_id=node_id)
        battery_sim.battery_level = state["battery_level"]
        battery_sim.device_class = state["device_class"]
    
    # ALL clients recharge at the beginning of each round (whether selected or not)
    recharge_info = battery_sim.recharge()
    
    # Save updated battery state after recharge
    context.state["battery_sim_state"] = ConfigRecord({
        "battery_level": battery_sim.battery_level,
        "device_class": battery_sim.device_class,
    })
    
    # Return current battery status for this specific client (including recharge info)
    metrics = MetricRecord({
        "battery_level": battery_sim.battery_level,
        "device_class": battery_sim.device_class,
        "battery_recharged": recharge_info.get("recharged", 0.0),
        "previous_battery_level": recharge_info.get("previous_battery_level", 0.0),
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

    # Restore battery simulator state (should have been created in query(), but defensive programming)
    if "battery_sim_state" not in context.state:
        battery_sim = BatterySimulator(client_id=node_id)
        context.state["battery_sim_state"] = ConfigRecord({
            "battery_level": battery_sim.battery_level,
            "device_class": battery_sim.device_class,
            "client_id": node_id
        })
    else:
        # Restore from saved state
        state = context.state["battery_sim_state"]
        battery_sim = BatterySimulator(client_id=node_id)
        battery_sim.battery_level = state["battery_level"]
        battery_sim.device_class = state["device_class"]
    
    # Consume battery for training (only selected clients do this)
    round_info = battery_sim.consume(local_epochs)

    # Save updated battery state back to context
    context.state["battery_sim_state"] = ConfigRecord({
        "battery_level": battery_sim.battery_level,
        "device_class": battery_sim.device_class,
        "client_id": node_id
    })

    if not round_info["training_completed"]:
        #battery is insufficient, return error
        error = Error(
            code=1,
            reason=f"ran out of battery"
        )
        return Message(error=error, reply_to=msg)

    # battery is sufficient - proceed with actual training
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
    
    # Return updated model and metrics, including battery info
    model_record = ArrayRecord(model.state_dict())
    metrics = MetricRecord({
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "num-examples": len(trainloader.dataset),
        # Battery metrics from the training attempt (consumption only, recharge happens in query)
        "device_class": round_info.get("device_class", "unknown"),
        "battery_level": round_info.get("battery_level", 0.0),
        "previous_battery_level": round_info.get("previous_battery_level", 0.0),
        "battery_consumed": round_info.get("consumed", 0.0),
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