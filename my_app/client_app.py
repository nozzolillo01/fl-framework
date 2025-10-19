"""my-app: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict, Error
from flwr.clientapp import ClientApp

from my_app.task import Net, load_data
from my_app.task import test as test_fn
from my_app.task import train as train_fn
from my_app.battery_simulator import BatterySimulator

# Flower ClientApp
app = ClientApp()

# Global battery simulator (persistent across rounds)
battery_sim = None


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data with battery simulation."""
    global battery_sim
    
    # Initialize battery simulator on first call
    node_id = context.node_id

    if battery_sim is None:
        battery_sim = BatterySimulator(client_id=node_id)

    local_epochs = context.run_config["local-epochs"]

    # Check if training was completed successfully
    if not battery_sim.can_train(local_epochs):
        # Client doesn't have enough battery - consume what's left and report error
        battery_sim.update(local_epochs)

        # Create error with code and reason
        error = Error(
            code=1,  # Custom error code for battery exhaustion
            reason=f"ran out of battery"
        )

        return Message(error=error, reply_to=msg)

    round_info = battery_sim.update(local_epochs)

    # Training completed successfully - proceed with actual training
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
        # Battery metrics from the training attempt
        "device_class": round_info.get("device_class", "unknown"),
        "battery_level": round_info.get("battery_level", 0.0),
        "previous_battery_level": round_info.get("previous_battery_level", 0.0),
        "battery_consumed": round_info.get("consumed", 0.0),
        "battery_recharged": round_info.get("recharged", 0.0),
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
