# Energy-Aware Federated Learning Framework

> **A Flower extension for experimenting with battery-aware client selection strategies in Federated Learning**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Flower](https://img.shields.io/badge/Flower-1.22.0+-green.svg)](https://flower.ai/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## 📋 Overview

This project extends the [Flower](https://flower.ai/) Federated Learning framework to enable **energy-aware client selection** strategies. It provides a modular architecture for simulating battery behavior in edge devices and comparing different client selection policies based on energy consumption, fairness, and model accuracy.

### 🎯 Key Features

- **🔋 Realistic Battery Simulation**: Simulate different device classes (low/mid/high power) with energy harvesting capabilities
- **📊 Multiple Selection Strategies**: 
  - `random`: Uniform random selection
  - `battery_aware`: Weighted selection based on battery levels
  - `all_available`: Full participation
- **📈 Comprehensive Metrics Tracking**: 
  - Training accuracy, loss, and convergence
  - Battery consumption and fairness (Jain's index)
  - Client participation patterns
  - Energy efficiency metrics
- **🔗 Weights & Biases Integration**: Automatic logging and visualization of all metrics
- **⚙️ Highly Configurable**: All parameters accessible via `pyproject.toml`

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/nozzolillo01/fl-framework.git
   cd fl-framework
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   ```

### Run a Simulation

Execute a local simulation with 10 virtual clients:

```bash
flwr run .
```

The simulation will:
- Train a CNN on CIFAR-10 dataset (Dirichlet partitioning, α=0.5)
- Run 3 federated rounds by default
- Log metrics to Weights & Biases
- Save the final model as `final_model.pt`

---

## ⚙️ Configuration

All hyperparameters are configured in `pyproject.toml`:

```toml
[tool.flwr.app.config]
# Federated Learning settings
num-server-rounds = 3           # Number of FL rounds
selection-fraction = 0.5        # Fraction of clients selected per round
local-epochs = 5                # Local training epochs per client
lr = 0.01                       # Learning rate

# Client selection strategy
selection-strategy = "battery_aware"  # Options: "random", "battery_aware", "all_available"

# Battery-aware strategy parameters
alpha = 2.0                     # Battery weight exponent (higher = more preference for high battery)
min-battery-threshold = 0.2     # Minimum battery level to be eligible (0.0-1.0)

[tool.flwr.federations.local-simulation]
options.num-supernodes = 10     # Number of virtual clients
```

### Example Configurations

**Aggressive Energy Conservation:**
```toml
selection-strategy = "battery_aware"
alpha = 3.0
min-battery-threshold = 0.3
selection-fraction = 0.3
```

**Baseline Random Selection:**
```toml
selection-strategy = "random"
selection-fraction = 0.5
```

**Full Participation:**
```toml
selection-strategy = "all_available"
```

---

## 🏗️ Architecture

```
my_app/
├── server_app.py              # Server orchestration and strategy initialization
├── client_app.py              # Client-side training and evaluation logic
├── strategy.py                # CustomFedAvg with battery-aware client selection
├── selection_strategies.py    # Pluggable selection strategy implementations
├── battery_simulator.py       # Battery simulation and fleet management
├── task.py                    # Model definition and training/testing functions
└── wandb_utils.py             # Weights & Biases logging utilities
```

### Key Components

- **`CustomFedAvg`**: Extends Flower's `FedAvg` strategy with custom client selection
- **`FleetManager`**: Tracks battery levels, consumption, and participation for all clients
- **`BatterySimulator`**: Simulates energy consumption and harvesting per device class
- **`Selection Strategies`**: Modular functions implementing different selection policies

---

## 📊 Experiments & Results

### Tracked Metrics

**Model Performance:**
- Train loss and accuracy  (federated training, aggregation of training metrics in each client)
- Eval  loss and accuracy  (federated evaluation, aggregation of test metrics in each client)
- Test  loss and accuracy  (centralized evaluation, Server-side test on a fresh dataset)

**Energy & Fairness:**
- Total energy consumption
- Battery level distribution (min, max, avg)
- Number of dead clients per round
- Jain's fairness index

**Client Participation:**
- Selected vs. active clients
- Client-level details (battery, selection probability, device class)

### Viewing Results

Results are automatically logged to Weights & Biases. Access your dashboard at:
```
https://wandb.ai/<your-username>/fl-client-selection-framework
```

---

## 🧪 Running Experiments

### Experiment Template

```bash
# Edit pyproject.toml with desired configuration
nano pyproject.toml

# Run simulation
flwr run .

# Results saved to:
# - final_model.pt (model weights)
# - wandb/ (experiment logs)
```

---

## 🔬 Extending the Framework

### Adding a New Selection Strategy

1. **Define the strategy function** in `selection_strategies.py`:
   ```python
   def select_my_strategy(
       available_nodes: list[int], 
       fleet_manager: FleetManager, 
       params: dict
   ) -> tuple[list[int], dict[int, float]]:
       """Your custom selection logic."""
       # Implementation
       return selected_nodes, prob_map
   ```

2. **Register it** in the `STRATEGIES` dictionary:
   ```python
   STRATEGIES = {
       "random": select_random,
       "battery_aware": select_battery_aware,
       "my_strategy": select_my_strategy,  # Add here
   }
   ```

3. **Use it** in `pyproject.toml`:
   ```toml
   selection-strategy = "my_strategy"
   ```

### Customizing Battery Behavior

Modify `BatterySimulator.DEVICE_CLASSES` in `battery_simulator.py`:

```python
DEVICE_CLASSES = {
    "ultra_low_power": {
        "consumption_range": (0.005, 0.010), # Change the values
        "harvesting_range": (0.0, 0.020),
    },
    # Add more classes...
}
```

---


## 📖 Research Context

This framework is designed for research on **energy-efficient Federated Learning** in resource-constrained environments (IoT, mobile devices, edge computing). It enables investigation of:

- Trade-offs between model accuracy and energy efficiency
- Fairness in client participation under energy constraints
- Impact of heterogeneous device capabilities
- Energy harvesting strategies in FL

---

## 📚 Additional Resources

### Flower Documentation
- [Flower Docs](https://flower.ai/docs/)
- [Simulation Guide](https://flower.ai/docs/framework/how-to-run-simulations.html)
- [Strategy Development](https://flower.ai/docs/framework/ref-api/flwr.server.strategy.html)

### Related Thesis
- **Enery-Aware Federated Learing Frameworl**: Angelo Andrea Nozzolillo, "TITOLO TESI" (2025)


### Community
- [Flower Slack](https://flower.ai/join-slack/)
- [Flower Discuss](https://discuss.flower.ai/)
- [GitHub Issues](https://github.com/nozzolillo01/fl-framework/issues)

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built on [Flower](https://flower.ai/) framework
- CIFAR-10 dataset via [Hugging Face Datasets](https://huggingface.co/datasets/uoft-cs/cifar10)
- Experiment tracking with [Weights & Biases](https://wandb.ai/)

---

## 📧 Contact

**Author**: Angelo Andrea Nozzolillo  
**Repository**: [github.com/nozzolillo01/fl-framework](https://github.com/nozzolillo01/fl-framework)  
**Issues**: [GitHub Issues](https://github.com/nozzolillo01/fl-framework/issues)

---

**⭐ If you find this framework useful for your research, please consider starring the repository!**
