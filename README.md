# Energy-Aware Federated Learning Framework

> **A Flower extension for experimenting with battery-aware client selection strategies in Federated Learning**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Flower](https://img.shields.io/badge/Flower-1.22.0+-green.svg)](https://flower.ai/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## 📋 Overview

This project extends the [Flower](https://flower.ai/) Federated Learning framework to enable **energy-aware client selection** strategies for Smart Agriculture applications. It provides a modular architecture for simulating battery behavior in edge devices and comparing different client selection policies based on energy consumption, fairness, and model accuracy.

### 🎯 Key Features

- **🔋 Realistic Battery Simulation**: Simulate different device classes (low/mid/high power) with energy harvesting capabilities
- **📊 Multiple Selection Strategies**: 
  - `random`: Uniform random selection
  - `battery_aware`: Weighted selection based on battery levels
  - `efficiency_aware`: Selection based on battery level/consumption ratio
  - `all_available`: Full participation
- **📈 Comprehensive Metrics Tracking**: 
  - Training accuracy, loss, and convergence
  - Battery consumption and fairness (Jain's index)
  - Client participation patterns
  - Energy efficiency metrics
- **🔗 Weights & Biases Integration**: Automatic logging and visualization of all metrics
- **📊 Design of Experiments (DOE) Support**: Automated CSV export of experimental results for statistical analysis
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

Execute a local simulation with 20 virtual clients:

```bash
flwr run .
```

The simulation will:
- Train a CNN on PlantVillage10 dataset (Dirichlet partitioning, α=0.5)
- Run 5 federated rounds by default
- Log metrics to Weights & Biases
- Save the final model as `final_model.pt`
- Export DOE results to `exp4.csv` (snapshots every 10 rounds + final round)

---

## ⚙️ Configuration

All hyperparameters are configured in `pyproject.toml`:

```toml
[tool.flwr.app.config]
# Federated Learning settings
num-server-rounds = 5           # Number of FL rounds
selection-fraction = 0.5        # Fraction of clients selected per round
local-epochs = 5                # Local training epochs per client
lr = 0.01                       # Learning rate

# Client selection strategy
selection-strategy = "random"  # Options: "random", "battery_aware", "efficiency_aware", "all_available"

# Battery-aware and efficiency-aware strategy parameters
alpha = 2.0                     # Battery weight exponent (for battery_aware strategy)
min-battery-threshold = 0.2     # Minimum battery level to be eligible (0.0-1.0)

[tool.flwr.federations.local-simulation]
options.num-supernodes = 20     # Number of virtual clients
```

### Example Configurations

**Aggressive Energy Conservation:**
```toml
selection-strategy = "battery_aware"
alpha = 3.0
min-battery-threshold = 0.3
selection-fraction = 0.3
```

**Efficiency-Based Selection:**
```toml
selection-strategy = "efficiency_aware"
min-battery-threshold = 0.2
selection-fraction = 0.5
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

## 🔧 Overriding Parameters via CLI

You can override the parameters defined in `pyproject.toml` directly from the command line using the `--run-config` or `--federation-config` flags.

### Overriding Run Configuration Parameters

Use the `--run-config` flag to specify the parameters to override:

```bash
flwr run . --run-config '"num-server-rounds=5" "selection-fraction=0.7"'
```

This command will execute a simulation with:
- **5 rounds** of Federated Learning
- **70%** of available clients selected per round


### Overriding Federation Configuration Parameters

Use the `--federation-config` flag to override federation-related parameters:

```bash
flwr run . --federation-config '"options.num-supernodes=20"'
```

This command will execute a simulation with **20 virtual clients**.

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
- **`FleetManager`**: Tracks battery levels, consumption, and participation for all clients (server-side)
- **`BatterySimulator`**: Simulates energy consumption and harvesting per device class (client-side)
  - **Device Class 0 (low_power)**: Consumption 1.5-2.5%, Harvesting 0-1.5%
  - **Device Class 1 (mid_power)**: Consumption 2.5-3.5%, Harvesting 0-2.5%
  - **Device Class 2 (high_power)**: Consumption 3.5-4.5%, Harvesting 0-3.5%
- **`Selection Strategies`**: Modular functions implementing different selection policies

---

## 📊 Experiments & Results

### Tracked Metrics

**Model Performance:**
- Train loss and accuracy (federated training, aggregation of training metrics in each client)
- Eval loss and accuracy (federated evaluation, aggregation of test metrics in each client)
- Centralized loss and accuracy (server-side evaluation on full test dataset)

**Energy & Fairness:**
- Total cumulative energy consumption
- Battery level distribution (min, max, avg) before each round
- Number of dead clients per round (selected but failed due to battery exhaustion)
- Jain's fairness index (participation equity across all clients)

**Client Participation:**
- Selected vs. responded clients
- Client-level details for all clients:
  - Battery level (current and previous)
  - Energy consumed and recharged
  - Selection probability
  - Device class
  - Participation count
  - Completion status (not_selected, completed, failed)

### Viewing Results

**Weights & Biases Dashboard:**
Results are automatically logged to Weights & Biases project `exp_4`. Access your dashboard at:
```
https://wandb.ai/<your-username>/exp_4
```

**Design of Experiments (DOE) Export:**
Experimental results are automatically exported to `exp4.csv` with:
- Snapshots saved every 10 rounds
- Final round always saved
- Includes all configuration factors and response variables
- Ready for statistical analysis (ANOVA, regression, etc.)

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
       "efficiency_aware": select_efficiency_aware,
       "all_available": select_all_available,
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
    0: {  # low_power_device
        "consumption_range": (0.015, 0.025),  # Consumption per epoch
        "harvesting_range": (0.0, 0.015),     # Harvesting per round
    },
    1: {  # mid_power_device
        "consumption_range": (0.025, 0.035),
        "harvesting_range": (0.0, 0.025),
    },
    2: {  # high_power_device
        "consumption_range": (0.035, 0.045),
        "harvesting_range": (0.0, 0.035),
    },
}
```

---


## 📖 Research Context

This framework is designed for research on **energy-efficient Federated Learning** in resource-constrained environments, with a focus on **Smart Agriculture** applications (IoT sensors, edge devices, agricultural monitoring systems). It enables investigation of:

- Trade-offs between model accuracy and energy efficiency
- Fairness in client participation under energy constraints
- Impact of heterogeneous device capabilities on FL performance
- Energy harvesting strategies in FL scenarios
- Client selection policies for battery-powered agricultural sensors

---

## 📚 Additional Resources

### Flower Documentation
- [Flower Docs](https://flower.ai/docs/)
- [Simulation Guide](https://flower.ai/docs/framework/how-to-run-simulations.html)
- [Strategy Development](https://flower.ai/docs/framework/ref-api/flwr.server.strategy.html)

### Related Thesis
- **FedLEAF: An Energy-Aware Federated Learning Framework for Client Selection in Smart Agriculture**: Angelo Andrea Nozzolillo (2025)


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
- PlantVillage10 dataset via [Hugging Face Datasets](https://huggingface.co/datasets/nozzolillo01/PlantVillage10)
- Experiment tracking with [Weights & Biases](https://wandb.ai/)

---

## 📧 Contact

**Author**: Angelo Andrea Nozzolillo  
**Repository**: [github.com/nozzolillo01/fl-framework](https://github.com/nozzolillo01/fl-framework)  
**Issues**: [GitHub Issues](https://github.com/nozzolillo01/fl-framework/issues)

---

**⭐ If you find this framework useful for your research, please consider starring the repository!**
