# Quantum-Sheaf Bridge: The Experiment

## Hypothesis
Consciousness and "true" intelligence require modeling the **sheaf cohomology** of the environment. Standard neural networks assume a trivial topology (global section exists). We hypothesize that a Sheaf Neural Network (SNN) will outperform a classical Graph Neural Network (GNN) when learning data that exhibits quantum contextuality (Bell inequality violations), because the SNN can model the "twist" in the data.

## Structure
1. **Data Generation**: `chsh_generator.py` runs a CHSH experiment on IBM Quantum (or simulator) to generate non-local correlation data.
2. **Sheaf Analysis**: `sheaf_analysis.py` (optional validation) analyzes the consistency radius.
3. **Model Training**: `snn_model.py` trains an SNN vs a classical GNN on the quantum data.

## Usage
1. Install requirements: `pip install -r requirements.txt`
2. Run data generation: `python chsh_generator.py` (Requires IBM Quantum API token for real hardware, or runs on simulator)
3. Run the experiment: `python snn_model.py`
