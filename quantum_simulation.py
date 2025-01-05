# Install Qiskit and necessary dependencies
#!pip install qiskit qiskit-aer --quiet

# Import required libraries
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit import transpile
import matplotlib.pyplot as plt

# Step 1: Create a Quantum Circuit with 1 qubit and 1 classical bit
circuit = QuantumCircuit(1, 1)

# Step 2: Apply Hadamard Gate to the qubit
circuit.h(0)

# Step 3: Measure the qubit
circuit.measure(0, 0)

# Visualize the circuit
print("Quantum Circuit:")
print(circuit.draw())

# Step 4: Simulate the Circuit
simulator = AerSimulator()
compiled_circuit = transpile(circuit, simulator)
result = simulator.run(compiled_circuit, shots=1024).result()
counts = result.get_counts()

# Step 5: Plot the Results
print("Measurement Results:", counts)

# Ensure histogram displays correctly
fig, ax = plt.subplots(figsize=(8, 6))
plot_histogram(counts, ax=ax)
plt.title("Measurement Results")
plt.show()
