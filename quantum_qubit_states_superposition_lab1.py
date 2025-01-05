from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

# Create a Quantum Circuit with 1 qubit
qc = QuantumCircuit(1)
qc.h(0)  # Apply Hadamard gate
qc.measure_all()  # Measure the qubit

# Visualize the circuit
print("Quantum Circuit:")
print(qc.draw())

# Execute the circuit using the AerSimulator
simulator = AerSimulator()
result = simulator.run(qc).result()
counts = result.get_counts()

# Plotting the results
plt.bar(counts.keys(), counts.values())
plt.title('Measurement Results')
plt.xlabel('State')
plt.ylabel('Counts')
plt.show()