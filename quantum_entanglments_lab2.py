from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

# Create a Quantum Circuit with 2 qubits
qc = QuantumCircuit(2)
qc.h(0)  # Apply Hadamard gate to the first qubit
qc.cx(0, 1)  # Apply CNOT gate (control on qubit 0, target on qubit 1)
qc.measure_all()  # Measure both qubits

# Visualize the circuit
print("Quantum Circuit:")
print(qc.draw())

# Execute the circuit using the AerSimulator
simulator = AerSimulator()
result = simulator.run(qc).result()
counts = result.get_counts()

# Plotting the results
plt.bar(counts.keys(), counts.values())
plt.title('Measurement Results for Entangled States')
plt.xlabel('State')
plt.ylabel('Counts')
plt.show()