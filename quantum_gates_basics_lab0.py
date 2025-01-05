from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

# Function to visualize the circuit and state
def visualize_circuit_and_state(qc, initial_state, title):
    print(title)
    print(qc.draw())
    qc.measure_all()  # Measure all qubits
    simulator = AerSimulator()
    result = simulator.run(qc).result()
    counts = result.get_counts()
    plt.bar(counts.keys(), counts.values(), color='skyblue')
    plt.title(title)
    plt.xlabel('States')
    plt.ylabel('Counts')
    plt.show()

# Create a Quantum Circuit with 1 qubit for Hadamard gate
qc_h = QuantumCircuit(1)
qc_h.h(0)  # Apply Hadamard gate

# Visualize Hadamard Gate
visualize_circuit_and_state(qc_h, '|0⟩', 'Hadamard Gate')

# Create a Quantum Circuit with 1 qubit for Pauli-X gate
qc_x = QuantumCircuit(1)
qc_x.x(0)  # Apply Pauli-X gate

# Visualize Pauli-X Gate
visualize_circuit_and_state(qc_x, '|0⟩', 'Pauli-X Gate')

# Create a Quantum Circuit with 1 qubit for Pauli-Z gate
qc_z = QuantumCircuit(1)
qc_z.z(0)  # Apply Pauli-Z gate

# Visualize Pauli-Z Gate
visualize_circuit_and_state(qc_z, '|0⟩', 'Pauli-Z Gate')