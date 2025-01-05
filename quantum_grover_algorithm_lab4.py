from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
import numpy as np

def grover_oracle(circuit, n, marked_state):
    """Oracle that marks the desired state."""
    # Convert marked state to binary
    binary_state = format(marked_state, f'0{n}b')
    for i, bit in enumerate(binary_state):
        if bit == '0':
            circuit.x(i)  # Flip to |1⟩ if 0
    circuit.h(n - 1)  # Apply Hadamard to the last qubit
    
    # Multi-controlled Toffoli implementation
    circuit.mcp(np.pi, list(range(n - 1)), n - 1)  # Controlled phase gate
    circuit.h(n - 1)  # Apply Hadamard to the last qubit
    
    for i, bit in enumerate(binary_state):
        if bit == '0':
            circuit.x(i)  # Flip back to |0⟩

def grover_algorithm(n, marked_state):
    circuit = QuantumCircuit(n)
    # Apply Hadamard gates to all qubits
    circuit.h(range(n))
    # Apply the oracle for Grover's algorithm
    grover_oracle(circuit, n, marked_state)
    # Apply diffusion operator
    circuit.h(range(n))
    circuit.x(range(n))
    circuit.h(n - 1)
    circuit.mcp(np.pi, list(range(n - 1)), n - 1)  # Controlled phase gate
    circuit.h(n - 1)
    circuit.x(range(n))
    circuit.h(range(n))
    circuit.measure_all()
    return circuit

# Set the number of qubits and the marked state
n = 3  # Number of qubits
marked_state = 5  # Example: |101⟩ is marked

circuit = grover_algorithm(n, marked_state)

# Draw and execute the circuit
print("Grover's Circuit:")
print(circuit.draw())

simulator = AerSimulator()
result = simulator.run(circuit).result()
counts = result.get_counts()

# Plotting results
plt.bar(counts.keys(), counts.values(), color='skyblue')
plt.title('Grover\'s Algorithm Results')
plt.xlabel('States')
plt.ylabel('Counts')
plt.show()