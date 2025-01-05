from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

def deutsch_jozsa_oracle(circuit, n, constant=True):
    if constant:
        pass  # Constant function
    else:
        for i in range(n):
            circuit.cx(i, n)  # Balanced function

def deutsch_jozsa_algorithm(n, constant=True):
    circuit = QuantumCircuit(n + 1, n)
    circuit.x(n)  # Initialize output to |1⟩
    circuit.h(range(n + 1))  # Apply Hadamard gates
    deutsch_jozsa_oracle(circuit, n, constant)
    circuit.h(range(n))  # Apply Hadamard gates again
    circuit.measure(range(n), range(n))
    return circuit

# Set the number of qubits
n = 3
circuit = deutsch_jozsa_algorithm(n, constant=False)

# Draw and execute the circuit
print("Deutsch-Josza Circuit:")
print(circuit.draw())
simulator = AerSimulator()
result = simulator.run(circuit).result()
counts = result.get_counts()

# Plotting results
plt.bar(counts.keys(), counts.values(), color='skyblue')
plt.title('Deutsch-Josza Algorithm Results')
plt.xlabel('States')
plt.ylabel('Counts')
plt.show()
