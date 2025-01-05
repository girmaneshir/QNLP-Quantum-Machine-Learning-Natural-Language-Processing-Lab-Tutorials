from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

# Create a Quantum Circuit with 3 qubits and 2 classical bits
qc = QuantumCircuit(3, 2)

# Step 1: Create an entangled pair of qubits (qubit 1 and qubit 2)
qc.h(1)          # Apply Hadamard gate to qubit 1
qc.cx(1, 2)      # Apply CNOT gate (control: qubit 1, target: qubit 2)

# Step 2: Prepare the state to be teleported (qubit 0)
# For demonstration, let's teleport the state |+⟩ = (|0⟩ + |1⟩) / √2
qc.h(0)          # Apply Hadamard gate to prepare state |+⟩

# Step 3: Perform Bell state measurement on qubit 0 and qubit 1
qc.cx(0, 1)      # CNOT from qubit 0 to qubit 1
qc.h(0)          # Hadamard on qubit 0
qc.measure(0, 0) # Measure qubit 0
qc.measure(1, 1) # Measure qubit 1

# Step 4: Apply corrections based on measurement results on qubit 2
qc.x(2).c_if(qc.clbits[0], 1)  # Apply X gate if qubit 0 measurement is |1⟩
qc.z(2).c_if(qc.clbits[1], 1)  # Apply Z gate if qubit 1 measurement is |1⟩

# Visualize the circuit
print("Quantum Teleportation Circuit:")
print(qc.draw())

# Execute the circuit using the AerSimulator directly
simulator = AerSimulator()
result = simulator.run(qc).result()
counts = result.get_counts()

# Plotting the results
plt.bar(counts.keys(), counts.values())
plt.title('Measurement Results for Quantum Teleportation')
plt.xlabel('State')
plt.ylabel('Counts')
plt.show()