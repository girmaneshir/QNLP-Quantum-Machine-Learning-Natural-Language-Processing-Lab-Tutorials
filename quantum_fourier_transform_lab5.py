from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

def qft(circuit, n):
    """Apply the Quantum Fourier Transform to the first n qubits in the circuit."""
    for j in range(n):
        circuit.h(j)  # Apply Hadamard gate
        for k in range(j + 1, n):
            angle = 2 * 3.141592653589793 / (2 ** (k - j + 1))
            circuit.cp(angle, k, j)  # Apply controlled phase rotation

def create_qft_circuit(n):
    """Create a quantum circuit for QFT."""
    qc = QuantumCircuit(n)

    # Step 1: Prepare the input state |x>
    qc.x(0)  # Example: prepare |001> (the state |1>)

    # Step 2: Apply QFT
    qft(qc, n)

    # Step 3: Measure the qubits
    qc.measure_all()
    
    return qc

def prepare_plot(counts):
    """Prepare a bar plot of the measurement results."""
    plt.bar(counts.keys(), counts.values())
    plt.title('Measurement Results of Quantum Fourier Transform')
    plt.xlabel('States')
    plt.ylabel('Counts')
    plt.xticks(rotation=45)
    plt.show()

# Example of usage (not executed here):
# To run the QFT and plot results, you would execute the following code:
qc = create_qft_circuit(3)
# Draw and execute the circuit
print("Grover's Circuit:")
print(qc.draw())
# result = execute(qc, backend=simulator, shots=1024).result()
# counts = result.get_counts()
# prepare_plot(counts)
simulator = AerSimulator()
result = simulator.run(qc).result()
counts = result.get_counts()

# Plotting the results
plt.bar(counts.keys(), counts.values())
plt.xlabel('State')
plt.ylabel('Counts')
plt.xticks(rotation=45)
plt.show()