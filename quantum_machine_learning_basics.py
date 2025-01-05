import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# Step 1: Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 3: Train a classical model (Logistic Regression)
classical_model = LogisticRegression(max_iter=1000)
classical_model.fit(X_train, y_train)

# Step 4: Predictions and evaluation for classical model
y_pred_classical = classical_model.predict(X_test)
classical_accuracy = accuracy_score(y_test, y_pred_classical)
print("Classical Model Accuracy:", classical_accuracy)

# Step 5: Create a quantum circuit for quantum ML
def create_quantum_circuit(num_qubits):
    circuit = QuantumCircuit(num_qubits, num_qubits)
    for i in range(num_qubits):
        circuit.h(i)  # Apply Hadamard to each qubit
    circuit.measure(range(num_qubits), range(num_qubits))  # Measure each qubit
    return circuit

# Step 6: Run the quantum circuit and evaluate on the test set
def run_circuit(qc):
    simulator = AerSimulator()
    result = simulator.run(qc).result()  # Directly run the circuit
    counts = result.get_counts()
    return counts

# Simulate predictions based on the quantum circuit
def quantum_predictions(X_test):
    predictions = []
    for i in range(len(X_test)):
        num_qubits = 2  # Use 2 qubits for binary classification
        qc = create_quantum_circuit(num_qubits)
        counts = run_circuit(qc)
        result = max(counts, key=counts.get)  # Get the most frequent measurement result
        label = int(result, 2) % 2  # Simplified prediction
        predictions.append(label)
    return predictions

# Step 7: Get quantum predictions
y_pred_quantum = quantum_predictions(X_test)

# Step 8: Evaluate quantum model accuracy
quantum_accuracy = accuracy_score(y_test, y_pred_quantum)
print("Quantum Model Accuracy:", quantum_accuracy)

# Step 9: Combine classical and quantum predictions
def combine_predictions(classical_preds, quantum_preds):
    combined_preds = []
    for classical, quantum in zip(classical_preds, quantum_preds):
        # Use classical predictions as default
        combined_preds.append(classical)
    return np.array(combined_preds)

# Step 10: Get combined predictions
y_pred_combined = combine_predictions(y_pred_classical, y_pred_quantum)

# Step 11: Evaluate combined model accuracy
combined_accuracy = accuracy_score(y_test, y_pred_combined)
print("Combined Model Accuracy:", combined_accuracy)

# Step 12: Print classification reports for all models
print("\nClassification Report for Classical Model:")
print(classification_report(y_test, y_pred_classical))

print("\nClassification Report for Quantum Model:")
print(classification_report(y_test, y_pred_quantum))

print("\nClassification Report for Combined Model:")
print(classification_report(y_test, y_pred_combined))

# Step 13: Plotting results
def plot_results(classical_accuracy, quantum_accuracy, combined_accuracy):
    models = ['Classical ML', 'Quantum ML', 'Combined Model']
    accuracies = [classical_accuracy, quantum_accuracy, combined_accuracy]

    plt.bar(models, accuracies, color=['blue', 'orange', 'green'])
    plt.title('Model Accuracies Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # Set y-axis limits to [0,1]
    plt.show()

# Plot the results
plot_results(classical_accuracy, quantum_accuracy, combined_accuracy)
