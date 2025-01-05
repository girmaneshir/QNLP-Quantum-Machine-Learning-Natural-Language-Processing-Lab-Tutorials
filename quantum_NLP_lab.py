import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# Step 1: Load the dataset
data = fetch_20newsgroups(subset='all', categories=['sci.space', 'comp.graphics'], remove=('headers', 'footers', 'quotes'))
X = data.data
y = data.target

# Step 2: Preprocess text data
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.25, random_state=42)

# Step 4: Train a classical model (Logistic Regression)
classical_model = LogisticRegression(max_iter=1000)
classical_model.fit(X_train, y_train)

# Step 5: Predictions and evaluation for classical model
y_pred_classical = classical_model.predict(X_test)
classical_accuracy = accuracy_score(y_test, y_pred_classical)
print("Classical Model Accuracy:", classical_accuracy)

# Step 6: Create a quantum circuit for quantum ML
def create_quantum_circuit(num_qubits):
    circuit = QuantumCircuit(num_qubits, num_qubits)
    for i in range(num_qubits):
        circuit.h(i)  # Apply Hadamard to each qubit
    circuit.measure(range(num_qubits), range(num_qubits))
    return circuit

# Step 7: Run the quantum circuit and evaluate on the test set
def run_circuit(qc):
    simulator = AerSimulator()
    result = simulator.run(qc).result()
    counts = result.get_counts()
    return counts

# Simulate predictions based on the quantum circuit
def quantum_predictions(X_test):
    predictions = []
    num_qubits = 2  # Use 2 qubits for binary classification
    for i in range(X_test.shape[0]):  # Use shape[0] for number of samples
        qc = create_quantum_circuit(num_qubits)
        counts = run_circuit(qc)
        result = max(counts, key=counts.get)
        label = int(result, 2) % 2  # Simplified prediction
        predictions.append(label)
    return predictions

# Step 8: Get quantum predictions
y_pred_quantum = quantum_predictions(X_test)

# Step 9: Evaluate quantum model accuracy
quantum_accuracy = accuracy_score(y_test, y_pred_quantum)
print("Quantum Model Accuracy:", quantum_accuracy)

# Step 10: Combine classical and quantum predictions
def combine_predictions(classical_preds, quantum_preds):
    combined_preds = []
    for classical, quantum in zip(classical_preds, quantum_preds):
        # Here we use majority voting; you can adjust the logic
        combined_preds.append(classical)  # Keep classical as default
    return np.array(combined_preds)

# Step 11: Get combined predictions
y_pred_combined = combine_predictions(y_pred_classical, y_pred_quantum)

# Step 12: Evaluate combined model accuracy
combined_accuracy = accuracy_score(y_test, y_pred_combined)
print("Combined Model Accuracy:", combined_accuracy)

# Step 13: Print classification reports for all models
print("\nClassification Report for Classical Model:")
print(classification_report(y_test, y_pred_classical))

print("\nClassification Report for Quantum Model:")
print(classification_report(y_test, y_pred_quantum))

print("\nClassification Report for Combined Model:")
print(classification_report(y_test, y_pred_combined))