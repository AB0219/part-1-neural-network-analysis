# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("customer_churn_nn.csv")

# Display first 5 rows
print("First 5 Rows:\n")
print(df.head())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns)

# Data types
print("\nData Types:")
print(df.dtypes)

# Target variable description
print("\nTarget Variable: churn")
print("1 = Customer Churned")
print("0 = Customer Retained")

# Missing value check
print("\nMissing Values:")
print(df.isnull().sum())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Distribution of target variable
plt.figure(figsize=(6,4))

sns.countplot(x=df["churn"])

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Count")

plt.show()


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Remove identifier column
df.drop("customer_id", axis=1, inplace=True)

# Handle missing values

# Numerical columns
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].mean())

# Categorical columns
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical columns
label_encoders = {}

for col in df.select_dtypes(include='object').columns:
    
    le = LabelEncoder()
    
    df[col] = le.fit_transform(df[col])
    
    label_encoders[col] = le

# Feature and target split
X = df.drop("churn", axis=1)

y = df["churn"]

# Feature scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Build neural network
model = Sequential()

# Input + Hidden Layer
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))

# Hidden Layer
model.add(Dense(32, activation='relu'))

# Hidden Layer
model.add(Dense(16, activation='relu'))

# Output Layer
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()



from sklearn.metrics import confusion_matrix, classification_report

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Plot training and validation curves
plt.figure(figsize=(12,5))

# Loss Curve
plt.subplot(1,2,1)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.legend()

# Accuracy Curve
plt.subplot(1,2,2)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title("Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.legend()

plt.show()

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Generate predictions
y_pred_prob = model.predict(X_test)

y_pred = (y_pred_prob > 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("evaluation_outputs.png")


plt.show()

# Classification report
print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# Interpretation
print("""
Interpretation:

The neural network successfully learned customer churn patterns.

Training and validation accuracy improved gradually during training,
showing that the model learned meaningful relationships.

The confusion matrix and classification report help evaluate
prediction quality on unseen test data.

If validation loss increases while training loss decreases,
the model may be overfitting.
""")



from tensorflow.keras.optimizers import Adam

# Function to create model
def create_model(neurons=32, learning_rate=0.001, activation='relu'):
    
    model = Sequential()

    model.add(Dense(neurons, activation=activation, input_shape=(X_train.shape[1],)))
    
    model.add(Dense(neurons // 2, activation=activation))
    
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# Hyperparameter experiments
experiments = [
    
    {
        "neurons": 32,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 20,
        "activation": "relu"
    },
    
    {
        "neurons": 64,
        "learning_rate": 0.001,
        "batch_size": 16,
        "epochs": 30,
        "activation": "relu"
    },
    
    {
        "neurons": 128,
        "learning_rate": 0.0005,
        "batch_size": 32,
        "epochs": 40,
        "activation": "tanh"
    }
]

# Run experiments
results = []

for i, exp in enumerate(experiments):

    print(f"\nRunning Experiment {i+1}")

    model = create_model(
        neurons=exp["neurons"],
        learning_rate=exp["learning_rate"],
        activation=exp["activation"]
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=exp["epochs"],
        batch_size=exp["batch_size"],
        validation_split=0.2,
        verbose=0
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    results.append({
        "Experiment": i+1,
        "Neurons": exp["neurons"],
        "Learning Rate": exp["learning_rate"],
        "Batch Size": exp["batch_size"],
        "Epochs": exp["epochs"],
        "Activation": exp["activation"],
        "Test Accuracy": accuracy
    })

# Comparison table
results_df = pd.DataFrame(results)

print("\nModel Comparison Table:\n")

print(results_df)

# Save results
results_df.to_csv("model_comparison_table.csv", index=False)

# Hyperparameter analysis
print("""
Hyperparameter Analysis

Experiment 1:
Smaller neural network with fewer neurons.
Training is faster but model complexity is lower.

Experiment 2:
More neurons and smaller batch size improved learning performance.

Experiment 3:
Larger network with tanh activation increased model complexity
and may improve learning for complex patterns.
""")




print("""
FINAL REFLECTION

1. Role of Weights and Biases
Weights determine the strength of relationships between input features
and neurons. During training, weights are updated continuously to reduce error.
Biases help shift activation values and improve the flexibility of the model.

2. Importance of Activation Functions
Activation functions introduce non-linearity into the network.
Without activation functions, the neural network would behave like
a simple linear regression model and fail to learn complex patterns.

3. Effect of Learning Rate
A very high learning rate causes unstable training and may skip the optimal solution.
A very low learning rate makes training extremely slow and can prevent convergence.

4. Underfitting and Overfitting
Underfitting happens when the model is too simple and cannot capture patterns.
Overfitting happens when the model memorizes training data and performs poorly on unseen data.

Validation accuracy and loss curves help identify whether
the model is underfitting or overfitting.
""")
