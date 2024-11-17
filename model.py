# Install necessary libraries
!pip install pyswisseph
!pip install flatlib

# Import required libraries
import swisseph as swe
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# Function to calculate planetary positions
def calculate_planet_positions(date, time, latitude, longitude):
    try:
        dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
        jd = swe.julday(dt.year, dt.month, dt.day, dt.hour + dt.minute / 60.0)
        swe.set_topo(longitude, latitude, 0)

        planet_positions = {}
        planets = [
            swe.SUN, swe.MOON, swe.MERCURY, swe.VENUS, swe.MARS,
            swe.JUPITER, swe.SATURN, swe.URANUS, swe.NEPTUNE, swe.PLUTO
        ]
        for planet in planets:
            position, _ = swe.calc_ut(jd, planet)
            planet_positions[swe.get_planet_name(planet)] = position[0]

        return planet_positions
    except Exception as e:
        print(f"Error calculating planetary positions: {e}")
        return {}

# Custom Dataset for Marriage Astrology
class MarriageAstrologyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        person1_data, person2_data, outcome = data_point['person1'], data_point['person2'], data_point['outcome']

        # Calculate planetary positions
        planet_positions1 = calculate_planet_positions(*person1_data)
        planet_positions2 = calculate_planet_positions(*person2_data)

        # Extract features as lists
        planets = [
            swe.SUN, swe.MOON, swe.MERCURY, swe.VENUS, swe.MARS,
            swe.JUPITER, swe.SATURN, swe.URANUS, swe.NEPTUNE, swe.PLUTO
        ]
        features1 = [planet_positions1.get(swe.get_planet_name(planet), 0.0) for planet in planets]
        features2 = [planet_positions2.get(swe.get_planet_name(planet), 0.0) for planet in planets]

        # Combine features and outcome
        features = features1 + features2
        features = torch.tensor(features, dtype=torch.float32)
        outcome = torch.tensor(outcome, dtype=torch.float32)

        return features, outcome

# Define the model
class AstrologyMarriageModel(nn.Module):
    def __init__(self, input_size):
        super(AstrologyMarriageModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Training function
def train_model(model, criterion, optimizer, dataloader, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for features, labels in dataloader:
            features, labels = features.float(), labels.float().unsqueeze(1)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader):.4f}")

# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    true_labels = []
    predictions = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            predicted_labels = torch.round(outputs)
            true_labels.extend(labels.numpy())
            predictions.extend(predicted_labels.numpy())

    acc = accuracy_score(true_labels, predictions)
    conf_matrix = confusion_matrix(true_labels, predictions)
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Confusion Matrix:\n{conf_matrix}")

# Main script
if __name__ == "__main__":
    # Replace `dataset_of_100` with your actual data
    dataset_of_100 = [
        # Example data structure: {'person1': (date, time, lat, long), 'person2': (date, time, lat, long), 'outcome': 1/0}
        # Add your dataset here
    ]

    # Split data into train and test sets
    train_data = dataset_of_100[:70]
    test_data = dataset_of_100[70:]

    train_dataset = MarriageAstrologyDataset(train_data)
    test_dataset = MarriageAstrologyDataset(test_data)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize the model, loss function, and optimizer
    input_size = len(train_dataset[0][0])
    model = AstrologyMarriageModel(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print("Training the model...")
    train_model(model, criterion, optimizer, train_dataloader)

    # Evaluate the model
    print("\nEvaluating the model...")
    evaluate_model(model, test_dataloader)
