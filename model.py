# Function to calculate refined planetary aspects based on astrological principles
def calculate_aspects_refined(person1_planets, person2_planets):
    """Calculate aspects between two sets of planetary positions based on astrological compatibility."""
    aspects = []
    for p1, p2 in zip(person1_planets, person2_planets):
        angle = abs(p1 - p2)
        if angle > 180:  # Normalize angle to be within 0-180 degrees
            angle = 360 - angle

        # Aspect calculation with refined astrological values
        if angle <= 8:  # Conjunction (0° +/- 8°)
            aspects.append(1.0)
        elif 52 <= angle <= 68:  # Sextile (60° +/- 8°)
            aspects.append(0.6)
        elif 82 <= angle <= 98:  # Square (90° +/- 8°)
            aspects.append(-0.7)
        elif 112 <= angle <= 128:  # Trine (120° +/- 8°)
            aspects.append(0.9)
        elif 172 <= angle <= 188:  # Opposition (180° +/- 8°)
            aspects.append(-0.8)
        else:
            aspects.append(0)  # No major aspect

    return aspects

# Modified function to generate dataset with fine-tuned planetary aspects
def generate_extended_dataset_refined(num_samples):
    house_positions = np.random.rand(num_samples, 24)  # 12 houses for each person
    planetary_positions_1 = np.random.rand(num_samples, 10) * 360  # Positions of 10 planets (0-360 degrees)
    planetary_positions_2 = np.random.rand(num_samples, 10) * 360  # Positions of 10 planets for partner
    
    aspect_features = []
    for i in range(num_samples):
        aspects = calculate_aspects_refined(planetary_positions_1[i], planetary_positions_2[i])
        aspect_features.append(aspects)
    
    aspect_features = np.array(aspect_features)
    
    # Combine house positions and aspect features
    features = np.hstack((house_positions, aspect_features))
    
    # Binary labels (0 or 1) indicating successful marriage or not
    labels = np.random.randint(0, 2, size=num_samples)
    
    return features, labels

# Generate the new dataset with refined aspects
train_features, train_labels = generate_extended_dataset_refined(1000)
test_features, test_labels = generate_extended_dataset_refined(200)

# The rest of the code remains the same, using the new dataset with fine-tuned aspect values
train_dataset = MarriageAstrologyDataset(train_features, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = MarriageAstrologyDataset(test_features, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Updating model to handle 34 inputs (24 house positions + 10 planetary aspects)
input_size = 34  # 24 house features + 10 planetary aspect features
model = AstrologyMarriageModel(input_size)

# Train and evaluate the model as before
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, criterion, optimizer, train_loader)
evaluate_model(model, test_loader)
