from sklearn.linear_model import LinearRegression
import numpy as np

# Replace these with your actual dataset arrays
train_images = [...]  # numpy arrays of images
train_damage_extent = [...]  # numeric values (e.g., float), representing damage extent
test_images = [...]

def prepare_labels(data):
    # Directly convert damage extent to numpy array
    labels = np.array(data)
    return labels

def prepare_features(data):
    features = []
    for image in data:
        # Example statistical features
        mean_value = np.mean(image)
        std_value = np.std(image)
        max_value = np.max(image)
        min_value = np.min(image)
        features.append([mean_value, std_value, max_value, min_value])
    return np.array(features)

# Prepare features and labels for regression
features = prepare_features(train_images)
labels = prepare_labels(train_damage_extent)

# Train regression model
regression_model = LinearRegression()
regression_model.fit(features, labels)

# Estimate repair costs
test_features = prepare_features(test_images)
predicted_damage_extent = regression_model.predict(test_features)

# Cost calculation (update factor as per your requirement)
cost_factor = 10000  # example: ₹10,000 per unit extent
estimated_costs = predicted_damage_extent * cost_factor

# Print results
for i, cost in enumerate(estimated_costs):
    print(f"Test Image {i}: Predicted Damage Extent: {predicted_damage_extent[i]:.2f}, Estimated Cost: ₹{cost:.2f}")
