import json
from tensorflow.keras.models import load_model

# Load the h5 model we created
model = load_model('accident_classification_model.h5')

# Save the model architecture to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Save the weights with the correct file naming convention
model.save_weights("model.weights.h5")

print("Model converted successfully!") 