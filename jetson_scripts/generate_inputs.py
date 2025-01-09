import numpy as np
import json


# Create or load input data
input_data = np.random.randn(1, 80, 201).astype(np.float32)  # Example shape and dtype

# Save as .npy file
#np.save("input0.npy", input_data)

# Load the JSON file
output_file = "output.json"
with open(output_file, "r") as f:
    data = json.load(f)
output_data = np.array(data[0]['values'])
output_shape = (1, 51, 1025)
output_data = output_data.reshape(output_shape)
tokens_logits = output_data
print(tokens_logits)
max_tokens = np.argmax(tokens_logits, axis=-1)
print(max_tokens.shape)
print(f"Max tokens: {max_tokens}")


