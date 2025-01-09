import onnx
from onnx import numpy_helper
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Fix DequantizeLinear nodes in an ONNX model.")
parser.add_argument("model_path", type=str, help="Path to the ONNX model.")
args = parser.parse_args()

# Load the ONNX model
model_path = args.model_path
onnx_model = onnx.load(model_path)

# Track changes
changes_made = False
counter = 0

# Process DequantizeLinear nodes
for node in list(onnx_model.graph.node):  # Use list to safely remove nodes
    if node.op_type == "DequantizeLinear":
        input_value, input_scale, input_zero_point = node.input
        bias_initializer = next((init for init in onnx_model.graph.initializer if init.name == input_value), None)
        scale_initializer = next((init for init in onnx_model.graph.initializer if init.name == input_scale), None)
        zero_point_initializer = next((init for init in onnx_model.graph.initializer if init.name == input_zero_point), None)

        if bias_initializer and scale_initializer and zero_point_initializer:
            print(f"Fixing DequantizeLinear node: {node.name}")

            # Load data
            bias_array = numpy_helper.to_array(bias_initializer)
            scale_array = numpy_helper.to_array(scale_initializer)
            zero_point_array = numpy_helper.to_array(zero_point_initializer)

            if bias_initializer.data_type == onnx.TensorProto.INT32:
                float_bias_array = (bias_array - zero_point_array) * scale_array
                unique_name = f"{bias_initializer.name}_dequantized_{counter}"
                counter += 1

                new_initializer = numpy_helper.from_array(float_bias_array.astype("float32"), unique_name)
                onnx_model.graph.initializer.append(new_initializer)

                for consumer_node in onnx_model.graph.node:
                    for i, input_name in enumerate(consumer_node.input):
                        if input_name == node.output[0]:
                            consumer_node.input[i] = unique_name
                            print(f"Updated input of node {consumer_node.name} to use {unique_name}")

                onnx_model.graph.node.remove(node)
                changes_made = True

# Save the modified model
if changes_made:
    onnx.save(onnx_model, model_path)
    print(f"Modified model saved to {model_path}")
else:
    print("No changes made to the model.")
