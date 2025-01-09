import onnx
from onnx import numpy_helper
import numpy as np
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Adjust uint8 tensors in an ONNX model.")
parser.add_argument("model_path", type=str, help="Path to the ONNX model.")
args = parser.parse_args()

# Load the ONNX model
model_path = args.model_path
model = onnx.load(model_path)

# Track changes
changes_made = False

# Iterate through initializers
for initializer in model.graph.initializer:
    tensor_array = numpy_helper.to_array(initializer)
    if (tensor_array == 128).all():
        print(f"Updating Initializer Tensor: {initializer.name}")

        # Modify the tensor: convert uint8 to int8 and subtract zero point
        modified_tensor_array = (tensor_array.astype(np.int32) - 128).astype(np.int8)
        initializer.CopyFrom(numpy_helper.from_array(modified_tensor_array, name=initializer.name))
        initializer.data_type = 3  # Set type to INT8
        changes_made = True

# Iterate through Constant nodes
for node in model.graph.node:
    if node.op_type == "Constant":
        for attr in node.attribute:
            if attr.name == "value":
                tensor_array = numpy_helper.to_array(attr.t)
                if (tensor_array == 128).all():
                    print(f"Updating Constant Node: {node.name}")
                    modified_tensor_array = (tensor_array.astype(np.int32) - 128).astype(np.int8)
                    attr.t.CopyFrom(numpy_helper.from_array(modified_tensor_array))
                    attr.t.data_type = 3  # Set type to INT8
                    changes_made = True

# Update QuantizeLinear nodes' output types
for node in model.graph.node:
    if node.op_type == "QuantizeLinear":
        output_name = node.output[0]
        for value_info in model.graph.value_info:
            if value_info.name == output_name:
                value_info.type.tensor_type.elem_type = onnx.TensorProto.INT8
                changes_made = True

# Save the modified model
if changes_made:
    onnx.save(model, model_path)
    print(f"Modified model saved to {model_path}")
else:
    print("No tensors with value=128 found. No changes made.")
















