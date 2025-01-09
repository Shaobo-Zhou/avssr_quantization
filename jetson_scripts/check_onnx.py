import onnx
from onnx import numpy_helper, helper
import numpy as np
import onnxruntime as ort

# Load the ONNX model
#model_path = "/home/shzhou/models/nopre_model_sim.onnx"
model_path = "/home/shzhou/models/hist_eq_channelwise_0.9999_sym_nopre_id.onnx"
model = onnx.load(model_path)
onnx.checker.check_model(model_path)
print("Model is valid.")

# Get the graph
graph = model.graph

# Print all available layers and their outputs
print("Original nodes and outputs:")
for node in graph.node:
    print(f"Node: {node.name}, Outputs: {node.output}")
    
# Outputs of interest
#outputs_to_add = [
    #("/encoder/pre_encode/out/Add_output_0", [1, 51, 176]),  # Add shape if known
    #("/encoder/layers.0/norm_feed_forward1/Add_1_output_0", [1, 51, 176]),
#]
outputs_to_add = [
    ("/asr_encoder/pre_encode/out/Add_output_0", [1, 51, 176]),  # Add shape if known
    ("/asr_encoder/0/norm_feed_forward1/Add_1_output_0", [1, 51, 176]),
]

""" # Generate outputs for all layers (0-15)
layer_indices = range(16) 

for i in layer_indices:
    # Add output for `/asr_encoder/i/norm_out/Add_1`
    outputs_to_add.append(
        (f"/asr_encoder/{i}/norm_out/Add_1_output_0", [1, 51, 176])  # Adjust shape as needed
    )
    # Add output for `/asr_encoder/i/norm_feed_forward2/Add_1`
    outputs_to_add.append(
        (f"/asr_encoder/{i}/norm_feed_forward2/Add_1_output_0", [1, 51, 176])
    )
    # Add outputs for feed_forward2 components
    outputs_to_add.append(
        (f"/asr_encoder/{i}/feed_forward2/DequantizeLinear_output_0", [1, 51, 176])
    )
    #outputs_to_add.append(
        #(f"/asr_encoder/{i}/feed_forward2/linear2/DequantizeLinear_output_0", [1, 51, 704])
    #)
    #outputs_to_add.append(
        #(f"/asr_encoder/{i}/feed_forward2/dequant/DequantizeLinear_output_0", [1, 51, 704])
    #)
    #outputs_to_add.append(
        #(f"/asr_encoder/{i}/feed_forward2/linear1/DequantizeLinear_output_0", [1, 51, 704])
    #)
    # Add output for `/asr_encoder/i/norm_feed_forward1/Add_1`
    outputs_to_add.append(
        (f"/asr_encoder/{i}/norm_feed_forward1/Add_1_output_0", [1, 51, 176])
    )
    # Add outputs for feed_forward1 components
    outputs_to_add.append(
        (f"/asr_encoder/{i}/feed_forward1/DequantizeLinear_output_0", [1, 51, 176])
    )
    #outputs_to_add.append(
        #(f"/asr_encoder/{i}/feed_forward1/linear2/DequantizeLinear_output_0", [1, 51, 704])
    #)
    outputs_to_add.append(
        (f"/asr_encoder/{i}/feed_forward1/dequant/DequantizeLinear_output_0", [1, 51, 704])
    )
    #outputs_to_add.append(
        #(f"/asr_encoder/{i}/feed_forward1/linear1/DequantizeLinear_output_0", [1, 51, 704])
    #)
    outputs_to_add.append(
        (f"/asr_encoder/{i}/conv/DequantizeLinear_2_output_0", [1, 51, 176])
    )
print(outputs_to_add) """
for output_name, shape in outputs_to_add:
    # Ensure it isn't already an output
    existing_outputs = [output.name for output in graph.output]
    if output_name not in existing_outputs:
        output_value_info = helper.make_tensor_value_info(
            name=output_name,
            elem_type=onnx.TensorProto.FLOAT,  # Assuming float32
            shape=shape
        )
        graph.output.append(output_value_info)
modified_model_path = "/home/shzhou/models/hist_eq_channelwise_0.9999_sym_nopre_id_new.onnx"
onnx.save(model, modified_model_path)
print(f"Modified ONNX model saved to {modified_model_path}")

# Verify the model
onnx.checker.check_model(modified_model_path)
print("Modified model is valid.")




