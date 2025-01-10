import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  
import tensorrt as trt
import os
import argparse
import time
from tqdm import tqdm

if not hasattr(np, 'bool'):
    np.bool = np.bool_

class TensorRTInference:
    def __init__(self, engine_path):
        # Initialize TensorRT runtime and load engine
        self.logger = trt.Logger(trt.Logger.VERBOSE)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        # Allocate buffers for inputs and outputs
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)
        # Retrieve output layer names
        self.output_names = self.get_output_names()

    def load_engine(self, engine_path):
        """Loads the TensorRT engine from a file."""
        with open(engine_path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        return engine

    class HostDeviceMem:
        def __init__(self, host_mem, device_mem, shape):
            # Track host and device memory
            self.host = host_mem
            self.device = device_mem
            self.shape = shape  # Track shape for un-flattening

    def allocate_buffers(self, engine):
        """Allocates host and device buffers for inputs and outputs."""
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(tensor_name)
            size = trt.volume(shape)
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append device buffer address to bindings
            bindings.append(int(device_mem))

            # Append to input or output list based on tensor mode
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(self.HostDeviceMem(host_mem, device_mem, shape))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem, shape))

        return inputs, outputs, bindings, stream

    def get_output_names(self):
        """Retrieves the names of all output tensors."""
        return [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
        ]

    def infer(self, input_data):
        """Runs inference and retrieves outputs."""
        # Copy input data to device
        np.copyto(self.inputs[0].host, input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)

        # Set tensor addresses for inputs and outputs
        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])

        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy outputs from device to host
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output.host, output.device, self.stream)

        # Synchronize the stream
        self.stream.synchronize()

        # Reshape and return outputs
        outputs = []
        for output in self.outputs:
            reshaped_output = output.host.reshape(output.shape)
            outputs.append(reshaped_output)
        return outputs

# Main execution logic
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TensorRT inference on multiple input files.")
    parser.add_argument("engine_path", type=str, help="Path to the TensorRT engine file.")
    #parser.add_argument("output_folder", type=str, help="Path to the folder to save output .npy files.")
    parser.add_argument(
        "--input_shape",
        type=int,
        nargs='+',
        required=True,
        help="Shape of the random input tensor. E.g., --input_shape 1 3 224 224 for 1x3x224x224.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of random samples to process. Defaults to 1000.",
    )
    args = parser.parse_args()

    # Ensure output folder exists
    #os.makedirs(args.output_folder, exist_ok=True)

    # Initialize TensorRT inference
    tensorrt_infer = TensorRTInference(args.engine_path)

    sample_count=0

    # Processing loop
    for i in tqdm(range(args.num_samples), desc="Processing Samples"):
        try:
            # Generate random input
            input_signal = np.random.rand(*args.input_shape).astype(np.float32)
            # Define input signal length
            input_signal_length = np.array([201], dtype=np.int32)

            # Copy input signal length to device if necessary
            np.copyto(tensorrt_infer.inputs[1].host, input_signal_length.ravel())
            cuda.memcpy_htod(tensorrt_infer.inputs[1].device, tensorrt_infer.inputs[1].host)
            # Perform inference
            inference_start_time = time.time()
            outputs = tensorrt_infer.infer(input_signal)
            inference_end_time = time.time()

            # Save output
            output_path = os.path.join(args.output_folder, f"output_{i}.npy")
            np.save(output_path, outputs[0])

            # Increment sample count
            sample_count += 1

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

        end_time = time.time()  # Throughput calculation