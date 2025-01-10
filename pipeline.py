import numpy as np
import onnxruntime as ort
import os
import pickle
from tqdm import tqdm
import nemo.collections.asr as nemo_asr
from jiwer import wer
import pycuda.driver as cuda
import pycuda.autoinit  # Required to initialize PyCUDA
import tensorrt as trt
import torch
import argparse

# Define arguments
parser = argparse.ArgumentParser(description="WER Computation with TensorRT")
parser.add_argument("--engine_path", type=str, required=True, help="Path to the TensorRT engine file.")
parser.add_argument("--ss_model_path", type=str, required=True, help="Path to the Speech Separation ONNX model.")
parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder.")
args = parser.parse_args()

# Paths
engine_path = args.engine_path
ss_model_path = args.ss_model_path
output_folder = args.output_folder
os.makedirs(output_folder, exist_ok=True)

# Load TensorRT engine
class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.VERBOSE)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)

    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        return engine

    def allocate_buffers(self, engine):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(tensor_name)
            size = trt.volume(shape)
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(self.HostDeviceMem(host_mem, device_mem, shape))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem, shape))
        return inputs, outputs, bindings, stream

    class HostDeviceMem:
        def __init__(self, host_mem, device_mem, shape):
            self.host = host_mem
            self.device = device_mem
            self.shape = shape

    def infer(self, input_data):
        np.copyto(self.inputs[0].host, input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)
        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output.host, output.device, self.stream)
        self.stream.synchronize()
        outputs = [output.host.reshape(output.shape) for output in self.outputs]
        return outputs

# Initialize models and preprocessor
tensorrt_infer = TensorRTInference(engine_path)
ss_session = ort.InferenceSession(ss_model_path, providers=['CPUExecutionProvider'])
first_asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("stt_en_conformer_ctc_small")
preprocessor = first_asr_model.preprocessor

def decode_predictions(logits, asr_model):
    """Decode logits into text using the ASR model's tokenizer."""
    return asr_model.tokenizer.ids_to_texts(asr_model.ctc_decoder_predictions_tensor(torch.tensor(logits)))

# Load dataloader
with open('/home/shzhou/dataloaders.pkl', 'rb') as file:
    dataloader = pickle.load(file)

# Metrics initialization
total_wer = 0.0
sample_count = 0

# Process samples
for i, sample in tqdm(enumerate(dataloader), desc="Processing Samples"):
    try:
        # SS model inference (Currently only possible with ONNXRuntime)
        inputs = {'mix_audio': sample['mix_audio'].cpu().numpy(), 's_video': sample['s_video'].cpu().numpy()}
        ss_outputs = ss_session.run(None, inputs)

        # Preprocessor 
        input_signal = torch.tensor(ss_outputs[0], dtype=torch.float32).to(preprocessor.featurizer.window.device)
        length = torch.tensor([ss_outputs[0].shape[-1]], dtype=torch.int64).to(preprocessor.featurizer.window.device)
        preprocessed_output = preprocessor(input_signal=input_signal, length=length)[0].cpu().numpy()

        # ASR model inference with TensorRT
        outputs = tensorrt_infer.infer(preprocessed_output)

        # Decode predictions
        predictions = decode_predictions(outputs[0], first_asr_model)

        # Compute WER
        reference = sample['transcription']  # Assuming 'transcription' contains the ground truth
        sample_wer = wer(reference, predictions[0])
        total_wer += sample_wer

        # Increment sample count
        sample_count += 1

    except Exception as e:
        print(f"Error processing sample {i}: {e}")
        continue

# Calculate average WER
average_wer = total_wer / sample_count if sample_count > 0 else float('nan')

# Print results
print(f"Processed {sample_count} samples.")
print(f"Average WER: {average_wer:.4f}")
