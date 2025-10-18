import onnxruntime as ort

model_path = "/home/lchang21/onnx/Qwen3-0.6B-ONNX/onnx/model.onnx"
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

print("✅ Model loaded successfully into ONNX Runtime!")
print("Inputs:", [inp.name for inp in session.get_inputs()])
print("Outputs:", [out.name for out in session.get_outputs()])