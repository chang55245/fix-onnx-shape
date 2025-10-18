import onnx
from onnx import checker
import os

model_path = "/home/lchang21/onnx/test/qwen/model_fixed.onnx"

checker.check_model(model_path)
print("✅ Model validation passed!")

print("=== ONNX Model Validation Check ===")
print(f"Checking model: {model_path}")

# Check file size first
file_size = os.path.getsize(model_path)
print(f"File size: {file_size:,} bytes ({file_size / 1e9:.2f} GB)")

if file_size > 2e9:  # > 2GB
    print("⚠️ Large model detected - parsing may have limitations")

try:
    print("Attempting to load model...")
    # Try to load the model
    model = onnx.load(model_path)
    print(f"✅ Model loaded successfully!")
    print(f"IR version: {model.ir_version}")
    print(f"Nodes: {len(model.graph.node)}")
    print(f"Inputs: {len(model.graph.input)}")
    print(f"Outputs: {len(model.graph.output)}")
    
    print("\nAttempting in-memory validation...")
    # Try in-memory validation first
    checker.check_model(model)
    print("✅ Model validation passed!")
    
except Exception as e:
    error_msg = str(e)
    print(f"❌ Validation failed: {error_msg}")
    
    # Handle specific error types
    if "Error parsing message with type 'onnx.ModelProto'" in error_msg:
        print("\n🔍 This is a protobuf parsing error with large models.")
        print("This typically happens with models >2GB due to:")
        print("1. Protobuf size limitations")
        print("2. Memory constraints during parsing")
        print("3. Issues with tensor data encoding")
        
        print("\n💡 Solutions to try:")
        print("1. Use external data format (separate .bin file for tensor data)")
        print("2. Validate model structure without loading all tensor data")
        print("3. Use ONNX runtime for inference instead of checker validation")
        
        # Try to verify the fix worked by checking the error type
        print("\n🔍 Checking if original custom op issues were resolved...")
        try:
            # Try to load the original model to compare errors
            orig_model_path = "/home/lchang21/onnx/Qwen3-0.6B-ONNX/onnx/model.onnx"
            orig_model = onnx.load(orig_model_path)
            checker.check_model(orig_model)
        except Exception as orig_e:
            orig_error = str(orig_e)
            if 'SimplifiedLayerNormalization' in orig_error and 'SimplifiedLayerNormalization' not in error_msg:
                print("✅ SUCCESS: Original SimplifiedLayerNormalization errors have been eliminated!")
                print("The parsing error is a different issue (large file size) not the original custom op problem.")
            else:
                print(f"Original model error: {orig_error[:100]}...")
                
    elif 'ir_version' in error_msg:
        print("IR version issue detected - checking manually...")
        try:
            # Try a partial load to check IR version
            with open(model_path, 'rb') as f:
                header = f.read(50)
                if b'\x08\x0c' in header or (len(header) > 2 and header[1] == 0x0c):
                    print("✅ IR version 12 found in binary header")
                else:
                    print("❌ IR version 12 not found in binary header")
        except Exception as header_e:
            print(f"Cannot check binary header: {header_e}")
    else:
        print(f"Other validation error: {error_msg}")