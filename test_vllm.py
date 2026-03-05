import torch
import traceback
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

def test_model_loading():
    """Test LLaVA model loading with detailed error reporting"""
    
    print("🔍 Testing LLaVA-NeXT model loading...")
    
    # Check PyTorch and GPU
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        print(f"Current GPU memory: {torch.cuda.memory_allocated() / 1024**3:.1f}GB allocated")
    else:
        print("No GPU available - will use CPU")
    
    # Test different models in order of complexity
    models_to_test = [
        "llava-hf/llava-1.5-7b-hf",           # Baseline model
        "llava-hf/llava-v1.6-mistral-7b-hf",  # Most recommended
        "llava-hf/llava-v1.6-vicuna-7b-hf",   # Alternative 7B
    ]
    
    for model_name in models_to_test:
        print(f"\n🧪 Testing model: {model_name}")
        
        try:
            # Test processor loading
            print("  Loading processor...")
            processor = LlavaNextProcessor.from_pretrained(model_name)
            print("  ✅ Processor loaded successfully")
            
            # Test model loading
            print("  Loading model...")
            gpu_available = torch.cuda.is_available()
            device_map = "auto" if gpu_available else "cpu"
            torch_dtype = torch.float16 if gpu_available else torch.float32
            
            print(f"  Device map: {device_map}, dtype: {torch_dtype}")
            
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            print(f"  ✅ Model loaded successfully on {device_map}")
            
            # Check memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                print(f"  GPU memory used: {memory_used:.1f}GB")
            
            # Clean up
            del model
            del processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"  ✅ {model_name} works - using this model")
            return model_name
            
        except Exception as e:
            print(f"  ❌ Failed to load {model_name}")
            print(f"  Error: {str(e)}")
            print(f"  Traceback: {traceback.format_exc()}")
            continue
    
    print("\n❌ All models failed to load")
    return None

if __name__ == "__main__":
    working_model = test_model_loading()
    if working_model:
        print(f"\n🎉 Use this model in your config: {working_model}")
    else:
        print("\n💡 Try installing dependencies:")
        print("pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6")
        print("pip install transformers accelerate")