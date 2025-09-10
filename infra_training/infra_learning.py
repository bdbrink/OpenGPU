#!/usr/bin/env python3
"""
SRE AI Training Script - Bootstrap with Qwen Model Loading
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import psutil
import os

def check_system():
    """Check system capabilities"""
    print("🔍 System Check")
    print("=" * 50)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"ROCm/CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print(f"CPU Count: {psutil.cpu_count()}")
    print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print()

def load_qwen_model():
    """Load a lightweight Qwen model for testing"""
    print("🤖 Loading Qwen Model")
    print("=" * 50)
    
    # Using Qwen2-0.5B for initial testing (smaller, faster to load)
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    
    print(f"Loading model: {model_name}")
    start_time = time.time()
    
    try:
        # Load tokenizer
        print("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Using device: {device}")
        
        # Load model with appropriate settings
        print("🧠 Loading model...")
        if device == "cuda":
            # For GPU, use bfloat16 to save memory
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:
            # For CPU
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        
        return tokenizer, model, device
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None

def test_inference(tokenizer, model, device):
    """Test basic inference with SRE-style prompt"""
    print("\n🚀 Testing Inference")
    print("=" * 50)
    
    # SRE-themed test prompt
    test_prompt = """You are an SRE AI assistant. A Kubernetes pod is in CrashLoopBackOff state. 
    What are the first three steps you would take to diagnose this issue?"""
    
    print(f"Prompt: {test_prompt}")
    print("\nResponse:")
    print("-" * 30)
    
    try:
        # Tokenize input
        inputs = tokenizer(test_prompt, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the original prompt from response
        response = response[len(test_prompt):].strip()
        
        print(response)
        print("\n✅ Inference test successful!")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")

def check_memory_usage():
    """Check current memory usage"""
    print(f"\n💾 Memory Usage")
    print("=" * 50)
    
    # RAM usage
    ram = psutil.virtual_memory()
    print(f"RAM: {ram.used / 1024**3:.1f} GB / {ram.total / 1024**3:.1f} GB ({ram.percent:.1f}%)")
    
    # GPU memory usage
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        cached = torch.cuda.memory_reserved(0)
        print(f"GPU Memory Allocated: {allocated / 1024**3:.1f} GB")
        print(f"GPU Memory Cached: {cached / 1024**3:.1f} GB")
        print(f"GPU Memory Total: {gpu_memory / 1024**3:.1f} GB")

def main():
    """Main execution flow"""
    print("🎯 SRE AI Training - Bootstrap Phase")
    print("=" * 60)
    
    # System check
    check_system()
    
    # Load model
    tokenizer, model, device = load_qwen_model()
    
    if tokenizer and model:
        # Test inference
        test_inference(tokenizer, model, device)
        
        # Check memory usage
        check_memory_usage()
        
        print(f"\n🎉 Bootstrap complete! Ready for k8s data integration.")
    else:
        print("❌ Bootstrap failed. Check your setup.")

if __name__ == "__main__":
    main()