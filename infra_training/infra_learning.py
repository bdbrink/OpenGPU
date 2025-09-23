#!/usr/bin/env python3
"""
SRE AI Training Script - Enhanced with Rust GPU Detection Integration
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import psutil
import os
import subprocess
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class GPUManager:
    """Interface between Rust GPU detection and Python training logic"""
    
    def __init__(self, rust_binary_path: Optional[str] = None):
        # Try to auto-detect the Rust binary location based on your project structure
        if rust_binary_path is None:
            possible_paths = [
                "../gpu-detect/target/release/gpu-detect"
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    rust_binary_path = path
                    print(f"Found Rust binary at: {path}")
                    break
            
            if rust_binary_path is None:
                rust_binary_path = "../gpu-detect/target/release/gpu_detect"  # Your structure
        
        self.rust_binary_path = rust_binary_path
        self.gpu_info = None
        self.system_specs = None
        self._detect_gpu_info()
    
    def _detect_gpu_info(self) -> None:
        """Run Rust GPU detection and parse results"""
        print("🔍 Running Rust GPU Detection...")
        
        try:
            # Check if Rust binary exists
            if not Path(self.rust_binary_path).exists():
                print(f"❌ Rust binary not found at {self.rust_binary_path}")
                print("💡 Run 'cargo build --release' to build the GPU detection tool")
                self._fallback_to_pytorch_detection()
                return
            
            # Run the Rust GPU detection - redirect stderr to suppress debug output
            result = subprocess.run(
                [self.rust_binary_path, "--json"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            
            if result.stdout.strip():
                try:
                    # The output contains debug logs, but JSON should be the last complete line
                    output_lines = result.stdout.strip().split('\n')
                    
                    # Find the line that looks like JSON (starts with { and ends with })
                    json_line = None
                    for line in reversed(output_lines):
                        stripped = line.strip()
                        if stripped.startswith('{') and stripped.endswith('}'):
                            json_line = stripped
                            break
                    
                    if json_line:
                        self.gpu_info = json.loads(json_line)
                        print(f"✅ GPU Detection successful via Rust")
                        self._print_gpu_summary()
                    else:
                        print("⚠️ No valid JSON line found, using fallback")
                        self._fallback_to_pytorch_detection()
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON decode error: {e}")
                    print(f"Raw output: {repr(result.stdout)}")
                    self._fallback_to_pytorch_detection()
            else:
                print("⚠️ No output from Rust detection, using fallback")
                self._fallback_to_pytorch_detection()
                
        except subprocess.TimeoutExpired:
            print("⚠️ Rust GPU detection timed out, using fallback")
            self._fallback_to_pytorch_detection()
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Rust GPU detection failed: {e}")
            self._fallback_to_pytorch_detection()
        except FileNotFoundError:
            print(f"⚠️ Rust binary not found at {self.rust_binary_path}")
            self._fallback_to_pytorch_detection()
        
    def _fallback_to_pytorch_detection(self) -> None:
        """Fallback to PyTorch's built-in GPU detection"""
        print("🔄 Using PyTorch GPU detection as fallback...")
        
        self.gpu_info = {
            "gpu_type": "CPU Only",
            "vram_gb": 0.0,
            "is_ml_ready": False,
            "compute_capability": None,
            "driver_version": None
        }
        
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            self.gpu_info = {
                "gpu_type": f"NVIDIA {torch.cuda.get_device_name(0)}",
                "vram_gb": props.total_memory / (1024**3),
                "is_ml_ready": props.total_memory > 2147483648,  # 2GB+
                "compute_capability": f"{props.major}.{props.minor}",
                "driver_version": None
            }
    
    def _print_gpu_summary(self) -> None:
        """Print a summary of detected GPU info"""
        if self.gpu_info:
            print(f"📊 GPU: {self.gpu_info.get('gpu_type', 'Unknown')}")
            if self.gpu_info.get('vram_gb', 0) > 0:
                print(f"📊 VRAM: {self.gpu_info['vram_gb']:.1f}GB")
            print(f"📊 ML Ready: {'Yes' if self.gpu_info.get('is_ml_ready', False) else 'No'}")
    
    def get_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on GPU/system capabilities"""
        if not self.gpu_info or not self.gpu_info.get('is_ml_ready', False):
            return 16  # Conservative for CPU or limited GPU
        
        vram_gb = self.gpu_info.get('vram_gb', 0)
        
        # Heuristic based on VRAM (leaving room for model weights)
        if vram_gb >= 24:
            return 128  # Large batch for high-end cards
        elif vram_gb >= 16:
            return 64   # Your 7800 XT falls here
        elif vram_gb >= 12:
            return 48
        elif vram_gb >= 8:
            return 32
        elif vram_gb >= 6:
            return 24
        else:
            return 16
    
    def should_use_mixed_precision(self) -> bool:
        """Determine if mixed precision training should be used"""
        if not self.gpu_info or not self.gpu_info.get('is_ml_ready', False):
            return False
        
        gpu_type = self.gpu_info.get('gpu_type', '').lower()
        
        # Enable for modern GPUs that support it well
        modern_nvidia = any(x in gpu_type for x in ['rtx', 'gtx 16', 'tesla', 'quadro rtx'])
        modern_amd = any(x in gpu_type for x in ['rx 6', 'rx 7', 'radeon pro'])
        
        return modern_nvidia or modern_amd
    
    def get_recommended_model_size(self) -> Tuple[str, str]:
        """Get recommended model size based on capabilities"""
        if not self.gpu_info:
            return ("small", "Qwen/Qwen2-0.5B-Instruct")
        
        vram_gb = self.gpu_info.get('vram_gb', 0)
        is_ml_ready = self.gpu_info.get('is_ml_ready', False)
        
        if not is_ml_ready:
            return ("small", "Qwen/Qwen2-0.5B-Instruct")
        elif vram_gb >= 20:
            return ("large", "Qwen/Qwen2-14B-Instruct")  # Could handle 14B
        elif vram_gb >= 12:
            return ("medium", "Qwen/Qwen2-7B-Instruct")   # 7B should fit nicely
        elif vram_gb >= 8:
            return ("medium", "Qwen/Qwen2-1.5B-Instruct") # Conservative for 8GB
        else:
            return ("small", "Qwen/Qwen2-0.5B-Instruct")
    
    def get_torch_device_config(self) -> Dict:
        """Get PyTorch device configuration based on detected hardware"""
        config = {
            "device": "cpu",
            "torch_dtype": torch.float32,
            "device_map": None,
            "low_cpu_mem_usage": True
        }
        
        if self.gpu_info and self.gpu_info.get('is_ml_ready', False):
            gpu_type = self.gpu_info.get('gpu_type', '').lower()
            
            if 'nvidia' in gpu_type and torch.cuda.is_available():
                config.update({
                    "device": "cuda",
                    "torch_dtype": torch.bfloat16 if self.should_use_mixed_precision() else torch.float16,
                    "device_map": "auto"
                })
            elif 'amd' in gpu_type:
                # AMD GPU detected - check if ROCm is available
                if torch.cuda.is_available():  # ROCm uses CUDA API
                    config.update({
                        "device": "cuda",  # ROCm uses cuda device string
                        "torch_dtype": torch.float16,
                        "device_map": "auto"
                    })
                else:
                    print("⚠️ AMD GPU detected but ROCm not available, using CPU")
        
        return config

def enhanced_system_check(gpu_manager: GPUManager):
    """Enhanced system check using Rust GPU detection results"""
    print("🔍 Enhanced System Check (Rust + Python)")
    print("=" * 60)
    
    # Basic PyTorch info
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    # GPU info from Rust detection
    if gpu_manager.gpu_info:
        print(f"GPU (Rust): {gpu_manager.gpu_info['gpu_type']}")
        if gpu_manager.gpu_info.get('vram_gb', 0) > 0:
            print(f"VRAM (Rust): {gpu_manager.gpu_info['vram_gb']:.1f}GB")
        if gpu_manager.gpu_info.get('compute_capability'):
            print(f"Compute Capability: {gpu_manager.gpu_info['compute_capability']}")
        print(f"ML Ready: {gpu_manager.gpu_info.get('is_ml_ready', False)}")
    
    # System specs
    print(f"CPU Cores: {psutil.cpu_count()}")
    print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f}GB")
    
    # Training recommendations
    print(f"\n📊 Training Recommendations:")
    print(f"Optimal Batch Size: {gpu_manager.get_optimal_batch_size()}")
    print(f"Mixed Precision: {gpu_manager.should_use_mixed_precision()}")
    
    model_size, model_name = gpu_manager.get_recommended_model_size()
    print(f"Recommended Model: {model_name} ({model_size})")
    print()

def load_model_with_gpu_config(gpu_manager: GPUManager):
    """Load model using GPU manager's recommended configuration"""
    print("🤖 Loading Model with Optimized Configuration")
    print("=" * 60)
    
    # Get recommended model and config
    model_size, model_name = gpu_manager.get_recommended_model_size()
    device_config = gpu_manager.get_torch_device_config()
    
    print(f"Selected Model: {model_name}")
    print(f"Device Config: {device_config['device']} | {device_config['torch_dtype']}")
    
    start_time = time.time()
    
    try:
        # Load tokenizer
        print("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Load model with optimized config
        print(f"🧠 Loading model on {device_config['device']}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=device_config['torch_dtype'],
            device_map=device_config['device_map'],
            trust_remote_code=True,
            low_cpu_mem_usage=device_config['low_cpu_mem_usage']
        )
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        
        return tokenizer, model, device_config['device']
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Falling back to smaller model...")
        
        # Fallback to smallest model
        fallback_model = "Qwen/Qwen2-0.5B-Instruct"
        try:
            tokenizer = AutoTokenizer.from_pretrained(fallback_model, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                fallback_model,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            print(f"✅ Fallback model {fallback_model} loaded successfully")
            return tokenizer, model, "cpu"
        except Exception as fallback_error:
            print(f"❌ Fallback also failed: {fallback_error}")
            return None, None, None

def optimized_inference_test(tokenizer, model, device, gpu_manager: GPUManager):
    """Run inference test with GPU-optimized settings"""
    print("\n🚀 Optimized Inference Test")
    print("=" * 50)
    
    # SRE-themed test prompt with more complexity for GPU testing
    test_prompt = """You are an SRE AI assistant. A distributed microservices application 
    is experiencing cascading failures. The load balancer shows 503 errors, 
    several Kubernetes pods are in CrashLoopBackOff, and monitoring alerts indicate 
    high memory usage. What systematic approach would you take to diagnose and resolve this issue?"""
    
    print(f"Prompt: {test_prompt[:100]}...")
    print(f"\nUsing batch size: {gpu_manager.get_optimal_batch_size()}")
    print(f"Mixed precision: {gpu_manager.should_use_mixed_precision()}")
    print("\nResponse:")
    print("-" * 30)
    
    try:
        # Tokenize with optimized settings
        inputs = tokenizer(test_prompt, return_tensors="pt")
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate with GPU-optimized parameters
        generation_config = {
            "max_new_tokens": 200,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": tokenizer.eos_token_id
        }
        
        # Add batch size if GPU is capable
        if gpu_manager.gpu_info and gpu_manager.gpu_info.get('is_ml_ready', False):
            generation_config["num_beams"] = 1  # Keep it simple for now
        
        inference_start = time.time()
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_config)
        
        inference_time = time.time() - inference_start
        
        # Decode and display response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(test_prompt):].strip()
        
        print(response)
        print(f"\n✅ Inference completed in {inference_time:.2f} seconds")
        print(f"📊 Tokens/second: ~{generation_config['max_new_tokens']/inference_time:.1f}")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")

def enhanced_memory_monitoring(gpu_manager: GPUManager):
    """Enhanced memory monitoring including GPU-specific metrics"""
    print(f"\n💾 Enhanced Memory Usage")
    print("=" * 50)
    
    # Standard RAM monitoring
    ram = psutil.virtual_memory()
    print(f"System RAM: {ram.used / 1024**3:.1f}GB / {ram.total / 1024**3:.1f}GB ({ram.percent:.1f}%)")
    
    # GPU memory monitoring based on detection
    if gpu_manager.gpu_info and gpu_manager.gpu_info.get('is_ml_ready', False):
        gpu_type = gpu_manager.gpu_info.get('gpu_type', '').lower()
        
        if 'nvidia' in gpu_type and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            total = torch.cuda.get_device_properties(0).total_memory
            
            print(f"GPU Memory Allocated: {allocated / 1024**3:.1f}GB")
            print(f"GPU Memory Reserved: {reserved / 1024**3:.1f}GB")
            print(f"GPU Memory Total: {total / 1024**3:.1f}GB")
            print(f"GPU Utilization: {(allocated/total)*100:.1f}%")
        
        elif 'amd' in gpu_type:
            print(f"AMD GPU Memory: {gpu_manager.gpu_info['vram_gb']:.1f}GB total")
            print("💡 Install ROCm tools for detailed AMD GPU memory monitoring")
        
        else:
            print("GPU memory monitoring not available for this GPU type")
    else:
        print("No ML-capable GPU detected")

def main():
    """Enhanced main execution flow with Rust GPU integration"""
    print("🎯 Enhanced SRE AI Training - Rust GPU Integration")
    print("=" * 70)
    
    # Initialize GPU manager (this runs Rust detection)
    gpu_manager = GPUManager()
    
    # Enhanced system check
    enhanced_system_check(gpu_manager)
    
    # Load model with GPU-optimized config
    tokenizer, model, device = load_model_with_gpu_config(gpu_manager)
    
    if tokenizer and model:
        # Run optimized inference test
        optimized_inference_test(tokenizer, model, device, gpu_manager)
        
        # Enhanced memory monitoring
        enhanced_memory_monitoring(gpu_manager)
        
        print(f"\n🎉 Enhanced bootstrap complete!")
        print(f"💡 Rust GPU detection provided: {gpu_manager.gpu_info['gpu_type']}")
        print(f"📊 Optimal batch size for training: {gpu_manager.get_optimal_batch_size()}")
        print(f"🚀 Ready for k8s data integration with optimized GPU settings!")
    else:
        print("❌ Enhanced bootstrap failed. Check your setup and GPU detection.")
        print("💡 Try running the Rust GPU detection manually: ./target/release/gpu_detect")

if __name__ == "__main__":
    main()