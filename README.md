# OpenGPU

🔬 **OpenGPU** is an experimental research project exploring how to make GPU resources in Kubernetes clusters **vendor-agnostic** and easier to consume at scale.  

Today, Kubernetes GPU support is fragmented across vendors (NVIDIA, AMD, Intel, and emerging accelerators). Each ecosystem provides its own device plugin, driver stack, and operator, leaving developers and platform teams to manage complexity.  

The vision of OpenGPU is to provide a **unified abstraction layer** where workloads can request GPU resources by capability (e.g., *“1 GPU with ≥24GB memory”*) without needing to know the underlying hardware architecture.  

---

## 🎯 Research Goals

- **Hardware Abstraction**  
  Detect and normalize GPU metadata across vendors (NVIDIA, AMD, Intel, and beyond).  

- **Cluster Integration**  
  Represent GPU capabilities as Kubernetes resources, independent of vendor driver/operator implementations.  

- **Developer Experience**  
  Enable developers to target a simple CRD (`GPURequest`) instead of vendor-specific annotations.  

- **Scalability & Heterogeneity**  
  Support mixed clusters (e.g., NVIDIA + AMD) and future accelerators with minimal operator overhead.  

---

## ✅ Current Prototype

The first proof-of-concept focuses on **GPU detection and classification**.  
- Detects installed GPUs from multiple vendors:
  - **NVIDIA** via `nvidia-smi`  
  - **AMD** via `rocminfo`  
  - **Intel** via `clinfo`  
- Extracts metadata such as:
  - Vendor  
  - Model name / architecture (e.g., NVIDIA Hopper, AMD `gfx1030`)  
  - Driver version  
  - Memory capacity  
- Returns this data in a normalized output for later use in Kubernetes labeling.  

---

## 🛠️ Roadmap (Research Phases)

1. **Phase 1 – Detection (PoC)**  
   Local GPU detection in Rust → output vendor, architecture, memory.  

2. **Phase 2 – Node Annotation**  
   Deploy detection agent as a Kubernetes **DaemonSet** that labels GPU-capable nodes automatically.  

3. **Phase 3 – Resource Abstraction**  
   Define a **Custom Resource Definition (CRD)** (`GPURequest`) to allow vendor-agnostic GPU requests.  

4. **Phase 4 – Scheduling**  
   Build a **Rust operator** (using [`kube-rs`](https://github.com/kube-rs/kube)) that matches workloads to available GPUs by capability.  

5. **Phase 5 – Advanced Resource Management**  
   Explore GPU partitioning (NVIDIA MIG, AMD MxGPU), time-slicing, and multi-vendor scheduling policies.  

---

## 📦 Usage

### Build
```bash
git clone https://github.com/bdbrink/OpenGPU.git
cd OpenGPU/gpu-detect
cargo build --release
```

### Run
`./target/release/gpu-detect`
or 
`cargo run`