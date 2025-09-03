use std::process::Command;
use std::fs;
use crate::detect::GpuInfo;

#[derive(Debug, Clone)]
pub enum ReadinessLevel {
    Excellent,
    Good,
    Adequate,
    Poor,
    Insufficient,
}

impl ReadinessLevel {
    fn emoji(&self) -> &'static str {
        match self {
            ReadinessLevel::Excellent => "🚀",
            ReadinessLevel::Good => "✅",
            ReadinessLevel::Adequate => "⚠️",
            ReadinessLevel::Poor => "❌",
            ReadinessLevel::Insufficient => "🚫",
        }
    }
    
    fn description(&self) -> &'static str {
        match self {
            ReadinessLevel::Excellent => "Excellent - Ready for large models",
            ReadinessLevel::Good => "Good - Can handle medium-large models",
            ReadinessLevel::Adequate => "Adequate - Suitable for small-medium models",
            ReadinessLevel::Poor => "Poor - Limited to small models only",
            ReadinessLevel::Insufficient => "Insufficient - Cannot run ML models effectively",
        }
    }
}

#[derive(Debug)]
pub struct SystemSpecs {
    pub cpu_cores: u32,
    pub cpu_model: String,
    pub total_ram_gb: f64,
    pub available_ram_gb: f64,
    pub disk_space_gb: f64,
    pub gpu_info: GpuInfo,
    pub python_version: Option<String>,
    pub rust_version: Option<String>,
    pub cuda_version: Option<String>,
    pub rocm_version: Option<String>,
}

#[derive(Debug)]
pub struct ModelRecommendation {
    pub model_size: String,
    pub examples: Vec<String>,
    pub ram_required: String,
    pub vram_required: String,
    pub inference_speed: String,
}

pub struct MLReadinessChecker;

impl MLReadinessChecker {
    pub fn check_system_with_gpu(gpu_info: GpuInfo) -> SystemSpecs {
        println!("🔍 Analyzing system for ML readiness...\n");
        
        let cpu_info = Self::get_cpu_info();
        let memory_info = Self::get_memory_info();
        let disk_info = Self::get_disk_info();
        let python_version = Self::get_python_version();
        let rust_version = Self::get_rust_version();
        let cuda_version = Self::get_cuda_version();
        let rocm_version = Self::get_rocm_version();
        
        SystemSpecs {
            cpu_cores: cpu_info.0,
            cpu_model: cpu_info.1,
            total_ram_gb: memory_info.0,
            available_ram_gb: memory_info.1,
            disk_space_gb: disk_info,
            gpu_info,
            python_version,
            rust_version,
            cuda_version,
            rocm_version,
        }
    }
    
    fn get_cpu_info() -> (u32, String) {
        let mut cores = 1u32;
        let mut model = "Unknown CPU".to_string();
        
        if let Ok(contents) = fs::read_to_string("/proc/cpuinfo") {
            let mut core_count = 0;
            
            for line in contents.lines() {
                if line.starts_with("processor") {
                    core_count += 1;
                }
                if line.starts_with("model name") && model == "Unknown CPU" {
                    if let Some(name) = line.split(':').nth(1) {
                        model = name.trim().to_string();
                    }
                }
            }
            cores = core_count;
        }
        
        (cores, model)
    }
    
    fn get_memory_info() -> (f64, f64) {
        let mut total_kb = 0u64;
        let mut available_kb = 0u64;
        
        if let Ok(contents) = fs::read_to_string("/proc/meminfo") {
            for line in contents.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(value) = line.split_whitespace().nth(1) {
                        total_kb = value.parse().unwrap_or(0);
                    }
                }
                if line.starts_with("MemAvailable:") {
                    if let Some(value) = line.split_whitespace().nth(1) {
                        available_kb = value.parse().unwrap_or(0);
                    }
                }
            }
        }
        
        let total_gb = total_kb as f64 / 1024.0 / 1024.0;
        let available_gb = available_kb as f64 / 1024.0 / 1024.0;
        
        (total_gb, available_gb)
    }
    
    fn get_disk_info() -> f64 {
        match Command::new("df").args(&["-h", "/"]).output() {
            Ok(output) => {
                let output_str = String::from_utf8_lossy(&output.stdout);
                if let Some(line) = output_str.lines().nth(1) {
                    if let Some(available) = line.split_whitespace().nth(3) {
                        let size_str = available.trim_end_matches('G')
                            .trim_end_matches('T')
                            .trim_end_matches('M');
                        if let Ok(size) = size_str.parse::<f64>() {
                            if available.ends_with('T') {
                                return size * 1024.0;
                            } else if available.ends_with('M') {
                                return size / 1024.0;
                            } else {
                                return size;
                            }
                        }
                    }
                }
            }
            Err(_) => {}
        }
        0.0
    }
    
    fn get_python_version() -> Option<String> {
        match Command::new("python3").args(&["--version"]).output() {
            Ok(output) => {
                if output.status.success() {
                    let version = String::from_utf8_lossy(&output.stdout);
                    return Some(version.trim().to_string());
                }
            }
            Err(_) => {}
        }
        None
    }
    
    fn get_rust_version() -> Option<String> {
        match Command::new("rustc").args(&["--version"]).output() {
            Ok(output) => {
                if output.status.success() {
                    let version = String::from_utf8_lossy(&output.stdout);
                    return Some(version.trim().to_string());
                }
            }
            Err(_) => {}
        }
        None
    }
    
    fn get_cuda_version() -> Option<String> {
        match Command::new("nvcc").args(&["--version"]).output() {
            Ok(output) => {
                if output.status.success() {
                    let info = String::from_utf8_lossy(&output.stdout);
                    for line in info.lines() {
                        if line.contains("release") {
                            return Some(line.trim().to_string());
                        }
                    }
                }
            }
            Err(_) => {}
        }
        None
    }
    
    fn get_rocm_version() -> Option<String> {
        match Command::new("rocminfo").output() {
            Ok(output) => {
                if output.status.success() {
                    return Some("ROCm Available".to_string());
                }
            }
            Err(_) => {}
        }
        None
    }
    
    pub fn evaluate_readiness(specs: &SystemSpecs) -> ReadinessLevel {
        let mut score = 0;
        
        // CPU scoring (max 20 points)
        if specs.cpu_cores >= 16 { score += 20; }
        else if specs.cpu_cores >= 8 { score += 15; }
        else if specs.cpu_cores >= 4 { score += 10; }
        else { score += 5; }
        
        // RAM scoring (max 30 points)
        if specs.total_ram_gb >= 64.0 { score += 30; }
        else if specs.total_ram_gb >= 32.0 { score += 25; }
        else if specs.total_ram_gb >= 16.0 { score += 20; }
        else if specs.total_ram_gb >= 8.0 { score += 10; }
        else { score += 5; }
        
        // GPU scoring (max 40 points)
        if specs.gpu_info.is_ml_ready {
            if specs.gpu_info.vram_gb >= 24.0 { score += 40; }
            else if specs.gpu_info.vram_gb >= 16.0 { score += 35; }
            else if specs.gpu_info.vram_gb >= 12.0 { score += 30; }
            else if specs.gpu_info.vram_gb >= 8.0 { score += 25; }
            else if specs.gpu_info.vram_gb >= 6.0 { score += 20; }
            else if specs.gpu_info.vram_gb >= 4.0 { score += 15; }
            else { score += 10; }
        } else {
            score += 5; // CPU-only gets minimal points
        }
        
        // Disk space scoring (max 10 points)
        if specs.disk_space_gb >= 500.0 { score += 10; }
        else if specs.disk_space_gb >= 100.0 { score += 8; }
        else if specs.disk_space_gb >= 50.0 { score += 5; }
        else { score += 2; }
        
        // Determine readiness level based on total score
        match score {
            90..=100 => ReadinessLevel::Excellent,
            70..=89 => ReadinessLevel::Good,
            50..=69 => ReadinessLevel::Adequate,
            30..=49 => ReadinessLevel::Poor,
            _ => ReadinessLevel::Insufficient,
        }
    }
    
    pub fn get_model_recommendations(specs: &SystemSpecs) -> Vec<ModelRecommendation> {
        let mut recommendations = Vec::new();
        
        if specs.gpu_info.vram_gb >= 24.0 && specs.total_ram_gb >= 32.0 {
            recommendations.push(ModelRecommendation {
                model_size: "Large (70B+ parameters)".to_string(),
                examples: vec![
                    "Llama 2 70B".to_string(),
                    "Code Llama 70B".to_string(),
                    "Mixtral 8x7B".to_string(),
                ],
                ram_required: "32-64GB".to_string(),
                vram_required: "24-48GB".to_string(),
                inference_speed: "Fast".to_string(),
            });
        }
        
        if specs.gpu_info.vram_gb >= 12.0 && specs.total_ram_gb >= 16.0 {
            recommendations.push(ModelRecommendation {
                model_size: "Medium (13B-34B parameters)".to_string(),
                examples: vec![
                    "Llama 2 13B".to_string(),
                    "Code Llama 13B".to_string(),
                    "Vicuna 13B".to_string(),
                    "WizardCoder 15B".to_string(),
                ],
                ram_required: "16-32GB".to_string(),
                vram_required: "12-20GB".to_string(),
                inference_speed: "Good".to_string(),
            });
        }
        
        if specs.gpu_info.vram_gb >= 6.0 || specs.total_ram_gb >= 16.0 {
            recommendations.push(ModelRecommendation {
                model_size: "Small (7B parameters)".to_string(),
                examples: vec![
                    "Llama 2 7B".to_string(),
                    "Code Llama 7B".to_string(),
                    "Mistral 7B".to_string(),
                    "Phi-2 (2.7B)".to_string(),
                ],
                ram_required: "8-16GB".to_string(),
                vram_required: "6-12GB".to_string(),
                inference_speed: "Moderate".to_string(),
            });
        }
        
        if specs.total_ram_gb >= 8.0 {
            recommendations.push(ModelRecommendation {
                model_size: "Tiny (1B-3B parameters)".to_string(),
                examples: vec![
                    "TinyLlama 1B".to_string(),
                    "Phi-1.5 (1.3B)".to_string(),
                    "DistilBERT".to_string(),
                ],
                ram_required: "4-8GB".to_string(),
                vram_required: "2-4GB (or CPU)".to_string(),
                inference_speed: "Fast on CPU".to_string(),
            });
        }
        
        recommendations
    }
    
    pub fn print_detailed_report(specs: &SystemSpecs) {
        println!("🖥️  SYSTEM SPECIFICATIONS");
        println!("========================");
        
        println!("🔥 CPU: {} ({} cores)", specs.cpu_model, specs.cpu_cores);
        println!("🧠 RAM: {:.1}GB total, {:.1}GB available", specs.total_ram_gb, specs.available_ram_gb);
        println!("💾 Disk: {:.1}GB free space", specs.disk_space_gb);
        
        println!("\n🎮 GPU INFORMATION");
        println!("==================");
        println!("GPU: {}", specs.gpu_info.gpu_type);
        if specs.gpu_info.vram_gb > 0.0 {
            println!("VRAM: {:.1}GB", specs.gpu_info.vram_gb);
        }
        if let Some(ref compute) = specs.gpu_info.compute_capability {
            println!("Compute: {}", compute);
        }
        if let Some(ref driver) = specs.gpu_info.driver_version {
            println!("Driver: {}", driver);
        }
        println!("ML Ready: {}", if specs.gpu_info.is_ml_ready { "Yes ✅" } else { "No ❌" });
        
        println!("\n🛠️  DEVELOPMENT ENVIRONMENT");
        println!("============================");
        if let Some(ref python) = specs.python_version {
            println!("🐍 {}", python);
        } else {
            println!("🐍 Python: Not installed ❌");
        }
        
        if let Some(ref rust) = specs.rust_version {
            println!("🦀 {}", rust);
        } else {
            println!("🦀 Rust: Not installed ❌");
        }
        
        if let Some(ref cuda) = specs.cuda_version {
            println!("🟢 CUDA: {}", cuda);
        }
        
        if let Some(ref rocm) = specs.rocm_version {
            println!("🔴 {}", rocm);
        }
        
        // Overall readiness
        let readiness = Self::evaluate_readiness(specs);
        println!("\n🎯 ML READINESS ASSESSMENT");
        println!("==========================");
        println!("{} {}", readiness.emoji(), readiness.description());
        
        // Model recommendations
        let recommendations = Self::get_model_recommendations(specs);
        if !recommendations.is_empty() {
            println!("\n🤖 RECOMMENDED MODELS");
            println!("====================");
            for rec in recommendations {
                println!("\n📦 {}", rec.model_size);
                println!("   Examples: {}", rec.examples.join(", "));
                println!("   RAM needed: {}", rec.ram_required);
                println!("   VRAM needed: {}", rec.vram_required);
                println!("   Speed: {}", rec.inference_speed);
            }
        }
        
        // Improvement suggestions
        Self::print_improvement_suggestions(specs);
    }
    
    fn print_improvement_suggestions(specs: &SystemSpecs) {
        println!("\n💡 IMPROVEMENT SUGGESTIONS");
        println!("==========================");
        
        if specs.total_ram_gb < 16.0 {
            println!("🔧 Consider upgrading RAM to at least 16GB for better ML performance");
        }
        
        if !specs.gpu_info.is_ml_ready {
            println!("🔧 A dedicated GPU with 8GB+ VRAM would significantly improve ML capabilities");
            println!("   Recommended: RTX 4070, RTX 4080, or AMD RX 7800 XT+");
        }
        
        if specs.gpu_info.vram_gb > 0.0 && specs.gpu_info.vram_gb < 8.0 {
            println!("🔧 Current GPU has limited VRAM. Consider upgrading for larger models");
        }
        
        if specs.disk_space_gb < 100.0 {
            println!("🔧 More disk space recommended - ML models can be 5-100GB each");
        }
        
        if specs.python_version.is_none() {
            println!("🔧 Install Python 3.8+ for ML frameworks (PyTorch, TensorFlow, etc.)");
        }
        
        if specs.cuda_version.is_none() && specs.gpu_info.gpu_type.contains("NVIDIA") {
            println!("🔧 Install CUDA for GPU acceleration with your NVIDIA card");
        }
        
        if specs.rocm_version.is_none() && specs.gpu_info.gpu_type.contains("AMD") {
            println!("🔧 Install ROCm for GPU acceleration with your AMD card");
        }
        
        println!("\n🚀 NEXT STEPS");
        println!("=============");
        println!("1. Install missing development tools");
        println!("2. Set up ML frameworks (PyTorch/Candle for Rust)");
        println!("3. Download and test a small model first");
        println!("4. Monitor resource usage during inference");
    }
}

pub fn run_ml_readiness_check_with_gpu(gpu_info: GpuInfo) {
    println!("🤖 ML System Readiness Checker");
    println!("==============================\n");
    
    let specs = MLReadinessChecker::check_system_with_gpu(gpu_info);
    MLReadinessChecker::print_detailed_report(&specs);
    
    println!("\n✨ Analysis complete! Use this info to plan your ML setup.");
}