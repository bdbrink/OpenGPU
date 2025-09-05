use std::process::Command;
use std::fs;
use crate::detect::GpuInfo;
use serde::{Deserialize, Serialize};
use reqwest;

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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub size_category: String,
    pub parameters: String,
    pub min_vram_gb: f64,
    pub min_ram_gb: f64,
    pub repo_id: Option<String>,
    pub description: String,
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
    pub models: Vec<ModelInfo>,
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

    // Fetch top models dynamically from multiple sources
    pub async fn fetch_top_models() -> Vec<ModelInfo> {
        let mut models = Vec::new();
        
        // Add some current top models with known specs (fallback data)
        let fallback_models = Self::get_fallback_models();
        
        // Try to fetch from HuggingFace API or leaderboard
        match Self::fetch_huggingface_models().await {
            Ok(mut hf_models) => {
                models.append(&mut hf_models);
            }
            Err(e) => {
                println!("⚠️ Failed to fetch latest models from HuggingFace: {}", e);
                println!("📦 Using curated fallback models...");
            }
        }
        
        // If we couldn't fetch online, use fallback
        if models.is_empty() {
            models = fallback_models;
        } else {
            // Merge with some fallback models for completeness
            let mut combined = fallback_models;
            combined.extend(models);
            models = combined;
        }
        
        // Sort by parameter count and deduplicate
        models.sort_by(|a, b| {
            let a_params = Self::extract_parameter_count(&a.parameters);
            let b_params = Self::extract_parameter_count(&b.parameters);
            b_params.partial_cmp(&a_params).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Remove duplicates and limit to top models per category
        Self::deduplicate_and_limit_models(models)
    }
    
    async fn fetch_huggingface_models() -> Result<Vec<ModelInfo>, Box<dyn std::error::Error>> {
        let client = reqwest::Client::new();
        
        // Try to get models from HF API - popular text-generation models
        let url = "https://huggingface.co/api/models?pipeline_tag=text-generation&sort=downloads&direction=-1&limit=50";
        
        let response = client
            .get(url)
            .header("User-Agent", "ML-Readiness-Checker/1.0")
            .send()
            .await?;
            
        if !response.status().is_success() {
            return Err(format!("API request failed: {}", response.status()).into());
        }
        
        let hf_response: serde_json::Value = response.json().await?;
        let mut models = Vec::new();
        
        if let Some(model_array) = hf_response.as_array() {
            for model_data in model_array.iter().take(20) {
                if let Some(model_id) = model_data["id"].as_str() {
                    // Skip if it's not a proper LLM
                    if model_id.contains("embedding") || model_id.contains("tokenizer") {
                        continue;
                    }
                    
                    let model_info = Self::infer_model_specs(model_id);
                    models.push(model_info);
                }
            }
        }
        
        Ok(models)
    }
    
    fn get_fallback_models() -> Vec<ModelInfo> {
        vec![
            // Large models (70B+)
            ModelInfo {
                name: "Llama 3.1 70B".to_string(),
                size_category: "Large".to_string(),
                parameters: "70B".to_string(),
                min_vram_gb: 40.0,
                min_ram_gb: 64.0,
                repo_id: Some("meta-llama/Llama-3.1-70B".to_string()),
                description: "Meta's flagship large model with excellent reasoning".to_string(),
            },
            ModelInfo {
                name: "Qwen2.5 72B".to_string(),
                size_category: "Large".to_string(),
                parameters: "72B".to_string(),
                min_vram_gb: 42.0,
                min_ram_gb: 64.0,
                repo_id: Some("Qwen/Qwen2.5-72B".to_string()),
                description: "Alibaba's advanced multilingual model".to_string(),
            },
            ModelInfo {
                name: "DeepSeek-V3".to_string(),
                size_category: "Large".to_string(),
                parameters: "671B (MoE)".to_string(),
                min_vram_gb: 48.0,
                min_ram_gb: 80.0,
                repo_id: Some("deepseek-ai/DeepSeek-V3".to_string()),
                description: "Mixture of Experts model with excellent performance".to_string(),
            },
            
            // Medium models (13-34B)
            ModelInfo {
                name: "Llama 3.1 8B".to_string(),
                size_category: "Medium".to_string(),
                parameters: "8B".to_string(),
                min_vram_gb: 6.0,
                min_ram_gb: 16.0,
                repo_id: Some("meta-llama/Llama-3.1-8B".to_string()),
                description: "Balanced model good for most tasks".to_string(),
            },
            ModelInfo {
                name: "Qwen2.5 14B".to_string(),
                size_category: "Medium".to_string(),
                parameters: "14B".to_string(),
                min_vram_gb: 10.0,
                min_ram_gb: 20.0,
                repo_id: Some("Qwen/Qwen2.5-14B".to_string()),
                description: "Strong performance in coding and math".to_string(),
            },
            ModelInfo {
                name: "Mistral 7B v0.3".to_string(),
                size_category: "Medium".to_string(),
                parameters: "7B".to_string(),
                min_vram_gb: 5.0,
                min_ram_gb: 12.0,
                repo_id: Some("mistralai/Mistral-7B-v0.3".to_string()),
                description: "Efficient and capable general-purpose model".to_string(),
            },
            
            // Small models (1-7B)
            ModelInfo {
                name: "Phi-3.5 Mini".to_string(),
                size_category: "Small".to_string(),
                parameters: "3.8B".to_string(),
                min_vram_gb: 3.0,
                min_ram_gb: 8.0,
                repo_id: Some("microsoft/Phi-3.5-mini-instruct".to_string()),
                description: "Microsoft's compact but powerful model".to_string(),
            },
            ModelInfo {
                name: "Gemma 2 2B".to_string(),
                size_category: "Small".to_string(),
                parameters: "2B".to_string(),
                min_vram_gb: 2.0,
                min_ram_gb: 6.0,
                repo_id: Some("google/gemma-2-2b".to_string()),
                description: "Google's efficient small model".to_string(),
            },
            ModelInfo {
                name: "SmolLM 1.7B".to_string(),
                size_category: "Small".to_string(),
                parameters: "1.7B".to_string(),
                min_vram_gb: 1.5,
                min_ram_gb: 4.0,
                repo_id: Some("HuggingFaceTB/SmolLM-1.7B".to_string()),
                description: "Compact model for resource-constrained environments".to_string(),
            },
        ]
    }
    
    fn infer_model_specs(model_id: &str) -> ModelInfo {
        let name = model_id.to_string();
        let (parameters, size_category, min_vram, min_ram) = 
            if model_id.contains("70B") || model_id.contains("72B") {
                ("70B+".to_string(), "Large".to_string(), 40.0, 64.0)
            } else if model_id.contains("34B") || model_id.contains("33B") {
                ("34B".to_string(), "Medium".to_string(), 20.0, 32.0)
            } else if model_id.contains("13B") || model_id.contains("14B") {
                ("13-14B".to_string(), "Medium".to_string(), 10.0, 20.0)
            } else if model_id.contains("8B") || model_id.contains("7B") {
                ("7-8B".to_string(), "Medium".to_string(), 6.0, 12.0)
            } else if model_id.contains("3B") || model_id.contains("2.7B") {
                ("3B".to_string(), "Small".to_string(), 2.5, 6.0)
            } else {
                ("Unknown".to_string(), "Small".to_string(), 2.0, 4.0)
            };
            
        ModelInfo {
            name: name.clone(),
            size_category,
            parameters,
            min_vram_gb: min_vram,
            min_ram_gb: min_ram,
            repo_id: Some(name.clone()),
            description: format!("Popular model from {}", model_id.split('/').next().unwrap_or("community")),
        }
    }
    
    fn extract_parameter_count(params: &str) -> f64 {
        let params_lower = params.to_lowercase();
        if let Some(pos) = params_lower.find('b') {
            let number_part = &params_lower[..pos];
            if let Ok(num) = number_part.parse::<f64>() {
                return num;
            }
        }
        0.0
    }
    
    fn deduplicate_and_limit_models(models: Vec<ModelInfo>) -> Vec<ModelInfo> {
        let mut seen_names = std::collections::HashSet::new();
        let mut result = Vec::new();
        
        let mut large_count = 0;
        let mut medium_count = 0;
        let mut small_count = 0;
        
        for model in models {
            if seen_names.contains(&model.name) {
                continue;
            }
            
            let should_include = match model.size_category.as_str() {
                "Large" => {
                    large_count += 1;
                    large_count <= 3
                }
                "Medium" => {
                    medium_count += 1;
                    medium_count <= 4
                }
                "Small" => {
                    small_count += 1;
                    small_count <= 4
                }
                _ => true,
            };
            
            if should_include {
                seen_names.insert(model.name.clone());
                result.push(model);
            }
        }
        
        result
    }
    
    // Rest of the original methods remain the same...
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
    
    pub async fn get_model_recommendations(specs: &SystemSpecs) -> Vec<ModelRecommendation> {
        let mut recommendations = Vec::new();
        let models = Self::fetch_top_models().await;
        
        // Group models by capability requirements
        let large_models: Vec<ModelInfo> = models.iter()
            .filter(|m| m.min_vram_gb >= 24.0 && specs.gpu_info.vram_gb >= m.min_vram_gb && specs.total_ram_gb >= m.min_ram_gb)
            .cloned()
            .take(3)
            .collect();
            
        let medium_models: Vec<ModelInfo> = models.iter()
            .filter(|m| m.min_vram_gb >= 6.0 && m.min_vram_gb < 24.0 && 
                    specs.gpu_info.vram_gb >= m.min_vram_gb && specs.total_ram_gb >= m.min_ram_gb)
            .cloned()
            .take(4)
            .collect();
            
        let small_models: Vec<ModelInfo> = models.iter()
            .filter(|m| m.min_vram_gb < 6.0 && 
                    (specs.gpu_info.vram_gb >= m.min_vram_gb || specs.total_ram_gb >= m.min_ram_gb))
            .cloned()
            .take(4)
            .collect();
        
        if !large_models.is_empty() {
            recommendations.push(ModelRecommendation {
                model_size: "Large (70B+ parameters)".to_string(),
                models: large_models,
                ram_required: "64-128GB".to_string(),
                vram_required: "24-48GB".to_string(),
                inference_speed: "Fast".to_string(),
            });
        }
        
        if !medium_models.is_empty() {
            recommendations.push(ModelRecommendation {
                model_size: "Medium (7B-34B parameters)".to_string(),
                models: medium_models,
                ram_required: "12-32GB".to_string(),
                vram_required: "6-20GB".to_string(),
                inference_speed: "Good".to_string(),
            });
        }
        
        if !small_models.is_empty() {
            recommendations.push(ModelRecommendation {
                model_size: "Small (1B-7B parameters)".to_string(),
                models: small_models,
                ram_required: "4-16GB".to_string(),
                vram_required: "2-6GB (or CPU)".to_string(),
                inference_speed: "Fast on CPU".to_string(),
            });
        }
        
        recommendations
    }
    
    pub async fn print_detailed_report(specs: &SystemSpecs) {
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
        
        // Dynamic model recommendations
        println!("\n🔄 Fetching latest model recommendations...");
        let recommendations = Self::get_model_recommendations(specs).await;
        if !recommendations.is_empty() {
            println!("\n🤖 RECOMMENDED MODELS (Updated {})", 
                chrono::Utc::now().format("%Y-%m-%d"));
            println!("=======================================");
            for rec in recommendations {
                println!("\n📦 {}", rec.model_size);
                println!("   RAM needed: {} | VRAM needed: {} | Speed: {}", 
                         rec.ram_required, rec.vram_required, rec.inference_speed);
                println!("   💎 Top Models:");
                for (i, model) in rec.models.iter().enumerate() {
                    println!("   {}. {} ({})", i + 1, model.name, model.parameters);
                    if let Some(ref repo) = model.repo_id {
                        println!("      🔗 {}", repo);
                    }
                    println!("      📝 {}", model.description);
                }
            }
        } else {
            println!("\n⚠️ No suitable models found for your current hardware configuration");
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
            println!("🔧 More disk space recommended - modern ML models can be 5-100GB each");
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
        println!("5. Check HuggingFace for the latest model releases");
    }
}

pub async fn run_ml_readiness_check_with_gpu(gpu_info: GpuInfo) {
    println!("🤖 Dynamic ML System Readiness Checker");
    println!("======================================\n");
    
    let specs = MLReadinessChecker::check_system_with_gpu(gpu_info);
    MLReadinessChecker::print_detailed_report(&specs).await;
    
    println!("\n✨ Analysis complete! Model recommendations updated from latest sources.");
}