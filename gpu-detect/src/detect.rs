
// detect.rs
use std::process::Command;

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub gpu_type: String,
    pub vram_gb: f64,
    pub compute_capability: Option<String>,
    pub driver_version: Option<String>,
    pub is_ml_ready: bool,
}

impl Default for GpuInfo {
    fn default() -> Self {
        Self {
            gpu_type: "CPU Only".to_string(),
            vram_gb: 0.0,
            compute_capability: None,
            driver_version: None,
            is_ml_ready: false,
        }
    }
}

pub struct GpuDetector;

impl GpuDetector {
    pub fn detect() -> GpuInfo {
        println!("[INFO] Starting GPU detection...");
        println!("[DEBUG] System info - OS: {}", std::env::consts::OS);
        
        if let Some(nvidia_info) = Self::detect_nvidia() {
            println!("[INFO] NVIDIA GPU detected");
            return nvidia_info;
        }
        
        if let Some(amd_info) = Self::detect_amd() {
            println!("[INFO] AMD GPU detected");
            return amd_info;
        }
        
        if let Some(intel_info) = Self::detect_intel() {
            println!("[INFO] Intel GPU detected");
            return intel_info;
        }
        
        if let Some(fallback_info) = Self::detect_fallback() {
            println!("[INFO] GPU detected via fallback method");
            return fallback_info;
        }
        
        println!("[INFO] No supported GPU detected - using CPU only");
        GpuInfo::default()
    }
    
    fn detect_nvidia() -> Option<GpuInfo> {
        println!("[DEBUG] Attempting to detect NVIDIA GPU...");
        
        match Command::new("nvidia-smi")
            .args(&["--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"])
            .output()
        {
            Ok(output) => {
                println!("[DEBUG] nvidia-smi command executed successfully");
                println!("[DEBUG] Exit status: {}", output.status);
                
                if !output.stderr.is_empty() {
                    println!("[DEBUG] Stderr: {}", String::from_utf8_lossy(&output.stderr));
                }
                
                if output.status.success() && !output.stdout.is_empty() {
                    let info = String::from_utf8_lossy(&output.stdout);
                    println!("[DEBUG] NVIDIA output: '{}'", info.trim());
                    
                    if let Some(line) = info.lines().next() {
                        let parts: Vec<&str> = line.split(',').collect();
                        if parts.len() >= 3 {
                            let name = parts[0].trim();
                            let driver = parts[1].trim();
                            let vram_mb: f64 = parts[2].trim().parse().unwrap_or(0.0);
                            
                            let compute_cap = Self::get_nvidia_compute_capability();
                            
                            return Some(GpuInfo {
                                gpu_type: format!("NVIDIA {}", name),
                                vram_gb: vram_mb / 1024.0,
                                compute_capability: compute_cap,
                                driver_version: Some(driver.to_string()),
                                is_ml_ready: vram_mb > 2048.0, // At least 2GB VRAM
                            });
                        }
                    }
                } else {
                    println!("[DEBUG] nvidia-smi failed or returned empty output");
                }
            }
            Err(e) => {
                println!("[DEBUG] Failed to execute nvidia-smi: {} ({})", e, e.kind());
            }
        }
        None
    }
    
    fn get_nvidia_compute_capability() -> Option<String> {
        match Command::new("nvidia-smi")
            .args(&["--query-gpu=compute_cap", "--format=csv,noheader"])
            .output()
        {
            Ok(output) => {
                if output.status.success() {
                    let cap = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    if !cap.is_empty() && cap != "[Not Supported]" {
                        return Some(cap);
                    }
                }
            }
            Err(_) => {}
        }
        None
    }
    
    fn detect_amd() -> Option<GpuInfo> {
        println!("[DEBUG] Attempting to detect AMD GPU...");
        
        // First try rocm-smi for VRAM info
        let vram_gb = Self::get_amd_vram();
        let architecture = Self::get_amd_architecture();
        let driver_version = Self::get_amdgpu_version();
        
        if architecture.is_some() || vram_gb > 0.0 {
            let arch_name = architecture.clone().unwrap_or("Unknown".to_string());
            return Some(GpuInfo {
                gpu_type: format!("AMD GPU ({})", arch_name),
                vram_gb,
                compute_capability: architecture,
                driver_version,
                is_ml_ready: vram_gb > 4096.0 / 1024.0, // AMD needs more VRAM typically (4GB+)
            });
        }
        
        // Fallback: check if AMD GPU exists via lspci
        if let Ok(output) = Command::new("lspci").output() {
            let info = String::from_utf8_lossy(&output.stdout);
            for line in info.lines() {
                if (line.contains("AMD") || line.contains("ATI")) && 
                   (line.contains("VGA") || line.contains("Display")) {
                    println!("[DEBUG] Found AMD GPU via lspci: {}", line);
                    return Some(GpuInfo {
                        gpu_type: "AMD GPU (ROCm not configured)".to_string(),
                        vram_gb: 0.0,
                        compute_capability: None,
                        driver_version: None,
                        is_ml_ready: false,
                    });
                }
            }
        }
        
        None
    }
    
    fn get_amd_vram() -> f64 {
        match Command::new("rocm-smi").args(&["--showmeminfo", "vram"]).output() {
            Ok(output) => {
                if output.status.success() {
                    let info = String::from_utf8_lossy(&output.stdout);
                    println!("[DEBUG] rocm-smi output: {}", info);
                    
                    for line in info.lines() {
                        if line.contains("Total VRAM") || line.contains("Memory") {
                            // Try to extract VRAM size
                            let words: Vec<&str> = line.split_whitespace().collect();
                            for (i, word) in words.iter().enumerate() {
                                if let Ok(size) = word.trim_end_matches("MB").parse::<f64>() {
                                    if word.ends_with("MB") && size > 100.0 { // Reasonable VRAM size
                                        return size / 1024.0; // Convert to GB
                                    }
                                }
                                if let Ok(size) = word.trim_end_matches("GB").parse::<f64>() {
                                    if word.ends_with("GB") && size > 1.0 {
                                        return size;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                println!("[DEBUG] rocm-smi failed: {}", e);
            }
        }
        0.0
    }
    
    fn get_amd_architecture() -> Option<String> {
        match Command::new("rocminfo").output() {
            Ok(output) => {
                if output.status.success() {
                    let info = String::from_utf8_lossy(&output.stdout);
                    println!("[DEBUG] rocminfo available, searching for gfx architecture...");
                    
                    for line in info.lines() {
                        if line.contains("gfx") && (line.contains("Name:") || line.contains("Marketing Name:")) {
                            println!("[DEBUG] Found gfx architecture: {}", line);
                            return Some(line.trim().to_string());
                        }
                    }
                    
                    // Alternative: look for any line containing gfx
                    for line in info.lines() {
                        if line.trim().starts_with("gfx") {
                            println!("[DEBUG] Found gfx line: {}", line);
                            return Some(line.trim().to_string());
                        }
                    }
                }
            }
            Err(e) => {
                println!("[DEBUG] rocminfo failed: {}", e);
            }
        }
        None
    }
    
    fn get_amdgpu_version() -> Option<String> {
        match Command::new("modinfo").arg("amdgpu").output() {
            Ok(output) => {
                if output.status.success() {
                    let info = String::from_utf8_lossy(&output.stdout);
                    for line in info.lines() {
                        if line.starts_with("version:") {
                            return Some(line.split(':').nth(1).unwrap_or("Unknown").trim().to_string());
                        }
                    }
                }
            }
            Err(_) => {}
        }
        None
    }
    
    fn detect_intel() -> Option<GpuInfo> {
        println!("[DEBUG] Attempting to detect Intel GPU...");
        
        match Command::new("clinfo").output() {
            Ok(output) => {
                if output.status.success() {
                    let info = String::from_utf8_lossy(&output.stdout);
                    println!("[DEBUG] clinfo available, searching for Intel devices...");
                    
                    if info.to_lowercase().contains("intel") {
                        // Look for device name
                        for line in info.lines() {
                            if line.contains("Device Name") && line.to_lowercase().contains("intel") {
                                let device_name = line.split(':')
                                    .nth(1)
                                    .unwrap_or("Intel GPU")
                                    .trim();
                                
                                println!("[DEBUG] Found Intel device: {}", device_name);
                                
                                return Some(GpuInfo {
                                    gpu_type: device_name.to_string(),
                                    vram_gb: 0.0, // Shared memory
                                    compute_capability: Some("OpenCL".to_string()),
                                    driver_version: None,
                                    is_ml_ready: false, // Limited ML capability
                                });
                            }
                        }
                        
                        // Fallback if no specific device name found
                        return Some(GpuInfo {
                            gpu_type: "Intel Integrated Graphics".to_string(),
                            vram_gb: 0.0,
                            compute_capability: Some("OpenCL".to_string()),
                            driver_version: None,
                            is_ml_ready: false,
                        });
                    }
                }
            }
            Err(e) => {
                println!("[DEBUG] clinfo failed: {}", e);
            }
        }
        None
    }
    
    fn detect_fallback() -> Option<GpuInfo> {
        println!("[DEBUG] Attempting fallback detection methods...");
        
        // Try lspci for PCI devices
        match Command::new("lspci").arg("-v").output() {
            Ok(output) => {
                if output.status.success() {
                    let info = String::from_utf8_lossy(&output.stdout);
                    println!("[DEBUG] lspci available, searching for GPU info...");
                    
                    for line in info.lines() {
                        let line_lower = line.to_lowercase();
                        if (line_lower.contains("vga") || 
                            line_lower.contains("3d controller") ||
                            line_lower.contains("display controller")) &&
                           (line_lower.contains("nvidia") || 
                            line_lower.contains("amd") || 
                            line_lower.contains("ati") ||
                            line_lower.contains("intel")) {
                            
                            println!("[DEBUG] Found GPU via lspci: {}", line);
                            
                            return Some(GpuInfo {
                                gpu_type: format!("GPU detected via lspci: {}", line.trim()),
                                vram_gb: 0.0,
                                compute_capability: None,
                                driver_version: None,
                                is_ml_ready: false, // Can't determine without proper drivers
                            });
                        }
                    }
                }
            }
            Err(e) => {
                println!("[DEBUG] lspci not available: {}", e);
            }
        }
        
        // Try checking /proc/driver/nvidia/version for NVIDIA
        if let Ok(content) = std::fs::read_to_string("/proc/driver/nvidia/version") {
            println!("[DEBUG] Found NVIDIA driver info in /proc");
            return Some(GpuInfo {
                gpu_type: "NVIDIA GPU (driver detected)".to_string(),
                vram_gb: 0.0,
                compute_capability: None,
                driver_version: Some(content.trim().to_string()),
                is_ml_ready: false, // Can't determine VRAM without nvidia-smi
            });
        }
        
        None
    }
}