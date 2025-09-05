// detect.rs - FIXED VERSION with proper VRAM parsing
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
        
        // Try multiple methods to get AMD info
        let vram_gb = Self::get_amd_vram();
        let (gpu_name, architecture) = Self::get_amd_gpu_info();
        let driver_version = Self::get_amdgpu_version();
        
        // If we found any AMD GPU info, create the struct
        if vram_gb > 0.0 || gpu_name.is_some() || architecture.is_some() {
            let final_gpu_name = gpu_name.unwrap_or_else(|| {
                if let Some(ref arch) = architecture {
                    format!("AMD GPU ({})", arch)
                } else {
                    "AMD GPU".to_string()
                }
            });
            
            let final_vram = if vram_gb > 0.0 { 
                vram_gb 
            } else {
                // Try to infer VRAM from GPU name
                Self::infer_amd_vram_from_name(&final_gpu_name)
            };
            
            println!("[DEBUG] AMD GPU detected: {} with {:.1}GB VRAM", final_gpu_name, final_vram);
            
            return Some(GpuInfo {
                gpu_type: final_gpu_name,
                vram_gb: final_vram,
                compute_capability: architecture,
                driver_version,
                is_ml_ready: final_vram >= 4.0, // AMD needs at least 4GB for ML
            });
        }
        
        // Fallback: check if AMD GPU exists via lspci
        if let Ok(output) = Command::new("lspci").output() {
            let info = String::from_utf8_lossy(&output.stdout);
            for line in info.lines() {
                if (line.contains("AMD") || line.contains("ATI")) && 
                   (line.contains("VGA") || line.contains("Display")) {
                    println!("[DEBUG] Found AMD GPU via lspci: {}", line);
                    
                    let vram = Self::infer_amd_vram_from_lspci(line);
                    return Some(GpuInfo {
                        gpu_type: format!("AMD GPU ({})", line.split(':').last().unwrap_or("Unknown").trim()),
                        vram_gb: vram,
                        compute_capability: None,
                        driver_version: None,
                        is_ml_ready: vram >= 4.0,
                    });
                }
            }
        }
        
        None
    }
    
    fn get_amd_vram() -> f64 {
        println!("[DEBUG] Trying to detect AMD VRAM...");
        
        // Method 1: Try rocm-smi for VRAM info
        if let Ok(output) = Command::new("rocm-smi").args(&["--showmeminfo", "vram"]).output() {
            if output.status.success() {
                let info = String::from_utf8_lossy(&output.stdout);
                println!("[DEBUG] rocm-smi --showmeminfo vram output:\n{}", info);
                
                // Look for VRAM size in various formats
                for line in info.lines() {
                    if let Some(vram) = Self::extract_memory_from_line(line) {
                        println!("[DEBUG] Found VRAM via rocm-smi: {:.1}GB", vram);
                        return vram;
                    }
                }
            }
        }
        
        // Method 2: Try rocm-smi without specific args
        if let Ok(output) = Command::new("rocm-smi").output() {
            if output.status.success() {
                let info = String::from_utf8_lossy(&output.stdout);
                println!("[DEBUG] rocm-smi output:\n{}", info);
                
                for line in info.lines() {
                    if let Some(vram) = Self::extract_memory_from_line(line) {
                        println!("[DEBUG] Found VRAM via basic rocm-smi: {:.1}GB", vram);
                        return vram;
                    }
                }
            }
        }
        
        // Method 3: Try radeontop for memory info
        if let Ok(output) = Command::new("radeontop").args(&["-d", "-", "-l", "1"]).output() {
            if output.status.success() {
                let info = String::from_utf8_lossy(&output.stdout);
                println!("[DEBUG] radeontop output:\n{}", info);
                
                for line in info.lines() {
                    if let Some(vram) = Self::extract_memory_from_line(line) {
                        println!("[DEBUG] Found VRAM via radeontop: {:.1}GB", vram);
                        return vram;
                    }
                }
            }
        }
        
        // Method 4: Try getting from /sys filesystem (amdgpu specific)
        if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
            for entry in entries.flatten() {
                let path = entry.path();
                let card_name = path.file_name().unwrap_or_default().to_str().unwrap_or("");
                
                if card_name.starts_with("card") && !card_name.contains("-") {
                    // Try multiple possible paths for VRAM info
                    let vram_paths = vec![
                        path.join("device/mem_info_vram_total"),
                        path.join("device/mem_info_vram_used"),
                        path.join("device/gpu_vram_size"),
                        path.join("device/mem_info_gtt_total"),
                    ];
                    
                    for vram_path in vram_paths {
                        if let Ok(content) = std::fs::read_to_string(&vram_path) {
                            println!("[DEBUG] Reading from {:?}: {}", vram_path, content.trim());
                            
                            if let Ok(bytes) = content.trim().parse::<u64>() {
                                let gb = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                                if gb > 1.0 { // Reasonable VRAM size
                                    println!("[DEBUG] Found VRAM via sysfs: {:.1}GB", gb);
                                    return gb;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Method 5: Try glxinfo for GPU memory (less reliable but worth trying)
        if let Ok(output) = Command::new("glxinfo").args(&["-B"]).output() {
            if output.status.success() {
                let info = String::from_utf8_lossy(&output.stdout);
                println!("[DEBUG] glxinfo output available, checking for memory info...");
                
                for line in info.lines() {
                    if let Some(vram) = Self::extract_memory_from_line(line) {
                        println!("[DEBUG] Found VRAM via glxinfo: {:.1}GB", vram);
                        return vram;
                    }
                }
            }
        }
        
        println!("[DEBUG] Could not detect AMD VRAM through any method");
        0.0
    }

    // FIXED: Added the missing extract_memory_from_line function
    fn extract_memory_from_line(line: &str) -> Option<f64> {
        let line_lower = line.to_lowercase();
        
        // Skip irrelevant lines
        if line_lower.contains("used") && !line_lower.contains("total") {
            return None;
        }
        
        // Look for memory patterns with various units
        let memory_keywords = vec![
            "vram", "memory", "mem", "total", "size", "dedicated"
        ];
        
        let has_memory_keyword = memory_keywords.iter().any(|&keyword| line_lower.contains(keyword));
        if !has_memory_keyword {
            return None;
        }
        
        // Simple pattern matching without regex dependency
        // Look for patterns like: "16384 MiB", "16 GB", "16384MB", etc.
        let words: Vec<&str> = line_lower.split_whitespace().collect();
        
        for (i, word) in words.iter().enumerate() {
            // Check if this word contains a number
            let number_part: String = word.chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            
            if !number_part.is_empty() {
                if let Ok(value) = number_part.parse::<f64>() {
                    // Check the rest of the word for unit
                    let unit_part: String = word.chars()
                        .skip_while(|c| c.is_ascii_digit() || *c == '.')
                        .collect();
                    
                    // Also check the next word for unit if current word is just a number
                    let unit = if unit_part.is_empty() && i + 1 < words.len() {
                        words[i + 1]
                    } else {
                        &unit_part
                    };
                    
                    let gb_value = match unit {
                        "mib" | "mi" => value / 1024.0,     // MiB to GB
                        "mb" | "m" => value / 1000.0,       // MB to GB (or assume MiB if large)
                        "gib" | "gi" => value * 1.073741824, // GiB to GB  
                        "gb" | "g" => value,                // GB
                        _ => {
                            // If no unit but large number, assume it might be MB or bytes
                            if value > 100000.0 {
                                // Probably bytes
                                value / (1024.0 * 1024.0 * 1024.0)
                            } else if value > 1000.0 {
                                // Probably MB
                                value / 1024.0
                            } else if value > 1.0 && value < 64.0 {
                                // Probably already GB
                                value
                            } else {
                                continue;
                            }
                        }
                    };
                    
                    // Sanity check - VRAM should be between 1GB and 128GB
                    if gb_value >= 1.0 && gb_value <= 128.0 {
                        println!("[DEBUG] Extracted {:.1}GB from: '{}'", gb_value, line.trim());
                        return Some(gb_value);
                    }
                }
            }
        }
        
        // Try to find patterns like "Memory: 16384M" or similar
        if let Some(colon_pos) = line_lower.find(':') {
            let after_colon = &line_lower[colon_pos + 1..].trim();
            let parts: Vec<&str> = after_colon.split_whitespace().collect();
            
            for part in parts {
                let number_part: String = part.chars()
                    .take_while(|c| c.is_ascii_digit() || *c == '.')
                    .collect();
                
                if !number_part.is_empty() {
                    if let Ok(value) = number_part.parse::<f64>() {
                        let unit_part: String = part.chars()
                            .skip_while(|c| c.is_ascii_digit() || *c == '.')
                            .collect();
                        
                        let gb_value = match unit_part.as_str() {
                            "mib" | "mi" | "m" => value / 1024.0,
                            "mb" => value / 1000.0,
                            "gib" | "gi" | "g" => value,
                            "gb" => value,
                            _ => {
                                if value > 100000.0 {
                                    value / (1024.0 * 1024.0 * 1024.0) // bytes
                                } else if value > 1000.0 {
                                    value / 1024.0 // MB
                                } else if value > 1.0 && value < 64.0 {
                                    value // GB
                                } else {
                                    continue;
                                }
                            }
                        };
                        
                        if gb_value >= 1.0 && gb_value <= 128.0 {
                            println!("[DEBUG] Extracted {:.1}GB from colon-separated: '{}'", gb_value, line.trim());
                            return Some(gb_value);
                        }
                    }
                }
            }
        }
        
        None
    }
    
    fn get_amd_gpu_info() -> (Option<String>, Option<String>) {
        if let Ok(output) = Command::new("rocminfo").output() {
            if output.status.success() {
                let info = String::from_utf8_lossy(&output.stdout);
                println!("[DEBUG] rocminfo available, parsing GPU info...");
                
                let mut marketing_name = None;
                let mut gfx_arch = None;
                
                // Parse the rocminfo output for marketing name and architecture
                for line in info.lines() {
                    let line = line.trim();
                    
                    // Look for marketing name (GPU model)
                    if line.starts_with("Marketing Name:") {
                        if let Some(name) = line.split(':').nth(1) {
                            marketing_name = Some(name.trim().to_string());
                            println!("[DEBUG] Found marketing name: {:?}", marketing_name);
                        }
                    }
                    
                    // Look for gfx architecture
                    if line.starts_with("Name:") && line.contains("gfx") {
                        if let Some(arch) = line.split(':').nth(1) {
                            gfx_arch = Some(arch.trim().to_string());
                            println!("[DEBUG] Found gfx architecture: {:?}", gfx_arch);
                        }
                    }
                }
                
                return (marketing_name, gfx_arch);
            }
        }
        
        (None, None)
    }
    
    fn infer_amd_vram_from_name(gpu_name: &str) -> f64 {
        let name_lower = gpu_name.to_lowercase();
        
        // Known VRAM amounts for popular AMD cards
        if name_lower.contains("7900 xtx") { return 24.0; }
        if name_lower.contains("7900 xt") { return 20.0; }
        if name_lower.contains("7800 xt") { return 16.0; }  // Your card!
        if name_lower.contains("7700 xt") { return 12.0; }
        if name_lower.contains("7600") { return 8.0; }
        if name_lower.contains("6900 xt") { return 16.0; }
        if name_lower.contains("6800 xt") { return 16.0; }
        if name_lower.contains("6800") { return 16.0; }
        if name_lower.contains("6700 xt") { return 12.0; }
        if name_lower.contains("6600 xt") { return 8.0; }
        if name_lower.contains("6600") { return 8.0; }
        if name_lower.contains("6500 xt") { return 4.0; }
        if name_lower.contains("5700 xt") { return 8.0; }
        if name_lower.contains("5700") { return 8.0; }
        
        // Default fallback
        4.0
    }
    
    fn infer_amd_vram_from_lspci(lspci_line: &str) -> f64 {
        let line_lower = lspci_line.to_lowercase();
        
        // Try to extract model from lspci line and infer VRAM
        if line_lower.contains("7800 xt") { return 16.0; }
        if line_lower.contains("7900 xtx") { return 24.0; }
        if line_lower.contains("7900 xt") { return 20.0; }
        if line_lower.contains("7700 xt") { return 12.0; }
        if line_lower.contains("6900 xt") || line_lower.contains("6800") { return 16.0; }
        if line_lower.contains("6700 xt") { return 12.0; }
        if line_lower.contains("6600") { return 8.0; }
        
        // Default for unknown AMD cards
        4.0
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
                            
                            let vram = if line_lower.contains("amd") || line_lower.contains("ati") {
                                Self::infer_amd_vram_from_lspci(line)
                            } else {
                                0.0
                            };
                            
                            return Some(GpuInfo {
                                gpu_type: format!("GPU detected via lspci: {}", line.trim()),
                                vram_gb: vram,
                                compute_capability: None,
                                driver_version: None,
                                is_ml_ready: vram >= 4.0,
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