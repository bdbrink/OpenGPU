use std::process::Command;

fn detect_nvidia() -> Option<String> {
    println!("[DEBUG] Attempting to detect NVIDIA GPU...");
    
    match Command::new("nvidia-smi")
        .arg("--query-gpu=name,driver_version,memory.total")
        .arg("--format=csv,noheader")
        .output()
    {
        Ok(output) => {
            println!("[DEBUG] nvidia-smi command executed successfully");
            println!("[DEBUG] Exit status: {}", output.status);
            println!("[DEBUG] Stdout length: {} bytes", output.stdout.len());
            println!("[DEBUG] Stderr length: {} bytes", output.stderr.len());
            
            if !output.stderr.is_empty() {
                println!("[DEBUG] Stderr content: {}", String::from_utf8_lossy(&output.stderr));
            }
            
            if output.status.success() {
                let info = String::from_utf8_lossy(&output.stdout);
                println!("[DEBUG] NVIDIA output: '{}'", info);
                if info.trim().is_empty() {
                    println!("[DEBUG] NVIDIA output is empty despite success status");
                    return None;
                }
                return Some(format!("NVIDIA GPU detected: {}", info.trim()));
            } else {
                println!("[DEBUG] nvidia-smi failed with exit code: {}", output.status);
            }
        }
        Err(e) => {
            println!("[DEBUG] Failed to execute nvidia-smi: {} ({})", e, e.kind());
            match e.kind() {
                std::io::ErrorKind::NotFound => println!("[DEBUG] nvidia-smi not found in PATH"),
                std::io::ErrorKind::PermissionDenied => println!("[DEBUG] Permission denied to execute nvidia-smi"),
                _ => println!("[DEBUG] Other error type: {:?}", e.kind()),
            }
        }
    }
    None
}

fn detect_amd() -> Option<String> {
    println!("[DEBUG] Attempting to detect AMD GPU...");
    
    match Command::new("rocminfo").output() {
        Ok(output) => {
            println!("[DEBUG] rocminfo command executed successfully");
            println!("[DEBUG] Exit status: {}", output.status);
            println!("[DEBUG] Stdout length: {} bytes", output.stdout.len());
            println!("[DEBUG] Stderr length: {} bytes", output.stderr.len());
            
            if !output.stderr.is_empty() {
                println!("[DEBUG] Stderr content: {}", String::from_utf8_lossy(&output.stderr));
            }
            
            if output.status.success() {
                let info = String::from_utf8_lossy(&output.stdout);
                println!("[DEBUG] AMD output (first 500 chars): '{}'", 
                         info.chars().take(500).collect::<String>());
                
                // Look for AMD "gfx" architecture line
                let arch_line = info
                    .lines()
                    .find(|line| {
                        let contains_gfx = line.contains("gfx");
                        if contains_gfx {
                            println!("[DEBUG] Found gfx line: '{}'", line);
                        }
                        contains_gfx
                    })
                    .unwrap_or("Unknown AMD GPU");
                    
                if arch_line == "Unknown AMD GPU" {
                    println!("[DEBUG] No 'gfx' line found in rocminfo output");
                    println!("[DEBUG] Searching for other AMD indicators...");
                    
                    // Look for other AMD indicators
                    let amd_lines: Vec<&str> = info
                        .lines()
                        .filter(|line| line.to_lowercase().contains("amd") || 
                                      line.to_lowercase().contains("radeon") ||
                                      line.to_lowercase().contains("gpu"))
                        .collect();
                    
                    if !amd_lines.is_empty() {
                        println!("[DEBUG] Found AMD-related lines: {:?}", amd_lines);
                    }
                }
                
                return Some(format!("AMD GPU detected: {}", arch_line.trim()));
            } else {
                println!("[DEBUG] rocminfo failed with exit code: {}", output.status);
            }
        }
        Err(e) => {
            println!("[DEBUG] Failed to execute rocminfo: {} ({})", e, e.kind());
            match e.kind() {
                std::io::ErrorKind::NotFound => println!("[DEBUG] rocminfo not found in PATH"),
                std::io::ErrorKind::PermissionDenied => println!("[DEBUG] Permission denied to execute rocminfo"),
                _ => println!("[DEBUG] Other error type: {:?}", e.kind()),
            }
        }
    }
    None
}

fn detect_intel() -> Option<String> {
    println!("[DEBUG] Attempting to detect Intel GPU...");
    
    match Command::new("clinfo").output() {
        Ok(output) => {
            println!("[DEBUG] clinfo command executed successfully");
            println!("[DEBUG] Exit status: {}", output.status);
            println!("[DEBUG] Stdout length: {} bytes", output.stdout.len());
            println!("[DEBUG] Stderr length: {} bytes", output.stderr.len());
            
            if !output.stderr.is_empty() {
                println!("[DEBUG] Stderr content: {}", String::from_utf8_lossy(&output.stderr));
            }
            
            if output.status.success() {
                let info = String::from_utf8_lossy(&output.stdout);
                println!("[DEBUG] Intel output (first 500 chars): '{}'", 
                         info.chars().take(500).collect::<String>());
                
                if info.contains("Intel") {
                    println!("[DEBUG] Found 'Intel' in clinfo output");
                    
                    let device_line = info
                        .lines()
                        .find(|line| {
                            let contains_device = line.contains("Device Name");
                            if contains_device {
                                println!("[DEBUG] Found device name line: '{}'", line);
                            }
                            contains_device
                        })
                        .unwrap_or("Unknown Intel GPU");
                        
                    if device_line == "Unknown Intel GPU" {
                        println!("[DEBUG] No 'Device Name' line found, searching for other Intel info...");
                        
                        let intel_lines: Vec<&str> = info
                            .lines()
                            .filter(|line| line.to_lowercase().contains("intel"))
                            .collect();
                            
                        if !intel_lines.is_empty() {
                            println!("[DEBUG] Found Intel-related lines: {:?}", intel_lines);
                        }
                    }
                    
                    return Some(format!("Intel GPU detected: {}", device_line.trim()));
                } else {
                    println!("[DEBUG] No 'Intel' found in clinfo output");
                }
            } else {
                println!("[DEBUG] clinfo failed with exit code: {}", output.status);
            }
        }
        Err(e) => {
            println!("[DEBUG] Failed to execute clinfo: {} ({})", e, e.kind());
            match e.kind() {
                std::io::ErrorKind::NotFound => println!("[DEBUG] clinfo not found in PATH"),
                std::io::ErrorKind::PermissionDenied => println!("[DEBUG] Permission denied to execute clinfo"),
                _ => println!("[DEBUG] Other error type: {:?}", e.kind()),
            }
        }
    }
    None
}

// Alternative detection methods for when the primary tools aren't available
fn detect_fallback() -> Option<String> {
    println!("[DEBUG] Attempting fallback detection methods...");
    
    // Try lspci for PCI devices
    match Command::new("lspci").arg("-v").output() {
        Ok(output) => {
            if output.status.success() {
                let info = String::from_utf8_lossy(&output.stdout);
                println!("[DEBUG] lspci available, searching for GPU info...");
                
                let gpu_lines: Vec<&str> = info
                    .lines()
                    .filter(|line| {
                        let line_lower = line.to_lowercase();
                        line_lower.contains("vga") || 
                        line_lower.contains("3d controller") ||
                        line_lower.contains("display controller")
                    })
                    .collect();
                
                if !gpu_lines.is_empty() {
                    println!("[DEBUG] Found GPU lines via lspci: {:?}", gpu_lines);
                    return Some(format!("GPU detected via lspci: {}", gpu_lines.join(", ")));
                }
            }
        }
        Err(e) => {
            println!("[DEBUG] lspci not available: {}", e);
        }
    }
    
    // Try checking /proc/driver/nvidia/version for NVIDIA
    if let Ok(content) = std::fs::read_to_string("/proc/driver/nvidia/version") {
        println!("[DEBUG] Found NVIDIA driver info in /proc: {}", content.trim());
        return Some(format!("NVIDIA driver detected via /proc: {}", content.trim()));
    } else {
        println!("[DEBUG] /proc/driver/nvidia/version not found");
    }
    
    None
}

pub fn run() {
    println!("[INFO] Starting GPU detection...");
    println!("[DEBUG] System info - OS: {}", std::env::consts::OS);
    
    if let Some(nvidia) = detect_nvidia() {
        println!("{}", nvidia);
    } else if let Some(amd) = detect_amd() {
        println!("{}", amd);
    } else if let Some(intel) = detect_intel() {
        println!("{}", intel);
    } else if let Some(fallback) = detect_fallback() {
        println!("{}", fallback);
    } else {
        println!("No supported GPU detected.");
        println!("[DEBUG] All detection methods failed. Consider:");
        println!("[DEBUG] - Installing nvidia-smi (NVIDIA drivers)");
        println!("[DEBUG] - Installing ROCm tools (for AMD)");
        println!("[DEBUG] - Installing OpenCL tools (clinfo)");
        println!("[DEBUG] - Checking if GPU drivers are installed");
    }
}