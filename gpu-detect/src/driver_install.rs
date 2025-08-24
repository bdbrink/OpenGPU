use std::process::Command;
use std::io::{self, Write};

pub enum GpuType {
    Nvidia,
    Amd,
    Intel,
    Unknown,
}

pub struct GpuInfo {
    pub gpu_type: GpuType,
    pub description: String,
}

impl GpuInfo {
    pub fn detect() -> Self {
        if let Some(nvidia_info) = detect_nvidia() {
            return GpuInfo {
                gpu_type: GpuType::Nvidia,
                description: nvidia_info,
            };
        }
        
        if let Some(amd_info) = detect_amd() {
            return GpuInfo {
                gpu_type: GpuType::Amd,
                description: amd_info,
            };
        }
        
        if let Some(intel_info) = detect_intel() {
            return GpuInfo {
                gpu_type: GpuType::Intel,
                description: intel_info,
            };
        }
        
        GpuInfo {
            gpu_type: GpuType::Unknown,
            description: "No supported GPU detected".to_string(),
        }
    }
    
    pub fn setup_environment(&self) -> Result<(), Box<dyn std::error::Error>> {
        match self.gpu_type {
            GpuType::Amd => self.setup_amd(),
            GpuType::Nvidia => self.setup_nvidia(),
            GpuType::Intel => self.setup_intel(),
            GpuType::Unknown => {
                println!("❌ Cannot set up environment for unknown GPU type");
                Ok(())
            }
        }
    }
    
    fn setup_amd(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔧 Setting up AMD ROCm environment...");
        
        // Check if ROCm is already installed
        if self.check_rocm_installed() {
            println!("✅ ROCm is already installed");
            self.verify_amd_setup()?;
            return Ok(());
        }
        
        println!("📦 ROCm not found. Installation steps:");
        println!("1. Add ROCm repository:");
        println!("   wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -");
        println!("   echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.7 jammy main' | sudo tee /etc/apt/sources.list.d/rocm.list");
        
        println!("\n2. Install ROCm:");
        println!("   sudo apt update");
        println!("   sudo apt install rocm-dev rocm-libs rocm-utils");
        
        println!("\n3. Add user to groups:");
        println!("   sudo usermod -a -G render,video $USER");
        
        println!("\n4. Reboot system");
        
        // Ask if user wants to auto-install
        print!("\n🤖 Would you like me to attempt automatic installation? (y/N): ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        
        if input.trim().to_lowercase() == "y" {
            self.auto_install_rocm()?;
        }
        
        Ok(())
    }
    
    fn setup_nvidia(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔧 Setting up NVIDIA CUDA environment...");
        
        if self.check_cuda_installed() {
            println!("✅ CUDA is already installed");
            self.verify_nvidia_setup()?;
            return Ok(());
        }
        
        println!("📦 CUDA not found. Installation steps:");
        println!("1. Install NVIDIA drivers:");
        println!("   sudo apt install nvidia-driver-535");
        
        println!("\n2. Install CUDA toolkit:");
        println!("   sudo apt install cuda-toolkit");
        
        println!("\n3. Add CUDA to PATH (add to ~/.bashrc):");
        println!("   export PATH=/usr/local/cuda/bin:$PATH");
        println!("   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH");
        
        Ok(())
    }
    
    fn setup_intel(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔧 Setting up Intel GPU environment...");
        
        println!("📦 Intel GPU setup steps:");
        println!("1. Install OpenCL runtime:");
        println!("   sudo apt install intel-opencl-icd");
        
        println!("\n2. Install Intel GPU tools:");
        println!("   sudo apt install clinfo intel-gpu-tools");
        
        Ok(())
    }
    
    fn check_rocm_installed(&self) -> bool {
        Command::new("rocminfo")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }
    
    fn check_cuda_installed(&self) -> bool {
        Command::new("nvcc")
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }
    
    fn auto_install_rocm(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🚀 Starting automatic ROCm installation...");
        
        // Add GPG key
        println!("Adding ROCm GPG key...");
        let status = Command::new("sh")
            .arg("-c")
            .arg("wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -")
            .status()?;
        
        if !status.success() {
            return Err("Failed to add ROCm GPG key".into());
        }
        
        // Add repository
        println!("Adding ROCm repository...");
        let status = Command::new("sh")
            .arg("-c")
            .arg("echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.7 jammy main' | sudo tee /etc/apt/sources.list.d/rocm.list")
            .status()?;
        
        if !status.success() {
            return Err("Failed to add ROCm repository".into());
        }
        
        // Update package list
        println!("Updating package list...");
        let status = Command::new("sudo")
            .args(&["apt", "update"])
            .status()?;
        
        if !status.success() {
            return Err("Failed to update package list".into());
        }
        
        // Install ROCm
        println!("Installing ROCm packages (this may take a while)...");
        let status = Command::new("sudo")
            .args(&["apt", "install", "-y", "rocm-dev", "rocm-libs", "rocm-utils"])
            .status()?;
        
        if !status.success() {
            return Err("Failed to install ROCm packages".into());
        }
        
        // Add user to groups
        println!("Adding user to render and video groups...");
        let username = std::env::var("USER").unwrap_or_else(|_| "user".to_string());
        let status = Command::new("sudo")
            .args(&["usermod", "-a", "-G", "render,video", &username])
            .status()?;
        
        if !status.success() {
            return Err("Failed to add user to groups".into());
        }
        
        println!("✅ ROCm installation completed!");
        println!("⚠️  Please reboot your system to complete the setup");
        
        Ok(())
    }
    
    fn verify_amd_setup(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔍 Verifying AMD/ROCm setup...");
        
        // Check rocminfo
        match Command::new("rocminfo").output() {
            Ok(output) => {
                if output.status.success() {
                    let info = String::from_utf8_lossy(&output.stdout);
                    if info.contains("gfx") {
                        println!("✅ ROCm is working - GPU detected");
                        
                        // Extract GPU architecture
                        if let Some(gfx_line) = info.lines().find(|line| line.contains("gfx")) {
                            println!("   Architecture: {}", gfx_line.trim());
                        }
                    } else {
                        println!("⚠️  ROCm installed but no GPU detected");
                    }
                } else {
                    println!("❌ rocminfo failed");
                }
            }
            Err(_) => println!("❌ rocminfo not available"),
        }
        
        // Check rocm-smi
        match Command::new("rocm-smi").output() {
            Ok(output) => {
                if output.status.success() {
                    println!("✅ rocm-smi is working");
                } else {
                    println!("⚠️  rocm-smi available but failed");
                }
            }
            Err(_) => println!("⚠️  rocm-smi not available"),
        }
        
        Ok(())
    }
    
    fn verify_nvidia_setup(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔍 Verifying NVIDIA/CUDA setup...");
        
        match Command::new("nvidia-smi").output() {
            Ok(output) => {
                if output.status.success() {
                    println!("✅ nvidia-smi is working");
                } else {
                    println!("❌ nvidia-smi failed");
                }
            }
            Err(_) => println!("❌ nvidia-smi not available"),
        }
        
        match Command::new("nvcc").arg("--version").output() {
            Ok(output) => {
                if output.status.success() {
                    println!("✅ CUDA compiler (nvcc) is available");
                } else {
                    println!("⚠️  nvcc available but failed");
                }
            }
            Err(_) => println!("⚠️  nvcc not available"),
        }
        
        Ok(())
    }
    
    pub fn get_rust_dependencies(&self) -> Vec<&'static str> {
        match self.gpu_type {
            GpuType::Amd => vec![
                "candle-core = { version = \"0.3\", features = [\"cuda\"] }",  // ROCm support via CUDA interface
                "tch = { version = \"0.13\", features = [\"download-libtorch\"] }",  // PyTorch bindings
                "ort = { version = \"1.16\", features = [\"rocm\"] }",  // ONNX Runtime
                "wgpu = \"0.17\"",  // WebGPU for compute shaders
            ],
            GpuType::Nvidia => vec![
                "candle-core = { version = \"0.3\", features = [\"cuda\"] }",
                "tch = { version = \"0.13\", features = [\"cuda\"] }",
                "cudarc = \"0.9\"",  // CUDA bindings
                "ort = { version = \"1.16\", features = [\"cuda\"] }",
            ],
            GpuType::Intel => vec![
                "wgpu = \"0.17\"",
                "ort = { version = \"1.16\", features = [\"openvino\"] }",
            ],
            GpuType::Unknown => vec![
                "candle-core = \"0.3\"",  // CPU only
                "tch = \"0.13\"",  // CPU only
            ],
        }
    }
    
    pub fn print_cargo_dependencies(&self) {
        println!("\n📦 Recommended Cargo.toml dependencies for your {} GPU:", 
                 match self.gpu_type {
                     GpuType::Amd => "AMD",
                     GpuType::Nvidia => "NVIDIA", 
                     GpuType::Intel => "Intel",
                     GpuType::Unknown => "Unknown",
                 });
        
        println!("[dependencies]");
        for dep in self.get_rust_dependencies() {
            println!("{}", dep);
        }
    }
}

// Detection functions (copied from your original code)
fn detect_nvidia() -> Option<String> {
    match Command::new("nvidia-smi")
        .arg("--query-gpu=name,driver_version,memory.total")
        .arg("--format=csv,noheader")
        .output()
    {
        Ok(output) => {
            if output.status.success() {
                let info = String::from_utf8_lossy(&output.stdout);
                if !info.trim().is_empty() {
                    return Some(format!("NVIDIA GPU detected: {}", info.trim()));
                }
            }
        }
        Err(_) => {}
    }
    None
}

fn detect_amd() -> Option<String> {
    match Command::new("rocminfo").output() {
        Ok(output) => {
            if output.status.success() {
                let info = String::from_utf8_lossy(&output.stdout);
                let arch_line = info
                    .lines()
                    .find(|line| line.contains("gfx"))
                    .unwrap_or("Unknown AMD GPU");
                return Some(format!("AMD GPU detected: {}", arch_line.trim()));
            }
        }
        Err(_) => {}
    }
    
    // Fallback to lspci if rocminfo not available
    match Command::new("lspci").output() {
        Ok(output) => {
            if output.status.success() {
                let info = String::from_utf8_lossy(&output.stdout);
                if info.contains("AMD") || info.contains("ATI") {
                    return Some("AMD GPU detected via lspci".to_string());
                }
            }
        }
        Err(_) => {}
    }
    
    None
}

fn detect_intel() -> Option<String> {
    match Command::new("clinfo").output() {
        Ok(output) => {
            if output.status.success() {
                let info = String::from_utf8_lossy(&output.stdout);
                if info.contains("Intel") {
                    let device_line = info
                        .lines()
                        .find(|line| line.contains("Device Name"))
                        .unwrap_or("Unknown Intel GPU");
                    return Some(format!("Intel GPU detected: {}", device_line.trim()));
                }
            }
        }
        Err(_) => {}
    }
    None
}

pub fn run_setup() {
    println!("🚀 GPU Setup Assistant");
    println!("=======================\n");
    
    let gpu_info = GpuInfo::detect();
    println!("Detected: {}\n", gpu_info.description);
    
    match gpu_info.setup_environment() {
        Ok(_) => {
            gpu_info.print_cargo_dependencies();
            println!("\n✅ Setup assistant completed!");
        }
        Err(e) => {
            println!("❌ Setup failed: {}", e);
        }
    }
}