use std::process::Command;

fn detect_nvidia() -> Option<String> {
    if let Ok(output) = Command::new("nvidia-smi")
        .arg("--query-gpu=name,driver_version,memory.total")
        .arg("--format=csv,noheader")
        .output()
    {
        if output.status.success() {
            let info = String::from_utf8_lossy(&output.stdout);
            return Some(format!("NVIDIA GPU detected: {}", info.trim()));
        }
    }
    None
}

fn detect_amd() -> Option<String> {
    if let Ok(output) = Command::new("rocminfo").output() {
        if output.status.success() {
            let info = String::from_utf8_lossy(&output.stdout);
            // Look for AMD "gfx" architecture line
            let arch_line = info
                .lines()
                .find(|line| line.contains("gfx"))
                .unwrap_or("Unknown AMD GPU");
            return Some(format!("AMD GPU detected: {}", arch_line.trim()));
        }
    }
    None
}

fn detect_intel() -> Option<String> {
    if let Ok(output) = Command::new("clinfo").output() {
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
    None
}

pub fn run() {
    if let Some(nvidia) = detect_nvidia() {
        println!("{}", nvidia);
    } else if let Some(amd) = detect_amd() {
        println!("{}", amd);
    } else if let Some(intel) = detect_intel() {
        println!("{}", intel);
    } else {
        println!("No supported GPU detected.");
    }
}
