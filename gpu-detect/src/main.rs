use std::env;
use serde_json::json;
use log::{info, debug};


mod detect;
mod system_check;

fn main() {
    let args: Vec<String> = env::args().collect();
    let output_json = args.contains(&"--json".to_string());

    if !output_json {
        env_logger::init();
    }
    
    // Run GPU detection
    let gpu_info = detect::GpuDetector::detect();
    
    if output_json {
        // Output JSON for Python integration
        let json_output = json!({
            "gpu_type": gpu_info.gpu_type,
            "vram_gb": gpu_info.vram_gb,
            "compute_capability": gpu_info.compute_capability,
            "driver_version": gpu_info.driver_version,
            "is_ml_ready": gpu_info.is_ml_ready,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "detection_method": "rust_gpu_detector"
        });
        
        println!("{}", json_output);
    } else {
        // Regular human-readable output
        println!("GPU Detection Results:");
        println!("=====================");
        println!("GPU: {}", gpu_info.gpu_type);
        if gpu_info.vram_gb > 0.0 {
            println!("VRAM: {:.1}GB", gpu_info.vram_gb);
        }
        if let Some(ref compute) = gpu_info.compute_capability {
            println!("Compute: {}", compute);
        }
        if let Some(ref driver) = gpu_info.driver_version {
            println!("Driver: {}", driver);
        }
        println!("ML Ready: {}", if gpu_info.is_ml_ready { "Yes" } else { "No" });
        
        // Run full readiness check if not in JSON mode
        if let Ok(rt) = tokio::runtime::Runtime::new() {
            rt.block_on(async {
                system_check::run_ml_readiness_check_with_gpu(gpu_info).await;
            });
        }
    }
}