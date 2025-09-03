mod detect;
mod system_check;

use detect::GpuDetector;
use system_check::run_ml_readiness_check_with_gpu;

fn main() {
    let gpu_info = GpuDetector::detect();
    run_ml_readiness_check_with_gpu(gpu_info);
}
