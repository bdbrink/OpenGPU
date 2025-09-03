mod detect;
mod driver_install;
mod system_check;

fn main() {
    detect::run();
    //driver_install::run_setup()
    system_check::run_ml_readiness_check()
}
