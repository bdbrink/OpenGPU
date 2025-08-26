mod detect;
mod driver_install;

fn main() {
    detect::run();
    driver_install::run_setup()
}
