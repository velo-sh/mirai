use std::process::Command;
use std::thread;
use std::time::Duration;

fn main() {
    println!("[watchdog] Starting Mirai Node supervisor...");

    loop {
        println!("[watchdog] Spawning Python Node (uv run main.py)...");
        
        // Use 'uv' to run the Python script. 
        // We assume 'uv' is in the PATH or we use the local path if necessary.
        // For the MVP, we assume 'uv' is available.
        let mut child = Command::new("uv")
            .arg("run")
            .arg("main.py")
            .spawn()
            .expect("Failed to start Python Node");

        println!("[watchdog] Node started with PID: {}", child.id());

        // Wait for the process to exit
        let status = child.wait().expect("Failed to wait on child process");

        println!("[watchdog] Node exited with status: {}. Restarting in 2 seconds...", status);
        
        thread::sleep(Duration::from_secs(2));
    }
}
