import subprocess
import sys
import os

if __name__ == "__main__":
    # Path to the python executable and the training script
    python_exe = sys.executable
    script_path = os.path.join("scripts", "train_v3.py")
    config_path = os.path.join("configs", "v3_nuclear.yaml")

    # Command to run
    cmd = [python_exe, script_path, "--config", config_path]

    print(f"Launching training process in a new terminal...")
    
    # Use CREATE_NEW_CONSOLE to spawn a new detached command prompt window
    CREATE_NEW_CONSOLE = 0x00000010
    
    # Launching the process
    # We use cmd.exe /k to keep the window open after the script finishes or crashes
    full_cmd = ["cmd.exe", "/k"] + cmd
    
    process = subprocess.Popen(
        full_cmd,
        creationflags=CREATE_NEW_CONSOLE,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    print(f"Process launched with PID: {process.pid}")
    print("You can safely close this terminal or let the process run in the background.")
