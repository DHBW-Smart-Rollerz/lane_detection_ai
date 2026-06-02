import os
import subprocess
import time
import re

WORKSPACE_DIR = "/home/smartrollerz/smarty_workspace"
ROS_PARAMS_FILE = os.path.join(WORKSPACE_DIR, "src/lane_detection_ai/config/ros_params.yaml")
RESULTS_FILE = os.path.join(WORKSPACE_DIR, "src/lane_detection_ai/scripts/performance_results.txt")

CONFIGS = [
    "config/02.05.2026_config.py",
    "config/16.03.2026_config.py",
    "config/09.04.2026_config.py",
]

def update_ros_params(config_path):
    with open(ROS_PARAMS_FILE, 'r') as f:
        content = f.read()
    
    # Replace the model_config_path value
    new_content = re.sub(
        r'model_config_path:\s*".*?"',
        f'model_config_path: "{config_path}"',
        content
    )
    
    with open(ROS_PARAMS_FILE, 'w') as f:
        f.write(new_content)

def rebuild_node():
    print("Rebuilding node...")
    subprocess.run(["rm", "-rf", "build/lane_detection_ai"], cwd=WORKSPACE_DIR)
    subprocess.run(["make", "build", "PKG:=lane_detection_ai"], cwd=WORKSPACE_DIR)

def kill_hailo_processes():
    print("Killing any existing processes using hailo or node...")
    subprocess.run("pkill -9 -f lane_detection_ai_node", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    subprocess.run("pkill -9 -f hailo", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    time.sleep(2)

def run_and_capture():
    kill_hailo_processes()
    print("Starting node for warmup...")
    process = subprocess.Popen(
        "source install/setup.zsh && ros2 launch lane_detection_ai lane_detection_ai.launch.py debug:=false",
        cwd=WORKSPACE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=True,
        executable="/bin/zsh"
    )
    
    print("Warming up for 30 seconds...")
    time.sleep(30)
    
    print("Capturing 100 entries...")
    totals = []
    inferences = []
    
    total_pattern = re.compile(r"'total':\s*([\d\.]+)")
    inference_pattern = re.compile(r"'inference':\s*([\d\.]+)")
    
    count = 0
    # Read output line by line from the running process
    while count < 100:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            print("Process terminated unexpectedly.")
            break
            
        if "lane_detection_ai_node" in line and "'total':" in line and "'inference':" in line:
            total_match = total_pattern.search(line)
            inference_match = inference_pattern.search(line)
            
            if total_match and inference_match:
                totals.append(float(total_match.group(1)))
                inferences.append(float(inference_match.group(1)))
                count += 1
                if count % 10 == 0:
                    print(f"Captured {count}/100 entries")
                    
    print("Stopping node...")
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        
    kill_hailo_processes()
    
    avg_total = sum(totals) / len(totals) if totals else 0
    avg_inference = sum(inferences) / len(inferences) if inferences else 0
    
    return avg_total, avg_inference

def main():
    results = {}
    
    for config in CONFIGS:
        print(f"\n--- Testing config: {config} ---")
        update_ros_params(config)
        rebuild_node()
        avg_total, avg_inference = run_and_capture()
        print(f"Results for {config}: Avg Total = {avg_total:.2f}, Avg Inference = {avg_inference:.2f}")
        results[config] = {"total": avg_total, "inference": avg_inference}
        
    print("\nWriting results to file...")
    with open(RESULTS_FILE, 'w') as f:
        f.write("Model Performance Comparison Results\n")
        f.write("="*50 + "\n\n")
        for config, data in results.items():
            f.write(f"Config: {config}\n")
            f.write(f"  Average total: {data['total']:.2f}\n")
            f.write(f"  Average inference: {data['inference']:.2f}\n")
            f.write("-" * 30 + "\n")
            
    print(f"Done! Results mapped into {RESULTS_FILE}")

if __name__ == "__main__":
    main()
