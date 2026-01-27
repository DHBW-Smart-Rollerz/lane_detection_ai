import subprocess
import re
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import yaml


class ModelBenchmark:
    def __init__(self, workspace_path: str = "~/Documents/Smartrollerz/smarty_workspace"):
        self.workspace_path = Path(workspace_path).expanduser()
        self.setup_bash = self.workspace_path / "install" / "setup.bash"
        self.results: Dict[str, Dict[str, float]] = {}
    
    def find_model_configs(self) -> List[Path]:
        """Find all model configuration files in the workspace."""
        config_dir = self.workspace_path / "src" / "lane_detection_ai" / "config"
        return sorted(config_dir.glob("**/*.yaml")) if config_dir.exists() else []
    
    def get_model_info_from_config(self, config_path: Path) -> Tuple[str, str]:
        """Extract model name and type (.pth or .hef) from config."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            model_config_path = None
            test_model = None

            if isinstance(config, dict):
                for _, node_config in config.items():
                    if isinstance(node_config, dict) and 'ros__parameters' in node_config:
                        ros_params = node_config['ros__parameters']
                        model_config_path = ros_params.get('model_config_path')
                        test_model = ros_params.get('test_model')
                        break

                if not model_config_path and not test_model:
                    model_config_path = config.get('model_config_path')
                    test_model = config.get('test_model', '')

            if not test_model and model_config_path:
                python_config = self.workspace_path / "src" / "lane_detection_ai" / model_config_path
                if python_config.exists():
                    test_model = self._extract_test_model_from_py(python_config)
                else:
                    print(f"Warning: Python config not found: {python_config}")

            if not test_model:
                print(f"Warning: Could not find 'test_model' in {config_path.name} or its Python config")
                return None, None

            test_model_path = Path(test_model)
            model_type = test_model_path.suffix  # .pth or .hef

            # Build a unique, readable label: "<config_stem>:<model_stem>"
            model_label = f"{config_path.stem}:{test_model_path.stem}"
            return model_label, model_type
        except Exception as e:
            print(f"Error reading config {config_path}: {e}")
            return None, None

    def _extract_test_model_from_py(self, py_config_path: Path) -> str:
        """Extract test_model path from Python config file."""
        try:
            with open(py_config_path, 'r') as f:
                content = f.read()
            
            # Look for test_model assignment
            # Matches: test_model = "path/to/model.pth" or test_model = 'path/to/model.hef'
            import re
            match = re.search(r'test_model\s*=\s*["\']([^"\']+)["\']', content)
            
            if match:
                return match.group(1)
            
            print(f"Warning: Could not find test_model in {py_config_path.name}")
            return None
        except Exception as e:
            print(f"Error reading Python config {py_config_path}: {e}")
            return None
    
    def run_benchmark(self, config_path: Path, duration: int = 10) -> Tuple[float, float]:
        """
        Run a single model benchmark and extract timing metrics.
        
        Args:
            config_path: Path to the model configuration (ROS params file)
            duration: How long to let the node run (seconds)
        
        Returns:
            Tuple of (total_time_avg, inference_time_avg)
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking: {config_path.name}")
        print(f"{'='*60}")
        
        try:
            # Build launch command with params file
            # Use bash explicitly and proper sourcing
            cmd = (
                f"bash -c 'source {self.setup_bash} && "
                f"ros2 launch lane_detection_ai lane_detection_ai.launch.py "
                f"params_file:={config_path}'"
            )
            
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                executable='/bin/bash'
            )
            
            total_times = []
            inference_times = []
            start_time = time.time()
            stabilization_time = 3  # Wait for values to stabilize
            
            # Pattern to match the timing output
            pattern = r"'total':\s*([\d.]+).*?'inference':\s*([\d.]+)"
            
            for line in process.stdout:
                elapsed = time.time() - start_time
                print(f"[{elapsed:.1f}s] {line.rstrip()}")
                
                # Only collect data after stabilization
                if elapsed > stabilization_time:
                    match = re.search(pattern, line)
                    if match:
                        total_times.append(float(match.group(1)))
                        inference_times.append(float(match.group(2)))
                
                # Stop after duration
                if elapsed > stabilization_time + duration:
                    break
            
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            
            # Calculate averages
            total_avg = sum(total_times) / len(total_times) if total_times else 0
            inference_avg = sum(inference_times) / len(inference_times) if inference_times else 0
            
            print(f"Total (avg): {total_avg:.2f}ms | Inference (avg): {inference_avg:.2f}ms")
            return total_avg, inference_avg
            
        except Exception as e:
            print(f"Error during benchmark: {e}")
            return 0, 0
    
    def benchmark_all_models(self, duration: int = 10):
        """Run benchmarks for all configured models."""
        configs = self.find_model_configs()
        
        if not configs:
            print("No model configurations found!")
            return
        
        print(f"Found {len(configs)} model configuration(s)")
        
        for config_path in configs:
            model_name, model_type = self.get_model_info_from_config(config_path)
            
            if not model_name or not model_type:
                print(f"Skipping {config_path.name}: Invalid config")
                continue
            
            total_time, inference_time = self.run_benchmark(config_path, duration)

            # Ensure unique key if same label repeats
            unique_name = model_name
            counter = 2
            while unique_name in self.results:
                unique_name = f"{model_name} #{counter}"
                counter += 1
            
            self.results[unique_name] = {
                'total': total_time,
                'inference': inference_time,
                'type': model_type
            }
    
    def plot_results(self, output_path: str = None):
        """Create a seaborn plot comparing model performance."""
        if not self.results:
            print("No results to plot!")
            return
        
        # Prepare data for plotting
        data = []
        for model_name, metrics in self.results.items():
            data.append({'Model': model_name, 'Time (ms)': metrics['total'], 'Metric': 'Total'})
            data.append({'Model': model_name, 'Time (ms)': metrics['inference'], 'Metric': 'Inference'})
        
        df = pd.DataFrame(data)
        
        # Stable ordering
        model_order = list(self.results.keys())
        metric_order = ['Inference', 'Total']
        
        # Plot
        plt.figure(figsize=(14, 7))
        ax = sns.barplot(
            data=df,
            x='Model',
            y='Time (ms)',
            hue='Metric',
            order=model_order,
            hue_order=metric_order,
            palette={'Inference': '#4c78a8', 'Total': '#f28e2b'}
        )
        
        # Legend
        ax.legend(title='Metric', loc='upper left')
        
        plt.title('Model Performance: Total vs Inference Time', fontsize=14, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Runtime (ms)', fontsize=12)
        plt.xticks(rotation=35, ha='right')
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to: {output_path}")
        
        plt.show()
    
    def save_results(self, output_path: str = None):
        """Save results to JSON file."""
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"Results saved to: {output_path}")


def main():
    benchmark = ModelBenchmark()
    
    # Run benchmarks (10 seconds per model)
    benchmark.benchmark_all_models(duration=10)
    
    # Save and plot results
    results_dir = Path.home() / "Documents" / "Smartrollerz" / "smarty_workspace" / "results"
    benchmark.save_results(results_dir / "model_performance.json")
    benchmark.plot_results(results_dir / "model_performance.png")


if __name__ == "__main__":
    main()