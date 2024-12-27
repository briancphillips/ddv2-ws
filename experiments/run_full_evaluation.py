"""
Run comprehensive evaluation of DynaDetect v2
"""
import logging
import os
import sys
from datetime import datetime
import shutil
import glob
import subprocess
import signal
import psutil
import argparse
import time

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynadetectv2.config import ConfigurationManager, TEST_SAMPLE_SIZES
from dynadetectv2.evaluation import ExperimentRunner

def setup_logging(timestamp):
    """Set up logging configuration."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'experiment_{timestamp}.log')
    
    # Configure logging with both file and console output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ],
        force=True  # Force reconfiguration
    )
    
    logging.info("Logging setup completed successfully")
    logging.info(f"Log file created at: {log_file}")
    
    return log_file

def cleanup_test_files():
    """Clean up results and logs from previous test runs and kill any running processes."""
    # Kill any running evaluation processes except the current one
    current_pid = os.getpid()
    try:
        # Get all Python processes running run_full_evaluation.py
        ps_output = subprocess.check_output(['pgrep', '-f', 'run_full_evaluation.py']).decode()
        pids = [int(pid) for pid in ps_output.split()]
        
        # Kill all except current process
        for pid in pids:
            if pid != current_pid:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass  # Process already terminated
        
        # Give processes time to clean up
        time.sleep(1)
    except subprocess.CalledProcessError:
        # No processes found
        pass
    except Exception as e:
        print(f"Error managing processes: {e}")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Handle results directory
    results_dir = os.path.join(base_dir, 'results')
    if os.path.exists(results_dir):
        print(f"Cleaning results directory: {results_dir}")
        try:
            # Remove contents but keep directory
            for item in os.listdir(results_dir):
                item_path = os.path.join(results_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            print("Results directory cleaned")
        except Exception as e:
            print(f"Error cleaning results directory: {e}")
    else:
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")
    
    # Handle logs directory
    logs_dir = os.path.join(base_dir, 'logs')
    if os.path.exists(logs_dir):
        print(f"Cleaning logs directory: {logs_dir}")
        try:
            # Force close any open log files
            logging.shutdown()
            # Remove all handlers
            for handler in logging.root.handlers[:]:
                handler.close()
                logging.root.removeHandler(handler)
            
            # Remove contents but keep directory
            for item in os.listdir(logs_dir):
                item_path = os.path.join(logs_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            print("Logs directory cleaned")
        except Exception as e:
            print(f"Error cleaning logs directory: {e}")
    else:
        os.makedirs(logs_dir)
        print(f"Created logs directory: {logs_dir}")
    
    print("Cleanup completed")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run evaluation of DynaDetect v2')
    parser.add_argument('--test', action='store_true', help='Run in test mode with reduced configurations')
    return parser.parse_args()

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    print("\nReceived interrupt signal. Cleaning up...")
    sys.exit(0)

def main():
    """Main function to run the evaluation."""
    try:
        # Set up signal handler
        signal.signal(signal.SIGINT, signal_handler)
        
        args = parse_args()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Clean up previous test files if in test mode
        if args.test:
            print("\nRunning in test mode - cleaning up previous files...")
            cleanup_test_files()
            print("Cleanup finished - starting evaluation\n")
        
        # Set up logging
        log_file = setup_logging(timestamp)
        print(f"Logging to: {log_file}")
        
        # Initialize configuration manager
        print("Initializing configuration manager...")
        config_manager = ConfigurationManager()
        
        # Get appropriate configurations based on mode
        print("Getting configurations...")
        config = config_manager.get_test_configs() if args.test else config_manager.get_full_configs()
        print(f"\nEvaluation Plan:")
        print(f"- Datasets: {[d.name for d in config.datasets]}")
        print(f"- Classifiers: {config.classifiers}")
        print(f"- Sample sizes: {TEST_SAMPLE_SIZES if args.test else 'Full'}")
        print(f"- Modes: {config.modes}")
        print(f"- Iterations: {config.iterations}")
        print("\nStarting evaluation...")
        
        # Initialize and run experiment
        runner = ExperimentRunner(config=config, test_mode=args.test)
        runner.run()
        print("\nExperiment completed successfully")
        print(f"Results saved to: {os.path.join('results', f'experiment_results_{timestamp}.csv')}")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user. Cleaning up...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
