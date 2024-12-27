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
    archive_dir = os.path.join(log_dir, 'archive')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'experiment_{timestamp}.log')
    
    # Archive existing log file if it exists
    if os.path.exists(log_file):
        archive_file = os.path.join(archive_dir, f'experiment_{timestamp}.log')
        os.rename(log_file, archive_file)
        print(f"Archived existing log to {archive_file}")
    
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
    archive_dir = os.path.join(results_dir, 'archive')
    if os.path.exists(results_dir):
        print(f"Archiving results from: {results_dir}")
        try:
            # Create archive directory if it doesn't exist
            os.makedirs(archive_dir, exist_ok=True)
            
            # Move files to archive
            for item in os.listdir(results_dir):
                if item == 'archive':  # Skip archive directory
                    continue
                    
                item_path = os.path.join(results_dir, item)
                archive_path = os.path.join(archive_dir, item)
                
                if os.path.isfile(item_path):
                    os.rename(item_path, archive_path)
                elif os.path.isdir(item_path):
                    shutil.move(item_path, archive_path)
            print("Results archived")
        except Exception as e:
            print(f"Error archiving results: {e}")
    else:
        os.makedirs(results_dir)
        os.makedirs(archive_dir)
        print(f"Created results directory: {results_dir}")
    
    # Handle logs directory
    logs_dir = os.path.join(base_dir, 'logs')
    archive_dir = os.path.join(logs_dir, 'archive')
    if os.path.exists(logs_dir):
        print(f"Archiving logs from: {logs_dir}")
        try:
            # Force close any open log files
            logging.shutdown()
            # Remove all handlers
            for handler in logging.root.handlers[:]:
                handler.close()
                logging.root.removeHandler(handler)
            
            # Create archive directory if it doesn't exist
            os.makedirs(archive_dir, exist_ok=True)
            
            # Move files to archive
            for item in os.listdir(logs_dir):
                if item == 'archive':  # Skip archive directory
                    continue
                    
                item_path = os.path.join(logs_dir, item)
                archive_path = os.path.join(archive_dir, item)
                
                if os.path.isfile(item_path):
                    os.rename(item_path, archive_path)
                elif os.path.isdir(item_path):
                    shutil.move(item_path, archive_path)
            print("Logs archived")
        except Exception as e:
            print(f"Error archiving logs: {e}")
    else:
        os.makedirs(logs_dir)
        os.makedirs(archive_dir)
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
        
        # Archive previous files (in both test and regular modes)
        print("\nArchiving previous files...")
        cleanup_test_files()
        print("Archiving finished - starting evaluation\n")
        
        # Set up logging
        log_file = setup_logging(timestamp)
        print(f"Logging to: {log_file}")
        
        # Initialize configuration manager
        print("Initializing configuration manager...")
        config_manager = ConfigurationManager()
        
        # Get appropriate configurations based on mode
        print("Getting configurations...")
        config = config_manager.get_test_configs() if args.test else config_manager.get_full_configs()
        
        # Set the results filename with the same timestamp
        config.results_file = f'experiment_results_{timestamp}.csv'
        
        print(f"\nEvaluation Plan:")
        print(f"- Datasets: {[d.name for d in config.datasets]}")
        print(f"- Classifiers: {config.classifiers}")
        print(f"- Sample sizes: {TEST_SAMPLE_SIZES if args.test else 'Full'}")
        print(f"- Modes: {config.modes}")
        print(f"- Iterations: {config.iterations}")
        print("\nStarting evaluation...")
        
        # Initialize and run experiment
        runner = ExperimentRunner(config=config, test_mode=args.test)
        runner.run(timestamp=timestamp)  # Pass the timestamp to the runner
        print("\nExperiment completed successfully")
        print(f"Results saved to: {os.path.join('results', config.results_file)}")
        
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
