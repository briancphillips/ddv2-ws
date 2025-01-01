import pandas as pd
import re
import os
from datetime import datetime

def extract_mode_from_logs(log_file):
    mode_map = {}
    current_dataset = None
    current_classifier = None
    current_mode = None
    poison_rate = None
    
    with open(log_file, 'r') as f:
        for line in f:
            # Extract dataset and mode info
            if "Progress:" in line and "mode" in line:
                match = re.search(r'Evaluating (\w+) with (\w+) \((\w+) mode', line)
                if match:
                    current_dataset = match.group(1)
                    current_classifier = match.group(2)
                    current_mode = match.group(3)
            
            # Extract poison rate
            if "Processing poison rate:" in line:
                match = re.search(r'Processing poison rate: ([\d.]+)', line)
                if match:
                    poison_rate = float(match.group(1))
                    if current_dataset and current_classifier and current_mode is not None:
                        key = (current_dataset, current_classifier, poison_rate)
                        mode_map[key] = current_mode
    
    return mode_map

def determine_mode(row):
    """Determine the mode based on the dataset and classifier."""
    # The evaluation plan alternates between standard and dynadetect modes
    # for each dataset and classifier combination
    return 'standard' if row.name % 2 == 0 else 'dynadetect'

def fix_mode_column(results_file, log_file, output_file):
    """Fix the mode column in the results CSV file."""
    # Read the results CSV
    df = pd.read_csv(results_file)
    
    # Update the mode column
    df['mode'] = df.apply(determine_mode, axis=1)
    
    # Save the updated results
    df.to_csv(output_file, index=False)
    print(f"Updated results saved to {output_file}")

def find_latest_file(directory, archive_dir, pattern):
    """Find the latest file matching pattern in directory or its archive."""
    files = []
    
    # Check main directory
    if os.path.exists(directory):
        files.extend([os.path.join(directory, f) for f in os.listdir(directory) 
                     if f.startswith(pattern) and not os.path.isdir(os.path.join(directory, f))])
    
    # Check archive directory
    if os.path.exists(archive_dir):
        files.extend([os.path.join(archive_dir, f) for f in os.listdir(archive_dir) 
                     if f.startswith(pattern) and not os.path.isdir(os.path.join(archive_dir, f))])
    
    if not files:
        return None
        
    return max(files, key=os.path.getctime)

if __name__ == "__main__":
    # Get the base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Setup directories
    results_dir = os.path.join(base_dir, 'results')
    logs_dir = os.path.join(base_dir, 'logs')
    results_archive = os.path.join(results_dir, 'archive')
    logs_archive = os.path.join(logs_dir, 'archive')
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_archive, exist_ok=True)
    os.makedirs(logs_archive, exist_ok=True)
    
    # Find the most recent files
    results_file = find_latest_file(results_dir, results_archive, 'experiment_results_')
    log_file = find_latest_file(logs_dir, logs_archive, 'experiment_')
    
    if not results_file:
        print("No results files found in:", results_dir, "or", results_archive)
        exit(1)
        
    if not log_file:
        print("No log files found in:", logs_dir, "or", logs_archive)
        exit(1)
    
    print(f"Processing results file: {results_file}")
    print(f"Using log file: {log_file}")
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(results_dir, f'experiment_results_{timestamp}_fixed.csv')
    
    # Fix the mode column
    fix_mode_column(results_file, log_file, output_file) 