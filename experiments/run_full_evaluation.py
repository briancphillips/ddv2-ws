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
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.config import DatasetConfig, ExperimentConfig, DATASET_CONFIGS, POISON_RATES, ATTACK_METHODS, TEST_SAMPLE_SIZES, TEST_DATASETS, TEST_ATTACK_METHODS, TEST_POISON_RATES, TEST_LABEL_FLIP_TYPES, CLASSIFIERS, EXPERIMENT_MODES
from dynadetectv2.core.dataset import DatasetHandler
from dynadetectv2.core.models import ModelFactory

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

def create_dataset_configs():
    """Create dataset configurations for evaluation."""
    configs = []
    
    # Common test parameters
    sample_size = 1000
    poison_rates = [0.05]  # Start with 5% poison rate
    
    # Configurations for each dataset
    datasets = ['CIFAR100', 'GTSRB', 'ImageNette']  # Add ImageNette back
    for dataset_name in datasets:
        for poison_rate in poison_rates:
            # Label flipping configuration
            config = DatasetConfig(
                name=dataset_name,
                dataset_type='image',
                sample_size=sample_size,
                poison_rate=poison_rate,
                modification_method='label_flipping',
                attack_params={'flip_type': 'random_to_random'},
                metric='accuracy'
            )
            configs.append(config)
            
            # PGD attack configuration
            config = DatasetConfig(
                name=dataset_name,
                dataset_type='image',
                sample_size=sample_size,
                poison_rate=poison_rate,
                modification_method='pgd',
                attack_params={
                    'eps': 0.1,
                    'alpha': 0.01,
                    'iters': 40
                },
                metric='accuracy'
            )
            configs.append(config)
    
    return configs

def get_test_configs():
    """Get test configurations for quick evaluation."""
    configs = []
    
    # Use all datasets
    for dataset_type, dataset_list in TEST_DATASETS.items():
        for dataset_name in dataset_list:
            # Get dataset specs
            dataset_specs = DATASET_CONFIGS[dataset_name]
            
            # Use test sample size
            sample_size = TEST_SAMPLE_SIZES[dataset_name]
            
            # For each attack method
            for attack_method in TEST_ATTACK_METHODS:
                # For each poison rate
                for poison_rate in TEST_POISON_RATES:
                    # For each mode (regular and dynadetect)
                    for mode in EXPERIMENT_MODES:
                        if attack_method == 'label_flipping':
                            # For each label flipping type
                            for flip_type in TEST_LABEL_FLIP_TYPES:
                                configs.append(
                                    DatasetConfig(
                                        name=dataset_name,
                                        dataset_type=dataset_type,
                                        sample_size=sample_size,
                                        metric='accuracy',
                                        modification_method=attack_method,
                                        poison_rate=poison_rate,
                                        attack_params={'flip_type': flip_type},
                                        mode=mode
                                    )
                                )
                        else:  # PGD attack
                            configs.append(
                                DatasetConfig(
                                    name=dataset_name,
                                    dataset_type=dataset_type,
                                    sample_size=sample_size,
                                    metric='accuracy',
                                    modification_method=attack_method,
                                    poison_rate=poison_rate,
                                    attack_params={
                                        'eps': 0.1,
                                        'alpha': 0.01,
                                        'iters': 40
                                    },
                                    mode=mode
                                )
                            )
    
    # Return as ExperimentConfig
    return ExperimentConfig(
        datasets=configs,
        classifiers=CLASSIFIERS,  # Use all classifiers
        modes=EXPERIMENT_MODES,
        iterations=1
    )

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
    
    # Clean up results directory
    results_dir = os.path.join(base_dir, 'results')
    if os.path.exists(results_dir):
        print(f"Cleaning up results directory: {results_dir}")
        try:
            shutil.rmtree(results_dir)
            os.makedirs(results_dir)
            print("Results directory cleaned and recreated")
        except Exception as e:
            print(f"Error cleaning results directory: {e}")
    else:
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")
    
    # Clean up logs directory
    logs_dir = os.path.join(base_dir, 'logs')
    if os.path.exists(logs_dir):
        print(f"Cleaning up logs directory: {logs_dir}")
        try:
            # Force close any open log files
            logging.shutdown()
            # Remove all handlers
            for handler in logging.root.handlers[:]:
                handler.close()
                logging.root.removeHandler(handler)
            
            # Remove the directory
            shutil.rmtree(logs_dir)
            os.makedirs(logs_dir)
            print("Logs directory cleaned and recreated")
        except Exception as e:
            print(f"Error cleaning logs directory: {e}")
    else:
        os.makedirs(logs_dir)
        print(f"Created logs directory: {logs_dir}")
    
    print("Cleanup completed")

def run_evaluation(timestamp, test_mode=False):
    """Run evaluation with all configurations."""
    # Create results directory if it doesn't exist
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Set up results file paths
    results_json = os.path.join(results_dir, f'experiment_results_{timestamp}.json')
    results_csv = os.path.join(results_dir, f'experiment_results_{timestamp}.csv')
    
    # Get configurations based on mode
    if test_mode:
        config = get_test_configs()  # Already returns ExperimentConfig
    else:
        dataset_configs = create_dataset_configs()
        config = ExperimentConfig(
            datasets=dataset_configs,
            classifiers=['LogisticRegression'],
            modes=['regular', 'dynadetect'],
            iterations=1,
            results_file=results_json,
            seed=42
        )
    
    # Log configuration summary
    logging.info("Starting evaluation with configuration:")
    logging.info(f"Number of dataset configs: {len(config.datasets)}")
    logging.info(f"Classifiers: {config.classifiers}")
    logging.info(f"Modes: {config.modes}")
    logging.info(f"Iterations: {config.iterations}")
    logging.info(f"Results will be saved to: {results_json}")
    
    # Run evaluation for each configuration
    all_results = []
    for dataset_config in config.datasets:
        for classifier in config.classifiers:
            for mode in config.modes:
                try:
                    # Create a result dictionary
                    result = {
                        'dataset': dataset_config.name,
                        'classifier': classifier,
                        'mode': mode,
                        'modification_method': dataset_config.modification_method,
                        'total_images': dataset_config.sample_size,
                        'num_poisoned': int(dataset_config.sample_size * dataset_config.poison_rate),
                        'poison_rate': dataset_config.poison_rate,
                        'flip_type': dataset_config.attack_params.get('flip_type', ''),
                        'iteration': 1,
                        'train_metrics': {},
                        'val_metrics': {}
                    }
                    
                    # Run evaluation
                    eval_result = run_evaluation_for_config(dataset_config, classifier, mode, timestamp)
                    if eval_result is not None:
                        result.update(eval_result)
                        # Save results after each configuration
                        all_results.append(result)
                        save_results(all_results, results_json)
                        
                except Exception as e:
                    logging.error(f"Error in configuration {dataset_config.name}:")
                    logging.error(str(e))
                    continue

def run_evaluation_for_config(dataset_config, classifier_name, mode, timestamp):
    """Run evaluation for a specific configuration."""
    try:
        # Initialize dataset handler
        handler = DatasetHandler(dataset_config.name)
        
        # Get train and validation datasets
        train_dataset = handler.get_train_dataset(sample_size=dataset_config.sample_size)
        val_dataset = handler.get_val_dataset(sample_size=dataset_config.sample_size)
        
        # Apply label flipping if specified
        if dataset_config.poison_rate > 0 and dataset_config.modification_method == 'label_flipping':
            flip_type = dataset_config.attack_params.get('flip_type', 'random_to_random')
            train_dataset = handler.apply_label_flipping(
                train_dataset, 
                poison_rate=dataset_config.poison_rate,
                flip_type=flip_type
            )
        
        # Extract features
        X_train, y_train = handler.extract_features(train_dataset)
        X_val, y_val = handler.extract_features(val_dataset)
        
        # Initialize and train classifier
        classifier = ModelFactory.get_classifier(classifier_name)
        logging.info(f"Training {classifier_name}...")
        start_time = time.time()
        classifier.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        start_time = time.time()
        train_preds = classifier.predict(X_train)
        val_preds = classifier.predict(X_val)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        train_metrics = evaluate_metrics(y_train, train_preds)
        val_metrics = evaluate_metrics(y_val, val_preds)
        
        # Store results
        results = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'latency': training_time + prediction_time
        }
        
        logging.info("\nEvaluation Results:")
        logging.info(f"Dataset: {dataset_config.name}")
        logging.info(f"Classifier: {classifier_name}")
        logging.info(f"Mode: {mode}")
        logging.info(f"Latency: {results['latency']:.4f}s\n")
        
        return results
    except Exception as e:
        logging.error(f"Error in configuration {dataset_config.name}:")
        logging.error(str(e))
        return None

def evaluate_metrics(y_true, y_pred):
    """Calculate evaluation metrics."""
    logging.info(f"Calculating metrics for {len(y_true)} samples...")
    try:
        # Calculate overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Calculate per-class metrics
        unique_classes = np.unique(y_true)
        per_class_metrics = {}
        for cls in unique_classes:
            cls_mask = y_true == cls
            if cls_mask.sum() > 0:
                cls_precision = precision_score(y_true == cls, y_pred == cls, zero_division=0)
                cls_recall = recall_score(y_true == cls, y_pred == cls, zero_division=0)
                cls_f1 = f1_score(y_true == cls, y_pred == cls, zero_division=0)
                per_class_metrics[str(cls)] = {
                    'precision': float(cls_precision),
                    'recall': float(cls_recall),
                    'f1': float(cls_f1)
                }
        
        # Calculate macro averages
        macro_precision = np.mean([m['precision'] for m in per_class_metrics.values()])
        macro_recall = np.mean([m['recall'] for m in per_class_metrics.values()])
        macro_f1 = np.mean([m['f1'] for m in per_class_metrics.values()])
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'per_class': per_class_metrics,
            'num_classes': len(unique_classes),
            'num_samples': len(y_true),
            'class_distribution': {str(cls): int((y_true == cls).sum()) for cls in unique_classes}
        }
        
        logging.info(f"Overall metrics - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        logging.info(f"Macro metrics - Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1: {macro_f1:.4f}")
        logging.info(f"Number of unique classes: {len(unique_classes)}")
        logging.info(f"Average samples per class: {len(y_true) / len(unique_classes):.1f}")
        return metrics
    except Exception as e:
        logging.error(f"Error calculating metrics: {str(e)}")
        return None

def save_results(results, results_file):
    """Save results to JSON and CSV files."""
    import json
    import os
    import pandas as pd
    import time
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    # Load existing results if file exists
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = []
    
    # Append new results
    all_results.extend(results)
    
    # Save updated results to JSON
    json_file = results_file
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    logging.info(f"Results saved to {json_file}")
    
    # Save to CSV
    csv_file = results_file.replace('.json', '.csv')
    csv_rows = []
    for result in all_results:
        # Basic metrics
        flat_result = {
            'Dataset': result['dataset'],
            'Classifier': result['classifier'],
            'Mode': result['mode'],
            'Modification_Method': result['modification_method'],
            'Total_Images': result['total_images'],
            'Num_Poisoned': result['num_poisoned'],
            'Poison_Rate': result.get('poison_rate', 0.0),
            'Flip_Type': result.get('flip_type', ''),
            'Accuracy': result['train_metrics'].get('accuracy', 0.0),
            'Precision': result['train_metrics'].get('macro_precision', 0.0),
            'Recall': result['train_metrics'].get('macro_recall', 0.0),
            'F1-Score': result['train_metrics'].get('macro_f1', 0.0),
            'Training_Time': result.get('training_time', 0.0),
            'Prediction_Time': result.get('prediction_time', 0.0),
            'Total_Latency': result.get('latency', 0.0)
        }
        
        # Add per-class metrics if available
        per_class = result['train_metrics'].get('per_class', {})
        for i in range(100):  # Maximum 100 classes
            class_key = str(i)
            if class_key in per_class:
                metrics = per_class[class_key]
                flat_result[f'Class_{i}_Accuracy'] = metrics.get('accuracy', 0.0)
                flat_result[f'Class_{i}_Precision'] = metrics.get('precision', 0.0)
                flat_result[f'Class_{i}_Recall'] = metrics.get('recall', 0.0)
                flat_result[f'Class_{i}_F1'] = metrics.get('f1', 0.0)
        
        csv_rows.append(flat_result)
    
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(csv_rows)
    df.to_csv(csv_file, index=False)
    logging.info(f"Results saved to {csv_file}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run evaluation of DynaDetect v2')
    parser.add_argument('--test', action='store_true', help='Run in test mode with reduced configurations')
    return parser.parse_args()

def main():
    """Main function to run the evaluation."""
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Clean up previous test files if in test mode
    if args.test:
        print("\nRunning in test mode - cleaning up previous files...")
        cleanup_test_files()
        print("Cleanup finished - starting evaluation\n")
    
    # Set up logging
    setup_logging(timestamp)
    
    # Run the evaluation
    run_evaluation(timestamp, test_mode=args.test)

if __name__ == "__main__":
    main()
