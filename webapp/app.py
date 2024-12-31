"""
Web interface for DynaDetect v2 experiment configuration and execution.
"""
import os
import sys
import logging
import threading
import traceback
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from dynadetectv2.core.dataset import DatasetHandler
from dynadetectv2.config import (
    ConfigurationManager,
    DATASET_CONFIGS,
    ATTACK_METHODS,
    CLASSIFIERS,
    MODES,
    POISON_RATES,
    DatasetConfig
)
from dynadetectv2.evaluation import ExperimentRunner
from dynadetectv2.core.experimental.monitoring import monitor

# Initialize Flask app with static folder configuration
app = Flask(__name__, 
    static_folder='static',
    static_url_path='/static'
)

# Create static directory if it doesn't exist
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
os.makedirs(static_dir, exist_ok=True)

# Initialize logging
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(base_dir, 'logs')
results_dir = os.path.join(base_dir, 'results')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'webapp.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Global state
experiment_status = {
    'running': False,
    'gpu_in_use': False,
    'current_experiment': None,
    'start_time': None,
    'progress': 0,
    'messages': [],
    'results': None,
    'error': None
}

def validate_dataset(dataset_config):
    """Validate dataset configuration using DatasetHandler.
    
    Args:
        dataset_config (dict): Dataset configuration including name, sample_size, and attack parameters
    """
    try:
        # Validate sample size
        sample_size = None
        if dataset_config.get('sample_size'):
            try:
                sample_size = int(dataset_config['sample_size'])
                if sample_size <= 0:
                    return False, "Sample size must be positive"
            except ValueError:
                return False, "Invalid sample size format"
        
        # Validate attack parameters
        attack_params = dataset_config.get('attack_params', {})
        if not attack_params.get('type') in ATTACK_METHODS:
            return False, f"Invalid attack type: {attack_params.get('type')}"
        
        # Create a DatasetConfig object
        config = DatasetConfig(
            name=dataset_config['name'],
            dataset_type=DATASET_CONFIGS[dataset_config['name']]['type'],
            sample_size=sample_size,
            attack_params={
                'type': attack_params.get('type'),
                'poison_rates': POISON_RATES
            }
        )
        
        # Test dataset loading
        handler = DatasetHandler(config)
        train_dataset = handler.get_train_dataset()
        val_dataset = handler.get_val_dataset()
        
        # Validate dataset sizes
        if sample_size and (len(train_dataset) < sample_size or len(val_dataset) < sample_size):
            return False, f"Sample size {sample_size} is larger than available data"
            
        logger.info(f"Successfully validated dataset {dataset_config['name']}")
        return True, None
        
    except Exception as e:
        error_msg = f"Error validating dataset '{dataset_config['name']}': {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg

def run_experiment_thread(config, timestamp):
    """Run experiment in a separate thread."""
    try:
        # Set up experiment-specific logging
        experiment_log_file = os.path.join(log_dir, f'experiment_{timestamp}.log')
        file_handler = logging.FileHandler(experiment_log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Add handler to root logger to capture all experiment logs
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        
        try:
            # Create configuration
            config_manager = ConfigurationManager()
            experiment_config = config_manager.get_full_configs()
            
            # Update config with user selections
            experiment_config.datasets = []
            for dataset in config['datasets']:
                # Convert sample size to int or None
                sample_size = None
                if dataset.get('sample_size'):
                    try:
                        sample_size = int(dataset['sample_size'])
                    except ValueError:
                        raise ValueError(f"Invalid sample size for dataset {dataset['name']}")
                
                # Create dataset config
                dataset_config = DatasetConfig(
                    name=dataset['name'],
                    dataset_type=DATASET_CONFIGS[dataset['name']]['type'],
                    sample_size=sample_size,
                    attack_params={
                        'type': dataset['attack_params']['type'],
                        'poison_rates': dataset['attack_params']['poison_rates']
                    }
                )
                experiment_config.datasets.append(dataset_config)
            
            experiment_config.classifiers = config['classifiers']
            experiment_config.modes = config['modes']
            experiment_config.iterations = config['iterations']
            experiment_config.results_file = f'experiment_results_{timestamp}.csv'
            
            # Calculate total combinations for progress tracking
            total_combinations = (
                len(experiment_config.datasets) *
                len(experiment_config.classifiers) *
                len(experiment_config.modes) *
                experiment_config.iterations
            )
            current_combination = 0
            
            # Log the configuration
            logger.info(f"Running experiment with configuration:")
            logger.info(f"- Datasets: {[(d.name, d.sample_size, d.attack_params) for d in experiment_config.datasets]}")
            logger.info(f"- Classifiers: {experiment_config.classifiers}")
            logger.info(f"- Modes: {experiment_config.modes}")
            logger.info(f"- Iterations: {experiment_config.iterations}")
            
            # Initialize experiment runner with progress callback
            class ProgressCallback:
                def __call__(self, dataset_name, classifier_name, mode, iteration):
                    nonlocal current_combination
                    current_combination += 1
                    progress = (current_combination / total_combinations) * 100
                    experiment_status['progress'] = progress
                    logger.info(f"\nProgress: {progress:.1f}% - Evaluating {dataset_name} with {classifier_name} ({mode} mode, iteration {iteration + 1}/{experiment_config.iterations})")

            runner = ExperimentRunner(experiment_config)
            runner.progress_callback = ProgressCallback()
            
            # Run experiment
            runner.run(timestamp)
            
            # Update status
            experiment_status['running'] = False
            experiment_status['progress'] = 100
            experiment_status['messages'].append({
                'type': 'success',
                'text': 'Experiment completed successfully',
                'timestamp': datetime.now().isoformat()
            })
            logger.info("Experiment completed successfully")
            
        finally:
            # Clean up logging
            root_logger.removeHandler(file_handler)
            file_handler.close()
        
    except Exception as e:
        error_msg = f"Error during experiment execution: {str(e)}"
        experiment_status['running'] = False
        experiment_status['error'] = error_msg
        experiment_status['messages'].append({
            'type': 'error',
            'text': error_msg,
            'timestamp': datetime.now().isoformat()
        })
        logger.error(error_msg, exc_info=True)

def start_experiment(config):
    """Start experiment using ExperimentRunner."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_status['current_experiment'] = timestamp
        experiment_status['start_time'] = datetime.now()
        experiment_status['running'] = True
        experiment_status['progress'] = 0
        experiment_status['error'] = None
        experiment_status['results'] = None
        experiment_status['messages'] = [{
            'type': 'info',
            'text': 'Starting experiment...',
            'timestamp': datetime.now().isoformat()
        }]
        
        # Start experiment in a separate thread
        thread = threading.Thread(
            target=run_experiment_thread,
            args=(config, timestamp)
        )
        thread.start()
        
        return True, "Experiment started successfully"
    except Exception as e:
        error_msg = f"Error starting experiment: {str(e)}"
        experiment_status['running'] = False
        logger.error(error_msg, exc_info=True)
        return False, error_msg

@app.route('/')
def index():
    try:
        return render_template(
            'index.html',
            datasets=DATASET_CONFIGS,
            attack_methods=ATTACK_METHODS,
            classifiers=CLASSIFIERS,
            modes=MODES,
            poison_rates=POISON_RATES
        )
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f"Internal server error: {str(e)}",
            'traceback': traceback.format_exc()
        }), 500

@app.route('/start', methods=['POST'])
def start():
    try:
        if experiment_status['running']:
            return jsonify({
                'success': False,
                'message': 'An experiment is already running'
            }), 400
        
        if not request.is_json:
            return jsonify({
                'success': False,
                'message': 'Request must be JSON'
            }), 400
        
        config = request.get_json()
        logger.info(f"Received configuration: {config}")
        
        # Validate basic configuration structure
        if not config or 'datasets' not in config or not config['datasets']:
            return jsonify({
                'success': False,
                'message': 'Invalid configuration: at least one dataset is required'
            }), 400
            
        if not config.get('classifiers'):
            return jsonify({
                'success': False,
                'message': 'Invalid configuration: at least one classifier is required'
            }), 400
            
        if not config.get('modes'):
            return jsonify({
                'success': False,
                'message': 'Invalid configuration: at least one mode is required'
            }), 400
            
        # Validate iterations
        try:
            iterations = int(config.get('iterations', 1))
            if iterations <= 0:
                return jsonify({
                    'success': False,
                    'message': 'Invalid configuration: iterations must be positive'
                }), 400
            config['iterations'] = iterations
        except ValueError:
            return jsonify({
                'success': False,
                'message': 'Invalid configuration: iterations must be a number'
            }), 400
        
        # Validate each dataset configuration
        for dataset_config in config['datasets']:
            if 'name' not in dataset_config:
                return jsonify({
                    'success': False,
                    'message': 'Invalid dataset configuration: name is required'
                }), 400
                
            if dataset_config['name'] not in DATASET_CONFIGS:
                return jsonify({
                    'success': False,
                    'message': f"Invalid dataset name: {dataset_config['name']}"
                }), 400
                
            # Ensure attack parameters exist
            if 'attack_params' not in dataset_config:
                dataset_config['attack_params'] = {
                    'type': list(ATTACK_METHODS.keys())[0],
                    'poison_rates': POISON_RATES
                }
                
            dataset_valid, error = validate_dataset(dataset_config)
            if not dataset_valid:
                return jsonify({'success': False, 'message': error}), 400
        
        # Validate classifiers
        for classifier in config['classifiers']:
            if classifier not in CLASSIFIERS:
                return jsonify({
                    'success': False,
                    'message': f"Invalid classifier: {classifier}"
                }), 400
                
        # Validate modes
        for mode in config['modes']:
            if mode not in MODES:
                return jsonify({
                    'success': False,
                    'message': f"Invalid mode: {mode}"
                }), 400
        
        success, message = start_experiment(config)
        return jsonify({'success': success, 'message': message})
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({
            'success': False,
            'message': error_msg,
            'traceback': traceback.format_exc()
        }), 500

@app.route('/status')
def status():
    try:
        return jsonify({
            'running': experiment_status['running'],
            'current_experiment': experiment_status['current_experiment'],
            'start_time': experiment_status['start_time'].isoformat() if experiment_status['start_time'] else None,
            'progress': experiment_status['progress'],
            'messages': experiment_status['messages'],
            'results': experiment_status['results'],
            'error': experiment_status['error']
        })
    except Exception as e:
        error_msg = f"Error getting status: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({
            'success': False,
            'message': error_msg,
            'traceback': traceback.format_exc()
        }), 500

@app.errorhandler(Exception)
def handle_error(error):
    error_msg = f"Unhandled error: {str(error)}"
    logger.error(error_msg, exc_info=True)
    return jsonify({
        'success': False,
        'message': error_msg,
        'traceback': traceback.format_exc()
    }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8504, debug=False) 