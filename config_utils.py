import yaml
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple
import torch

# Set up logging
logger = logging.getLogger(__name__)

class ConfigError(Exception):
    """Exception raised for errors in the configuration."""
    pass

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        ConfigError: If the file cannot be loaded or parsed
    """
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return config
    except yaml.YAMLError as e:
        raise ConfigError(f"Error parsing YAML in {config_path}: {str(e)}")
    except Exception as e:
        raise ConfigError(f"Error loading configuration from {config_path}: {str(e)}")

def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate the configuration for correctness and consistency.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of validation warnings (empty if no warnings)
        
    Raises:
        ConfigError: If the configuration is invalid
    """
    warnings = []
    
    # Validate general section
    if 'general' not in config:
        warnings.append("Missing 'general' section, using defaults")
    
    # Validate IGT module configuration
    validate_igt_config(config, warnings)
    
    # Validate TGI module configuration
    validate_tgi_config(config, warnings)
    
    # Validate blending configuration
    validate_blending_config(config, warnings)
    
    # Validate numerical stability configuration
    validate_numerical_stability_config(config, warnings)
    
    # Validate device configuration
    validate_device_config(config, warnings)
    
    # Validate memory configuration
    validate_memory_config(config, warnings)
    
    # Validate extreme conditions configuration
    validate_extreme_conditions_config(config, warnings)
    
    return warnings

def validate_igt_config(config: Dict[str, Any], warnings: List[str]) -> None:
    """Validate IGT module configuration."""
    igt_config = config.get('igt', {})
    
    # Check if IGT is enabled
    if not igt_config.get('enabled', True):
        warnings.append("IGT module is disabled")
        return
    
    # Validate gamma parameter
    gamma = igt_config.get('gamma', 50.0)
    min_gamma = igt_config.get('min_gamma', 1.0)
    max_gamma = igt_config.get('max_gamma', 100.0)
    
    if not isinstance(gamma, (int, float)):
        raise ConfigError(f"IGT gamma must be a number, got {type(gamma)}")
    
    if gamma < min_gamma or gamma > max_gamma:
        raise ConfigError(f"IGT gamma must be between {min_gamma} and {max_gamma}, got {gamma}")
    
    # Check for numerical safeguards
    if not igt_config.get('subtract_max', True):
        warnings.append("IGT subtract_max is disabled, which may cause numerical instability")

def validate_tgi_config(config: Dict[str, Any], warnings: List[str]) -> None:
    """Validate TGI module configuration."""
    tgi_config = config.get('tgi', {})
    
    # Check if TGI is enabled
    if not tgi_config.get('enabled', True):
        warnings.append("TGI module is disabled")
        return
    
    # Validate temperature parameter
    temperature = tgi_config.get('temperature', 0.1)
    min_temp = tgi_config.get('min_temperature', 0.01)
    max_temp = tgi_config.get('max_temperature', 100.0)
    
    if not isinstance(temperature, (int, float)):
        raise ConfigError(f"TGI temperature must be a number, got {type(temperature)}")
    
    if temperature < min_temp or temperature > max_temp:
        raise ConfigError(f"TGI temperature must be between {min_temp} and {max_temp}, got {temperature}")
    
    # Check for numerical safeguards
    if not tgi_config.get('subtract_max', True):
        warnings.append("TGI subtract_max is disabled, which may cause numerical instability")

def validate_blending_config(config: Dict[str, Any], warnings: List[str]) -> None:
    """Validate feature blending configuration."""
    blend_config = config.get('blending', {})
    
    # Check if blending is enabled
    if not blend_config.get('enabled', True):
        warnings.append("Feature blending is disabled")
        return
    
    # Validate blend weight
    weight = blend_config.get('blend_weight', 0.5)
    min_weight = blend_config.get('min_weight', 0.0)
    max_weight = blend_config.get('max_weight', 1.0)
    
    if not isinstance(weight, (int, float)):
        raise ConfigError(f"Blend weight must be a number, got {type(weight)}")
    
    if weight < min_weight or weight > max_weight:
        raise ConfigError(f"Blend weight must be between {min_weight} and {max_weight}, got {weight}")

def validate_numerical_stability_config(config: Dict[str, Any], warnings: List[str]) -> None:
    """Validate numerical stability configuration."""
    num_config = config.get('numerical_stability', {})
    
    # Validate epsilon
    eps = num_config.get('eps', 1.0e-8)
    if not isinstance(eps, (int, float)) or eps <= 0:
        raise ConfigError(f"Epsilon must be a positive number, got {eps}")
    
    # Validate scaling parameters
    scaling = num_config.get('scaling', {})
    min_scale = scaling.get('min_scale', 1.0e-6)
    max_scale = scaling.get('max_scale', 1.0e6)
    
    if min_scale >= max_scale:
        raise ConfigError(f"min_scale must be less than max_scale, got {min_scale} and {max_scale}")

def validate_device_config(config: Dict[str, Any], warnings: List[str]) -> None:
    """Validate device configuration."""
    device_config = config.get('device', {})
    
    # Check CUDA availability if requested
    use_cuda = device_config.get('use_cuda', True)
    if use_cuda and not torch.cuda.is_available():
        warnings.append("CUDA requested but not available, falling back to CPU")
        device_config['use_cuda'] = False
    
    # Validate device IDs
    device_ids = device_config.get('device_ids', [0])
    if use_cuda and torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        invalid_ids = [did for did in device_ids if did >= num_devices]
        if invalid_ids:
            warnings.append(f"Invalid CUDA device IDs: {invalid_ids}, using available devices only")
            device_config['device_ids'] = [did for did in device_ids if did < num_devices]

def validate_memory_config(config: Dict[str, Any], warnings: List[str]) -> None:
    """Validate memory optimization configuration."""
    memory_config = config.get('memory', {})
    
    # Validate batch sizes
    max_batch_size = memory_config.get('max_batch_size', 1024)
    chunk_size = memory_config.get('chunk_size', 256)
    
    if chunk_size > max_batch_size:
        warnings.append(f"Chunk size ({chunk_size}) greater than max batch size ({max_batch_size}), adjusting")
        memory_config['chunk_size'] = max_batch_size

def validate_extreme_conditions_config(config: Dict[str, Any], warnings: List[str]) -> None:
    """Validate extreme conditions configuration."""
    extreme_config = config.get('extreme_conditions', {})
    
    # Check for high-dimensional features support
    if extreme_config.get('high_dim_features', True):
        thresholds = extreme_config.get('extreme_value_thresholds', {})
        high_dim = thresholds.get('high_dim_threshold', 2048)
        
        if high_dim < 1024:
            warnings.append(f"High-dimensional threshold ({high_dim}) is quite low")

def get_device(config: Dict[str, Any]) -> torch.device:
    """
    Get the PyTorch device from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PyTorch device
    """
    device_config = config.get('device', {})
    use_cuda = device_config.get('use_cuda', True) and torch.cuda.is_available()
    
    if use_cuda:
        device_ids = device_config.get('device_ids', [0])
        if device_ids:
            return torch.device(f'cuda:{device_ids[0]}')
        else:
            return torch.device('cuda:0')
    else:
        return torch.device('cpu')

def get_dtype(config: Dict[str, Any]) -> torch.dtype:
    """
    Get the PyTorch dtype from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PyTorch dtype
    """
    features_config = config.get('features', {})
    dtype_str = features_config.get('dtype', 'float32')
    
    if dtype_str == 'float32':
        return torch.float32
    elif dtype_str == 'float64':
        return torch.float64
    else:
        logger.warning(f"Unsupported dtype: {dtype_str}, using float32")
        return torch.float32

def apply_config_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply default values to missing configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration with defaults applied
    """
    default_config = get_default_config()
    merged_config = merge_configs(default_config, config)
    return merged_config

def merge_configs(default_config: Dict[str, Any], user_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.
    
    Args:
        default_config: Default configuration dictionary
        user_config: User configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    result = default_config.copy()
    
    for key, value in user_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration values.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'general': {
            'seed': 42,
            'cache_dir': "./cache",
            'output_dir': "./output",
            'verbose': True
        },
        'dataset': {
            'name': "caltech101",
            'shots': 16,
            'augment_epoch': 1,
            'load_cache': False
        },
        'model': {
            'backbone': "ViT-B/32",
            'feature_dim': 512,
            'use_clip': True
        },
        'features': {
            'normalize': True,
            'normalize_eps': 1.0e-8,
            'clamp_values': True,
            'min_value': -1.0e6,
            'max_value': 1.0e6,
            'dtype': "float32"
        },
        'igt': {
            'enabled': True,
            'gamma': 50.0,
            'min_gamma': 1.0,
            'max_gamma': 100.0,
            'adaptive_gamma': False,
            'subtract_max': True,
            'use_safe_einsum': True,
            'support_2d': True,
            'support_3d': True,
            'numerical_safeguards': {
                'normalize_before': True,
                'normalize_after': True,
                'clamp_similarity': True,
                'use_stable_softmax': True
            },
            'advanced_options': {
                'weighted_aggregation': True,
                'skip_zero_weights': True
            }
        },
        'tgi': {
            'enabled': True,
            'temperature': 0.1,
            'min_temperature': 0.01,
            'max_temperature': 100.0,
            'adaptive_temperature': False,
            'subtract_max': True,
            'use_safe_einsum': True,
            'attention_mechanism': {
                'scale_by_dim': True,
                'attention_dropout': 0.0,
                'stable_attention': True
            },
            'advanced_options': {
                'residual_connection': False,
                'dynamic_scaling': False
            }
        },
        'blending': {
            'enabled': True,
            'blend_weight': 0.5,
            'min_weight': 0.0,
            'max_weight': 1.0,
            'adaptive_weight': False
        },
        'numerical_stability': {
            'eps': 1.0e-8,
            'check_nan': True,
            'check_inf': True,
            'safe_einsum': True,
            'normalize_inputs': True,
            'normalize_outputs': True,
            'mixed_precision': False,
            'dtypes': {
                'default': "float32",
                'high_precision': "float64",
                'auto_promote': True
            },
            'scaling': {
                'max_scale': 1.0e6,
                'min_scale': 1.0e-6,
                'auto_rescale': True
            }
        },
        'memory': {
            'optimize': True,
            'max_batch_size': 1024,
            'chunk_size': 256,
            'gradient_checkpointing': False,
            'free_memory_threshold': 0.2,
            'cpu_offload': False,
            'pin_memory': True
        },
        'device': {
            'use_cuda': True,
            'device_ids': [0],
            'fallback_to_cpu': True,
            'deterministic': False,
            'benchmark': True,
            'allow_tf32': True
        }
    }

def load_and_validate_config(config_path: str) -> Tuple[Dict[str, Any], List[str]]:
    """
    Load, validate, and apply defaults to a configuration file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Tuple containing (validated config dictionary, list of warnings)
        
    Raises:
        ConfigError: If the configuration is invalid or cannot be loaded
    """
    # Load raw configuration
    try:
        raw_config = load_config(config_path)
    except ConfigError as e:
        logger.error(f"Failed to load configuration: {e}")
        raise
    
    # Apply default values
    config = apply_config_defaults(raw_config)
    
    # Validate configuration
    try:
        warnings = validate_config(config)
    except ConfigError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise
    
    # Configure derived parameters
    config = derive_additional_parameters(config)
    
    logger.info(f"Configuration loaded from {config_path} with {len(warnings)} warnings")
    if warnings:
        for warning in warnings:
            logger.warning(f"Config warning: {warning}")
    
    return config, warnings


def derive_additional_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Derive additional parameters from the configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration with derived parameters
    """
    # Create a copy to avoid modifying the original
    result = config.copy()
    
    # Set up device parameters
    device_config = result.get('device', {}).copy()
    use_cuda = device_config.get('use_cuda', True) and torch.cuda.is_available()
    if use_cuda:
        device_ids = device_config.get('device_ids', [0])
        if device_ids:
            device_config['current_device_str'] = f'cuda:{device_ids[0]}'
            device_config['current_device'] = torch.device(device_config['current_device_str'])
        else:
            device_config['current_device_str'] = 'cuda:0'
            device_config['current_device'] = torch.device('cuda:0')
    else:
        device_config['current_device_str'] = 'cpu'
        device_config['current_device'] = torch.device('cpu')
    result['device'] = device_config
    
    # Set up dtype parameters
    features_config = result.get('features', {}).copy()
    dtype_str = features_config.get('dtype', 'float32')
    if dtype_str == 'float32':
        features_config['torch_dtype'] = torch.float32
    elif dtype_str == 'float64':
        features_config['torch_dtype'] = torch.float64
    else:
        logger.warning(f"Unsupported dtype: {dtype_str}, falling back to float32")
        features_config['torch_dtype'] = torch.float32
    result['features'] = features_config
    
    # Process memory thresholds if CUDA is available
    if use_cuda:
        memory_config = result.get('memory', {}).copy()
        free_memory_threshold = memory_config.get('free_memory_threshold', 0.2)
        try:
            total_memory = torch.cuda.get_device_properties(device_ids[0]).total_memory
            memory_config['total_memory_bytes'] = total_memory
            memory_config['min_free_memory_bytes'] = int(total_memory * free_memory_threshold)
            result['memory'] = memory_config
        except Exception as e:
            logger.warning(f"Failed to query GPU memory: {e}")
    
    return result


def get_configured_device(config: Dict[str, Any]) -> torch.device:
    """
    Get the configured device object.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PyTorch device object
    """
    device_config = config.get('device', {})
    if 'current_device' in device_config:
        return device_config['current_device']
    else:
        return get_device(config)


def get_configured_dtype(config: Dict[str, Any]) -> torch.dtype:
    """
    Get the configured dtype object.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PyTorch dtype object
    """
    features_config = config.get('features', {})
    if 'torch_dtype' in features_config:
        return features_config['torch_dtype']
    else:
        return get_dtype(config)


def initialize_system(config: Dict[str, Any]) -> None:
    """
    Initialize system settings based on configuration.
    
    Args:
        config: Configuration dictionary
    """
    # Set random seed for reproducibility
    seed = config.get('general', {}).get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Configure CUDA settings
    device_config = config.get('device', {})
    if torch.cuda.is_available() and device_config.get('use_cuda', True):
        if device_config.get('deterministic', False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = device_config.get('benchmark', True)
        
        if hasattr(torch.cuda, 'amp') and config.get('numerical_stability', {}).get('mixed_precision', False):
            logger.info("Automatic mixed precision is available")
    
    # Create necessary directories
    general_config = config.get('general', {})
    os.makedirs(general_config.get('cache_dir', './cache'), exist_ok=True)
    os.makedirs(general_config.get('output_dir', './output'), exist_ok=True)


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path where to save the configuration
        
    Raises:
        ConfigError: If the configuration cannot be saved
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Remove any torch-specific objects that can't be serialized
        clean_config = clean_config_for_saving(config)
        
        with open(output_path, 'w') as f:
            yaml.dump(clean_config, f, default_flow_style=False, sort_keys=False)
            
        logger.info(f"Configuration saved to {output_path}")
    except Exception as e:
        raise ConfigError(f"Failed to save configuration to {output_path}: {str(e)}")


def clean_config_for_saving(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove non-serializable objects from configuration for saving.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Clean configuration dictionary
    """
    result = {}
    
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = clean_config_for_saving(value)
        elif isinstance(value, (str, int, float, bool, list, tuple)):
            result[key] = value
        elif value is None:
            result[key] = None
        elif isinstance(value, torch.device):
            result[key] = str(value)
        elif isinstance(value, torch.dtype):
            if value == torch.float32:
                result[key] = "float32"
            elif value == torch.float64:
                result[key] = "float64"
            else:
                result[key] = str(value)
        else:
            # Skip non-serializable objects
            logger.debug(f"Skipping non-serializable config value for key {key}: {type(value)}")
    
    return result


def get_config_summary(config: Dict[str, Any]) -> str:
    """
    Get a human-readable summary of the configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Summary string
    """
    lines = []
    lines.append("=== TIMO Configuration Summary ===")
    
    # General information
    general = config.get('general', {})
    lines.append(f"- Cache directory: {general.get('cache_dir', './cache')}")
    lines.append(f"- Output directory: {general.get('output_dir', './output')}")
    lines.append(f"- Random seed: {general.get('seed', 42)}")
    
    # Dataset information
    dataset = config.get('dataset', {})
    lines.append(f"- Dataset: {dataset.get('name', 'unknown')}")
    lines.append(f"- Shots: {dataset.get('shots', 16)}")
    
    # Model information
    model = config.get('model', {})
    lines.append(f"- Backbone: {model.get('backbone', 'ViT-B/32')}")
    lines.append(f"- Feature dimension: {model.get('feature_dim', 512)}")
    
    # Module status
    igt = config.get('igt', {})
    tgi = config.get('tgi', {})
    blending = config.get('blending', {})
    lines.append(f"- IGT module: {'Enabled' if igt.get('enabled', True) else 'Disabled'}")
    lines.append(f"- TGI module: {'Enabled' if tgi.get('enabled', True) else 'Disabled'}")
    lines.append(f"- Feature blending: {'Enabled' if blending.get('enabled', True) else 'Disabled'}")
    
    # Key parameters
    lines.append(f"- IGT gamma: {igt.get('gamma', 50.0)}")
    lines.append(f"- TGI temperature: {tgi.get('temperature', 0.1)}")
    lines.append(f"- Blend weight: {blending.get('blend_weight', 0.5)}")
    
    # Hardware and precision
    device = config.get('device', {})
    features = config.get('features', {})
    lines.append(f"- Device: {device.get('current_device_str', get_device(config))}")
    lines.append(f"- Data type: {features.get('dtype', 'float32')}")
    
    # Memory settings
    memory = config.get('memory', {})
    lines.append(f"- Max batch size: {memory.get('max_batch_size', 1024)}")
    lines.append(f"- Chunk size: {memory.get('chunk_size', 256)}")
    
    return "\n".join(lines)
