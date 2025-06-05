# -*- encoding: utf-8 -*-
import json
import logging
import logging.config
import os
import re
from typing import Any, Optional, Tuple, List, Dict
import yaml


def setup_logging(
    output_directory,
    config_path='config/logging.yaml'
):
    """
    Load logging configuration from a YAML configuration file.
    """
  
    config = load_yaml_safe(config_path)

    log_dir = os.path.join(output_directory, "logs")
    os.makedirs(log_dir, exist_ok=True)
    config['handlers']['file']['filename'] = os.path.join(log_dir, f"tableeval.log")
    logging.config.dictConfig(config)


def load_yaml_safe(file_path):
    """
    Load configuration from a YAML file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file)
        if not config_data:
            raise ValueError(f"Config file {file_path} is empty or invalid.")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Config file {file_path} not found.") from e
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML file: {e}") from e
    return config_data


def load_api_configuration(model_name: str, config_file: str) -> Tuple[str, str, Optional[float], int]:
    """
    Load API configuration from a YAML file.
    Returns a tuple of (base_url, api_key, timeout, max_retries).
    """
    config_data = load_yaml_safe(config_file)
    api_config = config_data.get(model_name)
    if not api_config:
        raise ValueError(f"No configuration found for model '{model_name}'.")

    api_key = api_config.get('api_key')
    base_url = api_config.get('base_url')
    if not api_key:
        raise ValueError("Missing 'api_key' in API configuration.")
    if not model_name:
        raise ValueError("Missing 'model_name' in API configuration.")

    return api_config


def load_json(file_path):
    """
    Load JSON data from the specified file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.jsonl'):
                return [json.loads(line) for line in f]  # 逐行解析 JSONL
            elif file_path.endswith('.json'):
                return json.load(f)  # 解析标准 JSON
            else:
                raise ValueError("Only support .json and .jsonl")
    except (json.JSONDecodeError, OSError) as e:
        raise ValueError(f"JSON file load failed: {file_path}")

        

def save_json(data: Any, file_path: str, msg: Optional[str] = None) -> None:
    """
    Save data to a JSON file.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        if msg:
            logging.info(f"Save {msg} to {file_path}")
    except Exception as e:
        logging.error(f"Error saving file: {e}")


def load_txt(file_path: str) -> str:
    """
    Read and return the content of the specified file using UTF-8 encoding.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def sanitize_filename(filename: str) -> str:
    """
    Sanitize the filename by replacing illegal characters with underscores.
    """
    return re.sub(r'[\\/*?:"<>|]', "_", filename)


def generate_prediction_path(
    output_dir: str,
    model_name: str,
    reasoning_type: str,
    context_type: str,
    specific_tasks: Optional[List[str]] = None,
    specific_ids: bool = False
) -> str:
    """
    Generate the output file path based on provided parameters.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{sanitize_filename(model_name)}_{reasoning_type}_{context_type}")
    
    if specific_tasks:
        specific_tasks_str = "_".join(specific_tasks)
        output_path += "_" + specific_tasks_str
    
    if specific_ids:
        output_path += "_specific_ids"

    output_path += ".json"
    
    return output_path

