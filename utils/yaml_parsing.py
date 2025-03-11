import os
import yaml
from dataclasses import make_dataclass, dataclass
from typing import Union

def get_yaml_params(filename: str, config_dir: str = 'configs', as_dataclass: bool = False) -> Union[dict, dataclass]:
    """
    Load YAML parameters from a file and optionally create a dataclass from those parameters.

    Args:
        filename (str): The name of the YAML file without extension.
        config_dir (str, optional): The directory containing the YAML file. Defaults to 'configs'.
        as_dataclass (bool, optional): Whether to create a dataclass. Defaults to False.

    Returns:
        Union[dict, dataclass]: Loaded YAML data or a dataclass instance if as_dataclass is True.
    
    Raises:
        FileNotFoundError: If the specified YAML file is not found in the specified directory.
    """
    file_path = os.path.join(config_dir, f'{filename}.yml')

    # Ensure the directory exists before attempting to open the file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{filename}.yml' not found in '{config_dir}' directory")

    # Load YAML file
    with open(file_path) as f:
        data = yaml.safe_load(f)

    # Optionally create a dataclass
    if as_dataclass:
        YAMLParams = make_dataclass('YAMLParams', data.keys())
        data = YAMLParams(**data)

    return data
