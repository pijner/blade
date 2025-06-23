import yaml


def load_yaml(file_path):
    """
    Load a YAML file and return its content.

    :param file_path: Path to the YAML file.
    :return: Content of the YAML file as a dictionary.
    """
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

    return {}


def dump_yaml(data, file_path):
    """
    Dump data to a YAML file.

    :param data: Data to be written to the YAML file.
    :param file_path: Path where the YAML file will be saved.
    """
    with open(file_path, "w") as file:
        yaml.safe_dump(data, file, default_flow_style=False)
