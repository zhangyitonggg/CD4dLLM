import os

def get_jsonschema_cfg(instance_id: str="none") -> str:
    """
    Returns the CFG string for the JSON Schema language.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))  
    cfg_file_path = os.path.join(base_dir, "json", f"{instance_id}.lark")
    with open(cfg_file_path, "r") as f:
        cfg_str = f.read()
    return cfg_str



def get_cfg(short_name: str, instance_id: str="none") -> str:
    """
    Returns the CFG string for the given language short name.
    """
    if short_name == "jsonschema":
        return get_jsonschema_cfg(instance_id)

    cfg_file_map = {
        "cpp": "cpp.lark",
        "smiles": "smiles.lark",
    }

    if short_name not in cfg_file_map:
        raise ValueError(f"No CFG file found for language: {short_name}")

    cfg_file_path = os.path.join(
        os.path.dirname(__file__),
        cfg_file_map[short_name],
    )

    with open(cfg_file_path, "r") as f:
        cfg_str = f.read()

    return cfg_str

if __name__ == "__main__":
    # cpp_cfg = get_cfg("cpp")
    # print(cpp_cfg)

    jsonschema_cfg = get_cfg("jsonschema", "jsonschema_111")
    print(jsonschema_cfg)