def filter_user_query(user_query: dict, selected_keys: list = ["Mnemonic", "Description"]) -> dict:
    """
    Reduce the distractions.
    """
    user_query_filtered = {}
    for k in user_query:
        if k in selected_keys:
            user_query_filtered[k] = user_query[k]
    return user_query_filtered


def remove_keys_from_dict(input_dict: dict, keys_to_remove: list) -> dict:
    filtered_dict = {}
    for k in input_dict:
        if k not in keys_to_remove:
            filtered_dict[k] = input_dict[k]
    return filtered_dict


def fix_project_id_list(raw_list: list):
    project_id_list = []
    for raw_id in raw_list:
        # normalized_path = path.replace("\\", "/")
        # project_id = os.path.basename(normalized_path)
        project_id = raw_id.replace("$", "_")
        project_id = project_id.replace("\\", "+")
        project_id = project_id.replace("/", "+")
        project_id = project_id.replace(" ", "_")
        project_id_list.append(project_id)
    return project_id_list


"""
user_query_minimal_keys = ["Namespace", "Mnemonic", "Description", "Unit", "DataType"]
A tyipical user query is:
user_query = {
    "Namespace": "",
    "Mnemonic": "",
    "Description": "",
    "Unit": "",
    "DataType": "",
    "Others": "",
}
"""

metadata_profile_dict = {
    "Volve open data": {
        "Namespace": "http://ddhub.demo/zzz/<project_id>",
        "selected_keys": [
            "mnemonic",
            "unit",
            "curveDescription",
            "dataSource",
            "typeLogData",
        ],
        "key_mapping": {
            "Mnemonic": "mnemonic",
            "Description": "curveDescription",
            "DataType": "typeLogData",
            "Unit": "unit",
        },
    },
    "mnemonic_rich_scraped": {
        "Namespace": "http://ddhub.demo/zzz/",
        "selected_keys": [
            "LongMnemonic",
            "ShortMnemonic",
            "shortDescription",
            "longDescription",
            "DataType",
            "DataLength",
            "MetricUnits",
            "FPSUnits",
        ],
        "key_mapping": {
            "Mnemonic": "LongMnemonic",
            "Description": "longDescription",
            "DataType": "DataType",
            "Unit": "MetricUnits",
        },
    },
}
