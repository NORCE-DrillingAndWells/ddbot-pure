import os, json, typing, yaml
import preprocessor as pp
import rag_task_manager as rag
import chat_restapi as llm
import sparql_connector as sc
from difflib import get_close_matches
import logging, log_config
import utils as utils

logger = logging.getLogger(__name__)


def preprocess(s):
    s_after = s.replace("[", "").replace("]", "")
    s_after = s_after.replace(" ", "").replace("_", "").replace("-", "")
    s_after = s_after.lower()
    return s_after


def repair_terminology(targetStr: str, terminology_list: list) -> str:
    preprocessed_map = {preprocess(s): s for s in terminology_list}

    processed_target = preprocess(targetStr)

    if targetStr == "none" or targetStr == "None":
        return targetStr
    if processed_target in preprocessed_map:
        return preprocessed_map[processed_target]

    matches = get_close_matches(processed_target, preprocessed_map.keys(), n=1, cutoff=0.8)
    if matches:
        return preprocessed_map[matches[0]]
    else:
        print("no match found: " + targetStr)
        return "None"
    # raise ValueError("no match found: " + targetStr)


def repair_terminology_list(targetStr_list: list, terminology_list: list) -> list:
    repaired_list = []
    for targetStr in targetStr_list:
        repaired = repair_terminology(targetStr, terminology_list)
        repaired_list.append(repaired)
    return repaired_list


def supplement_candidates_extraContent(
    candidate_list: list, fullList_extraContent: dict, kickout_keys: list = None
) -> dict:
    candidates_extraContent = {}
    for c in candidate_list:
        if c == "none" or c == "None":
            continue
        extraContent = fullList_extraContent[c].copy()
        if kickout_keys is not None:
            for key in kickout_keys:
                if key in extraContent.keys():
                    extraContent.pop(key)
        candidates_extraContent.update({c: extraContent})
    return candidates_extraContent


def narrow_selection_range(
    task_type: str,
    prompt_templates: dict,
    metadata: dict = None,
    selection_range: list = None,
    interpretation: str = None,
    other_pairs: dict = None,
    model: str = llm.DEFAULT_MODEL,
) -> list:
    prompt_template_entry = prompt_templates[task_type]
    prompt_template = prompt_template_entry[prompt_template_entry["default"]]
    complementary_knowledge = prompt_templates["complementary_knowledge"]["v1"]

    user_query_filtered = utils.filter_user_query(metadata, ["Mnemonic", "Description", "Unit", "DataType"])
    kvPairs = {
        "<user_query>": user_query_filtered,
        "<complementary_knowledge>": complementary_knowledge,
    }
    if interpretation is not None:
        kvPairs.update({"<interpretation>": interpretation})
    if selection_range is not None:
        kvPairs.update({"<selection_range>": selection_range})
    if other_pairs is not None:
        kvPairs.update(other_pairs)
    prompt = rag.assemble_prompt(prompt_template, kvPairs)
    result = rag.run_rag_task_single(prompt, model)

    result_list = result["content"].split(",")
    result_content = [s.strip() for s in result_list]
    # result_content = result["content"]
    return result_content, prompt


def initialize_task_batch(preprocessed_metadata_batch: dict) -> dict:
    """
    preprocessed_metadata contains
        {"Namespace": "project URI",
        "Mnemonic": "uid",
        "Unit": "unit_name",...}
    """

    task_template = {
        "Namespace": "",
        "DrillingDataPoint_name": "",
        "PrototypeData_class": "",
        "PrototypeData_class_candidates": [],
        "MeasurableQuantity_class": "",
        "Quantity_class": "",
        "Unit_name": "",
        "Unit_class": "",
        "Unit_class_candidates": [],
        "Provider_name": "",
        "ToIdentify": "",
    }
    task_batch = {}
    for pp_metadata in preprocessed_metadata_batch.values():
        task = task_template.copy()
        DrillingDataPoint_name = pp_metadata["Mnemonic"]
        task["Namespace"] = pp_metadata["Namespace"]
        task["DrillingDataPoint_name"] = DrillingDataPoint_name
        task["Unit_name"] = pp_metadata["Unit"]
        task_batch.update({DrillingDataPoint_name: task})
    return task_batch


def generate_fullList_extraContent(KB_location: str) -> list:
    prototypeData_fullList_extraContent = sc.generate_PrototypeData_fullList_extraContent(KB_location)
    quantity_fullList_extraContent = sc.generate_Quantity_fullList_extraContent(KB_location)
    unit_fullList_extraContent = sc.generate_Unit_fullList_extraContent(KB_location)
    return (
        quantity_fullList_extraContent,
        unit_fullList_extraContent,
        prototypeData_fullList_extraContent,
    )


def run_cot_task_single(
    user_query: dict,
    prompt_templates: dict,
    quantity_fullList_extraContent: dict,
    unit_fullList_extraContent: dict,
    prototypeData_fullList_extraContent: dict,
    interpretation: str = None,
    models_high_low: list = [llm.DEFAULT_MODEL, llm.DEFAULT_MODEL],
):
    """
    Preparation: the full list with extra content as cache: quantity, prototypeData, unit.

    1. Interpret the metadata.
    2. Recognize the quantity.
        1. Retrieve all the quantities from KB.
        2. Use LLM to preselect top X candidates from the full list.
        3. Adjust the candidate list and supplement extra content.
        4. Use LLM to finally recognize quantity.
    3. Recognize the unit.
        1. Retrieve the units related to top X quantity candidates from KB.
        2. Use LLM to preselect top X candidates from the related list.
        3. Adjust the candidate list and supplement extra content.
        4. Use LLM to finally recognize unit.
    4. Recognize the prototypeData.
        1. Retrieve the prototypeData related to top X quantity candidates from KB.
        2. Use LLM to preselect top X candidates from the related list.
        3. Adjust the candidate list and supplement extra content.
        4. Use LLM to finally recognize prototypeData.
    5. Retrieve the MeasurableQuantity class from KB.
    """
    # quantity_fullList = list(quantity_fullList_extraContent.keys())
    # unit_fullList = list(unit_fullList_extraContent.keys())
    # prototypeData_fullList = list(prototypeData_fullList_extraContent.keys())

    # 1. Interpret the metadata.
    if interpretation is None:
        interpretation, prompt_interpret_mnemonic = interpret_mnemonic(
            user_query, prompt_templates, models_high_low[1]
        )
    # print("--------\n", "interpretation: ", interpretation, "\n")
    logger.info(f"interpretation: {interpretation}")

    # 2, 3, 4, 5. Recognize the quantity, unit, prototypeData.
    [recognized_class, candidates, prompts] = recoginize_metadata(
        user_query,
        interpretation,
        prompt_templates,
        quantity_fullList_extraContent,
        unit_fullList_extraContent,
        prototypeData_fullList_extraContent,
        models_high_low,
    )

    prompts.update({"prompt_interpret_mnemonic": prompt_interpret_mnemonic})

    toReturn = [interpretation, recognized_class, candidates, prompts]
    return toReturn


def recoginize_metadata(
    user_query: dict,
    interpretation: str,
    prompt_templates: dict,
    quantity_fullList_extraContent: dict,
    unit_fullList_extraContent: dict,
    prototypeData_fullList_extraContent: dict,
    models_high_low: list = [llm.DEFAULT_MODEL, llm.DEFAULT_MODEL],
) -> list:
    # Results to return
    # recognized_class, candidates, prompts = {}, {}, {}

    quantity_fullList = list(quantity_fullList_extraContent.keys())
    # unit_fullList = list(unit_fullList_extraContent.keys())
    # prototypeData_fullList = list(prototypeData_fullList_extraContent.keys())

    # 2.1. Retrieve all the quantities from KB.
    # 2.2. Use LLM to preselect top X candidates from the full list.
    quantity_candidates, prompt_preselect_quantity = preselect_quantity(
        user_query, quantity_fullList, prompt_templates, interpretation, models_high_low[1]
    )
    # print("quantity_candidates: ", quantity_candidates)
    logger.info(f"quantity_candidates: {quantity_candidates}")

    # 2.3. Adjust the candidate list and supplement extra content.
    if not quantity_candidates:
        quantity_candidates = list(quantity_fullList_extraContent.keys())
    quantity_candidates_extraContent = supplement_quantity_candidates(
        quantity_candidates, [], quantity_fullList_extraContent
    )
    # 2.4. Use LLM to finally recognize quantity.
    quantity_class, prompt_recognize_quantity = recognize_quantity(
        user_query, quantity_candidates_extraContent, prompt_templates, interpretation, models_high_low[1]
    )
    # print("quantity_class: ", quantity_class, "\n---")
    logger.info(f"quantity_class: {quantity_class}")

    # 3.1. Retrieve the units related to top X quantity candidates from KB.
    quantity_candidates = list(quantity_candidates_extraContent.keys())
    unit_candidates_related = retrieve_unit_relatedTo_Quantiy(quantity_candidates)
    # print("unit_candidates_related", unit_candidates_related)
    if not unit_candidates_related:
        unit_candidates_related = list(unit_fullList_extraContent.keys())

    # 3.2. Use LLM to preselect top X candidates from the related list.
    unit_candidates_preselected, prompt_preselect_unit = preselect_unit(
        user_query, unit_candidates_related, prompt_templates, interpretation, models_high_low[1]
    )

    # 3.3. Adjust the candidate list and supplement extra content.
    if not unit_candidates_preselected:
        unit_candidates_preselected = list(unit_fullList_extraContent.keys())
    unit_candidates_extraContent = supplement_unit_candidates(
        unit_candidates_preselected, [], unit_fullList_extraContent
    )
    # print("unit_candidates: ", list(unit_candidates_extraContent.keys()))
    logger.info(f"unit_candidates: {list(unit_candidates_extraContent.keys())}")
    # 3.4. Use LLM to finally recognize unit.
    unit_class, prompt_recognize_unit = recognize_unit(
        user_query, unit_candidates_extraContent, prompt_templates, interpretation, models_high_low[1]
    )
    # print("unit_class: ", unit_class, "\n---")
    logger.info(f"unit_class: {unit_class}")

    # 4.1. Retrieve the prototypeData related to top X quantity candidates from KB.
    prototypeData_candidates_related = retrieve_prototypeData_relatedTo_Quantiy(quantity_candidates)
    if not prototypeData_candidates_related:
        prototypeData_candidates_related = list(prototypeData_fullList_extraContent.keys())
    # 4.2. Use LLM to preselect top X candidates from the related list.
    prototypeData_candidates_preselected, prompt_preselect_prototypeData = preselect_prototypeData(
        user_query, prototypeData_candidates_related, prompt_templates, interpretation, models_high_low[0]
    )

    # 4.3. Adjust the candidate list and supplement extra content.
    supplementary_candidates_extraContent = rag.retrieve_relatedContext(
        user_query, prototypeData_fullList_extraContent
    )
    supplementary_candidates = []
    for c in supplementary_candidates_extraContent:
        p = c["ddhub:PrototypeData"]
        supplementary_candidates.append(p)
    prototypeData_candidates_extraContent = supplement_prototypeData_candidates(
        prototypeData_candidates_preselected,
        supplementary_candidates,
        prototypeData_fullList_extraContent,
    )
    # print("prototypeData_candidates: ", list(prototypeData_candidates_extraContent.keys()))
    logger.info(f"prototypeData_candidates: {list(prototypeData_candidates_extraContent.keys())}")

    # 4.4. Use LLM to finally recognize prototypeData.
    prototypeData_class, prompt_recognize_prototypeData = recognize_prototypeData(
        user_query, prototypeData_candidates_extraContent, prompt_templates, interpretation, models_high_low[0]
    )
    # print("prototypeData_class: ", prototypeData_class, "\n---")
    logger.info(f"prototypeData_class: {prototypeData_class}")

    # 5. Retrieve the MeasurableQuantity class from KB.
    if prototypeData_class != "None":
        MQuantity_class = prototypeData_fullList_extraContent[prototypeData_class]["ddhub:IsOfMeasurableQuantity"]
    else:
        MQuantity_class = "None"

    # package the results to return
    recognized_class = {
        "Quantity_class": quantity_class,
        "Unit_class": unit_class,
        "PrototypeData_class": prototypeData_class,
        "MeasurableQuantity_class": MQuantity_class,
    }
    candidates = {
        "Quantity_candidates": quantity_candidates,
        "Unit_candidates": list(unit_candidates_extraContent.keys()),
        "PrototypeData_candidates": list(prototypeData_candidates_extraContent.keys()),
    }
    prompts = {
        "prompt_preselect_quantity": prompt_preselect_quantity,
        "prompt_recognize_quantity": prompt_recognize_quantity,
        "prompt_preselect_unit": prompt_preselect_unit,
        "prompt_recognize_unit": prompt_recognize_unit,
        "prompt_preselect_prototypeData": prompt_preselect_prototypeData,
        "prompt_recognize_prototypeData": prompt_recognize_prototypeData,
    }

    toReturn = [recognized_class, candidates, prompts]
    return toReturn


def semanticize_raw_metadata(
    project_id: str,
    output_path: str,
    raw_metadata_file_path: str,
    raw_metadata_file_reader: typing.Callable[[str], dict],
    metadata_profile: dict,
    prompt_templates_path: str,
    user_interaction: bool = False,
    models_high_low: list = [llm.DEFAULT_MODEL, llm.DEFAULT_MODEL],
):
    project_folder_path = output_path + project_id
    if not os.path.exists(project_folder_path):
        os.makedirs(project_folder_path)

    log_config.change_logfile_path(project_folder_path + "/record.log")

    if prompt_templates_path.endswith(".yaml"):
        with open(prompt_templates_path, "r") as yaml_file:
            prompt_templates = yaml.safe_load(yaml_file)
    elif prompt_templates_path.endswith(".json"):
        with open(prompt_templates_path, "r") as json_file:
            prompt_templates = json.load(json_file)

    # Preprocess the raw metadata
    preprocessed_metadata_batch = pp.preprocess_metadata_batch(
        raw_metadata_file_path, raw_metadata_file_reader, metadata_profile
    )
    with open(project_folder_path + "/preprocessed_metadata_batch.json", "w") as json_file:
        json.dump(preprocessed_metadata_batch, json_file, indent=4)

    # Initialize the task_batch
    tb = initialize_task_batch(preprocessed_metadata_batch)
    with open(project_folder_path + "/task_batch.json", "w") as json_file:
        json.dump(tb, json_file, indent=4)
    # tb_detail = {}

    logger.info("Start retrieving knowledge.")
    quantity_fullList_extraContent = sc.generate_Quantity_fullList_extraContent(sc.KB_ttl_path)
    unit_fullList_extraContent = sc.generate_Unit_fullList_extraContent(sc.KB_ttl_path)
    prototypeData_fullList_extraContent = sc.generate_PrototypeData_fullList_extraContent(sc.KB_ttl_path)

    with open(project_folder_path + "/quantity_fullList_extraContent.json", "w") as json_file:
        json.dump(quantity_fullList_extraContent, json_file, indent=4)
    with open(project_folder_path + "/unit_fullList_extraContent.json", "w") as json_file:
        json.dump(unit_fullList_extraContent, json_file, indent=4)
    with open(project_folder_path + "/prototypeData_fullList_extraContent.json", "w") as json_file:
        json.dump(prototypeData_fullList_extraContent, json_file, indent=4)
    logger.info("Finish retrieving knowledge.")

    # 1. Interpret the metadata.
    for key, value in preprocessed_metadata_batch.items():
        Interpretation_LLM, prompt_interpret_mnemonic = interpret_mnemonic(value, prompt_templates, models_high_low[1])
        tb[key]["Interpretation_LLM"] = Interpretation_LLM
        tb[key]["Interpretation_user"] = "LLM: " + Interpretation_LLM
        # tb_detail[key] = tb[key].copy()
        # tb_detail[key]["prompt_interpret_mnemonic"] = prompt_interpret_mnemonic

    with open(project_folder_path + "/task_batch.json", "w") as json_file:
        json.dump(tb, json_file, indent=4)
    # with open(project_folder_path + "/task_batch_detail.json", "w") as json_file:
    #     json.dump(tb_detail, json_file, indent=4)

    # Wait for user interaction
    if user_interaction == True:
        user_input = ""
        while user_input != "Done":
            print(
                "Modify each interpretation (Interpretation_user) in File:",
                project_folder_path + "/task_batch.json.\n",
                "If no modification, the interpretation from LLM will be used.",
            )
            user_input = input("When you finish, please input 'Done' to continue...")

    with open(project_folder_path + "/task_batch.json", "r") as json_file:
        tb = json.load(json_file)
    # with open(project_folder_path + "/task_batch_detail.json", "r") as json_file:
    #     tb_detail = json.load(json_file)

    # 2, 3, 4, 5. Recognize the quantity, unit, prototypeData.
    for key, value in preprocessed_metadata_batch.items():
        interpretation_user = tb[key]["Interpretation_user"]
        user_query = utils.filter_user_query(value, ["Mnemonic", "Description", "DataType", "Unit"])

        [recognized_class, candidates, prompts] = recoginize_metadata(
            user_query,
            interpretation_user,
            prompt_templates,
            quantity_fullList_extraContent,
            unit_fullList_extraContent,
            prototypeData_fullList_extraContent,
            models_high_low,
        )
        # prompt_interpret_mnemonic = tb_detail[key]["prompt_interpret_mnemonic"]
        # prompts.update({"prompt_interpret_mnemonic": prompt_interpret_mnemonic})

        datapoint = {
            "Namespace": "http://ddhub.demo/zzz/" + project_id,
            "DrillingDataPoint_name": key,
            "PrototypeData_class": recognized_class["PrototypeData_class"],
            "PrototypeData_class_candidates": candidates["PrototypeData_candidates"],
            "MeasurableQuantity_class": recognized_class["MeasurableQuantity_class"],
            "Quantity_class": recognized_class["Quantity_class"],
            "Quantity_class_candidates": candidates["Quantity_candidates"],
            "Unit_name": preprocessed_metadata_batch[key]["Unit"],
            "Unit_class": recognized_class["Unit_class"],
            "Unit_class_candidates": candidates["Unit_candidates"],
            "Provider_name": "",
            "Interpretation_user": interpretation_user,
        }

        # datapoint_detail = {
        #     "Namespace": "http://ddhub.demo/zzz/" + project_id,
        #     "DrillingDataPoint_name": key,
        #     "PrototypeData_class": recognized_class["PrototypeData_class"],
        #     "PrototypeData_class_candidates": candidates["PrototypeData_candidates"],
        #     "MeasurableQuantity_class": recognized_class["MeasurableQuantity_class"],
        #     "Quantity_class": recognized_class["Quantity_class"],
        #     "Quantity_class_candidates": candidates["Quantity_candidates"],
        #     "Unit_name": preprocessed_metadata_batch[key]["Unit"],
        #     "Unit_class": recognized_class["Unit_class"],
        #     "Unit_class_candidates": candidates["Unit_candidates"],
        #     "Provider_name": "",
        #     "Interpretation_user": interpretation_user,
        #     "prompt_interpret_mnemonic": prompts["prompt_interpret_mnemonic"],
        #     "prompt_preselect_quantity": prompts["prompt_preselect_quantity"],
        #     "prompt_recognize_quantity": prompts["prompt_recognize_quantity"],
        #     "prompt_preselect_unit": prompts["prompt_preselect_unit"],
        #     "prompt_recognize_unit": prompts["prompt_recognize_unit"],
        #     "prompt_preselect_prototypeData": prompts["prompt_preselect_prototypeData"],
        #     "prompt_recognize_prototypeData": prompts["prompt_recognize_prototypeData"],
        # }

        tb.update({key: datapoint})
        with open(project_folder_path + "/task_batch.json", "w") as json_file:
            json.dump(tb, json_file, indent=4)
        # tb_detail.update({key: datapoint_detail})
        # with open(project_folder_path + "/task_batch_detail.json", "w") as json_file:
        #     json.dump(tb_detail, json_file, indent=4)


def interpret_mnemonic(
    metadata: dict,
    prompt_templates: dict,
    model: str = llm.DEFAULT_MODEL,
) -> list:
    prompt_template_entry = prompt_templates["Interpret_mnemonic"]
    prompt_template = prompt_template_entry[prompt_template_entry["default"]]
    complementary_knowledge = prompt_templates["complementary_knowledge"]["v1"]

    user_query_filtered = utils.filter_user_query(metadata, ["Mnemonic", "Description", "Unit", "DataType"])
    kvPairs = {
        "<user_query>": user_query_filtered,
        "<complementary_knowledge>": complementary_knowledge,
    }

    prompt = rag.assemble_prompt(prompt_template, kvPairs)
    result = rag.run_rag_task_single(prompt, model)
    interpretation = str(result["content"])
    # print(interpretation)
    logger.info(f"Interpret_mnemonic prompt:\n{prompt}")
    logger.info(f"Interpret_mnemonic result: {interpretation}")

    # interpretation, prompt = narrow_selection_range("Interpret_mnemonic", prompt_templates, metadata)
    return interpretation, prompt


def preselect_quantity(
    metadata: dict,
    quantity_fullList: list,
    prompt_templates: dict,
    interpretation: str,
    model: str = llm.DEFAULT_MODEL,
) -> list:
    task_type = "Preselect_quantity"
    result_content, prompt = narrow_selection_range(
        task_type,
        prompt_templates,
        metadata,
        quantity_fullList,
        interpretation,
        model=model,
    )
    # print("prompt: ", prompt)
    # print("result_content: ", result_content, "\n")
    logger.info(f"{task_type} prompt:\n{prompt}")
    logger.info(f"{task_type} result: {result_content}")
    preselected_quantity_list = repair_terminology_list(result_content, quantity_fullList)
    return preselected_quantity_list, prompt


def supplement_quantity_candidates(
    original_candidates: list,
    supplementary_candidates: list,
    fullList_extraContent: dict,
) -> dict:
    candidates = original_candidates + supplementary_candidates
    # include QuantityHasUnit, as this is for the preselected ones.
    candidates_extraContent = supplement_candidates_extraContent(candidates, fullList_extraContent)
    return candidates_extraContent


def recognize_quantity(
    metadata: dict,
    candidates_extraContent: dict,
    prompt_templates: dict,
    interpretation: str,
    model: str = llm.DEFAULT_MODEL,
) -> list:
    task_type = "Recognize_quantity"
    result_content, prompt = narrow_selection_range(
        task_type,
        prompt_templates,
        metadata,
        list(candidates_extraContent.values()),
        interpretation,
        model=model,
    )
    # print("prompt: ", prompt)
    # print("result_content: ", result_content, "\n")
    logger.info(f"{task_type} prompt:\n{prompt}")
    logger.info(f"{task_type} result: {result_content}")
    repaired_term = repair_terminology_list(result_content, candidates_extraContent.keys())
    return repaired_term[0], prompt


def retrieve_unit_relatedTo_Quantiy(quantity_candidates: list) -> list:
    unit_list_related = []
    for qc in quantity_candidates:
        unit_list_tmp = sc.query_Unit_relatedTo_Quantiy(sc.KB_ttl_path, qc)
        unit_list_related.extend(unit_list_tmp)
    return unit_list_related


def preselect_unit(
    metadata: dict,
    unit_fullList: list,
    prompt_templates: dict,
    interpretation: str,
    model: str = llm.DEFAULT_MODEL,
) -> list:
    task_type = "Preselect_unit"
    result_content, prompt = narrow_selection_range(
        task_type,
        prompt_templates,
        metadata,
        unit_fullList,
        interpretation,
        model=model,
    )
    # print("prompt: ", prompt)
    # print("result_content: ", result_content, "\n")
    logger.info(f"{task_type} prompt:\n{prompt}")
    logger.info(f"{task_type} result: {result_content}")
    preselected_quantity_list = repair_terminology_list(result_content, unit_fullList)
    return preselected_quantity_list, prompt


def supplement_unit_candidates(
    original_candidates: list,
    supplementary_candidates: list,
    fullList_extraContent: dict,
) -> list:
    candidates = original_candidates + supplementary_candidates
    candidates_extraContent = supplement_candidates_extraContent(candidates, fullList_extraContent)
    return candidates_extraContent


def recognize_unit(
    metadata: dict,
    candidates_extraContent: dict,
    prompt_templates: dict,
    interpretation: str,
    model: str = llm.DEFAULT_MODEL,
) -> list:
    task_type = "Recognize_unit"
    result_content, prompt = narrow_selection_range(
        task_type,
        prompt_templates,
        metadata,
        list(candidates_extraContent.values()),
        interpretation,
        model=model,
    )
    # print("prompt: ", prompt)
    # print("result_content: ", result_content, "\n")
    logger.info(f"{task_type} prompt:\n{prompt}")
    logger.info(f"{task_type} result: {result_content}")
    repaired_term = repair_terminology_list(result_content, list(candidates_extraContent.keys()))
    return repaired_term[0], prompt


def retrieve_prototypeData_relatedTo_Quantiy(quantity_candidates: list) -> list:
    prototypeData_list_related = []
    for qc in quantity_candidates:
        prototypeData_list_tmp = sc.query_PrototypeData_relatedTo_Quantiy(sc.KB_ttl_path, qc)
        prototypeData_list_related.extend(prototypeData_list_tmp)
    return prototypeData_list_related


def preselect_prototypeData(
    metadata: dict,
    prototypeData_fullList: list,
    prompt_templates: dict,
    interpretation: str,
    model: str = llm.DEFAULT_MODEL,
) -> list:
    task_type = "Preselect_prototypeData"
    result_content, prompt = narrow_selection_range(
        task_type,
        prompt_templates,
        metadata,
        prototypeData_fullList,
        interpretation,
        model=model,
    )
    # print("prompt: ", prompt)
    # print("result_content: ", result_content, "\n")
    logger.info(f"{task_type} prompt:\n{prompt}")
    logger.info(f"{task_type} result: {result_content}")
    preselect_prototypeData_list = repair_terminology_list(result_content, prototypeData_fullList)
    return preselect_prototypeData_list, prompt


def supplement_prototypeData_candidates(
    original_candidates: list,
    supplementary_candidates: list,
    fullList_extraContent: dict,
) -> list:
    candidates = original_candidates + supplementary_candidates
    # IsOfMeasurableQuantity should not be sent to LLM in any cases.
    candidates_extraContent = supplement_candidates_extraContent(
        candidates, fullList_extraContent, ["ddhub:IsOfMeasurableQuantity"]
    )
    return candidates_extraContent


def recognize_prototypeData(
    metadata: dict,
    candidates_extraContent: dict,
    prompt_templates: dict,
    interpretation: str,
    model: str = llm.DEFAULT_MODEL,
) -> list:
    task_type = "Recognize_prototypeData"
    result_content, prompt = narrow_selection_range(
        task_type,
        prompt_templates,
        metadata,
        list(candidates_extraContent.values()),
        interpretation,
        model=model,
    )
    # print("prompt: ", prompt)
    # print("result_content: ", result_content, "\n")
    logger.info(f"{task_type} prompt:\n{prompt}")
    logger.info(f"{task_type} result: {result_content}")
    repaired_term = repair_terminology_list(result_content, list(candidates_extraContent.keys()))
    return repaired_term[0], prompt


def test():
    user_query = {
        "Mnemonic": "HKLO",
        "Description": "zzz:undefined",
        "DataType": "double",
        "Unit": "kkgf",
    }
    currentFolder = os.path.dirname(os.path.realpath(__file__))
    prompt_templates_path = currentFolder + "/tasks/prompt_templates.json"
    with open(prompt_templates_path, "r") as json_file:
        prompt_templates = json.load(json_file)

    quantity_fullList_extraContent = sc.generate_Quantity_fullList_extraContent(sc.KB_ttl_path)
    unit_fullList_extraContent = sc.generate_Unit_fullList_extraContent(sc.KB_ttl_path)
    prototypeData_fullList_extraContent = sc.generate_PrototypeData_fullList_extraContent(sc.KB_ttl_path)

    run_cot_task_single(
        user_query,
        prompt_templates,
        quantity_fullList_extraContent,
        unit_fullList_extraContent,
        prototypeData_fullList_extraContent,
    )


if __name__ == "__main__":
    test()
