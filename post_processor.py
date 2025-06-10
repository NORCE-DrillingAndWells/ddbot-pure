import os, json, csv
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
    multilabel_confusion_matrix,
)
import pandas as pd
from utils import fix_project_id_list
from typing import List, Dict
from sklearn.preprocessing import MultiLabelBinarizer


def read_patch_file_csv(patch_file_path: str) -> list:
    patch_content = []
    with open(patch_file_path, newline="", encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)

        for row in csvreader:
            patch_content.append(row)
    return patch_content


def generate_table(project_folder_path: str):
    with open(project_folder_path + "/preprocessed_metadata_batch.json", "r", encoding="utf-8") as file:
        preprocessed_metadata_batch = json.load(file)
    with open(project_folder_path + "/task_batch.json", "r", encoding="utf-8") as file:
        task_batch = json.load(file)

    # Define the headers
    headers = [
        "Mnemonic",
        "Description in data log",
        "PrototypeData_class (ddhub:)",
        "PrototypeData_class_candidates (ddhub:)",
        "Unit",
        "Unit_class (ddhub:)",
        "Unit_class_candidates (ddhub:)",
        # "MeasurableQuantity_class (ddhub:)",
        "Quantity_class (ddhub:)",
        "DataType",
        "dataSource",
    ]

    # Add the rows of data
    data = []

    for Mnemonic in preprocessed_metadata_batch.keys():
        json_data_1 = preprocessed_metadata_batch[Mnemonic]
        json_data_2 = task_batch[Mnemonic]

        row = [
            json_data_1["Mnemonic"],
            json_data_1["Description"],
            json_data_2["PrototypeData_class"],
            str(json_data_2["PrototypeData_class_candidates"]),
            json_data_1["Unit"],
            json_data_2["Unit_class"],
            str(json_data_2["Unit_class_candidates"]),
            # json_data_2["MeasurableQuantity_class"],
            json_data_2["Quantity_class"],
            json_data_1["dataSource"] if "dataSource" in json_data_1 else ["none"],
            json_data_1["DataType"],
        ]

        data.append(row)

    # Create the DataFrame
    df = pd.DataFrame(data, columns=headers)

    # Save DataFrame to Excel
    df.to_excel(project_folder_path + "/table_for_validation.xlsx", index=False)


def generate_table_batch(project_package_folder_path: str, project_id_list: list):
    project_id_list_fixed = fix_project_id_list(project_id_list)
    for project_id in project_id_list_fixed:
        project_folder_path = project_package_folder_path + "/" + project_id
        generate_table(project_folder_path)


def excel_column_to_number(column_label):
    """
    column_label (str): Excel column_label:'A', 'B', ..., 'AA', 'AB', ...

    Returns:
    int: column_label
    """
    column_number = 0
    for char in column_label:
        column_number = column_number * 26 + (ord(char.upper()) - ord("A") + 1)
    return column_number - 1


def read_data_from_excel(
    file_path: str,
    sheet_index: int,
    col_index_pred: str,
    col_index_true: str,
    col_index_comp: str,
    delimiter: str = "|",
) -> dict:
    """
    Args：
    file_path (str): Excel file path.
    col_index_pred (str): column for y_pred.
    col_index_true (str): column for y_true.
    col_index_comp (str): column for complementary information.

    Return：
    dict: {"y_true": [], "y_pred": []}.
    """
    df = pd.read_excel(
        file_path,
        sheet_name=sheet_index,
        usecols=[
            excel_column_to_number(col_index_pred),
            excel_column_to_number(col_index_true),
            excel_column_to_number(col_index_comp),
        ],
        header=0,
        keep_default_na=False,
    )
    df.columns = ["y_pred", "y_true", "complementary"]
    y_true_list, y_pred_list = [], []
    for i in range(len(df)):
        y_pred = df["y_pred"][i] if df["y_pred"][i] != "" else "None"
        if df["y_true"][i] != "":
            y_true = df["y_true"][i]
        elif "No" in df["complementary"][i] or "no" in df["complementary"][i] or "?" in df["complementary"][i]:
            y_true = "None"
        else:
            raise ValueError("The true label is not provided.")
        y_true_list.append(y_true)
        y_pred_list.append(y_pred)
    result = {"y_true": y_true_list, "y_pred": y_pred_list}
    return result


def read_data_from_task_batch(file_path: str) -> dict:
    pass


def calc_confusion_matrix_and_metrics(data: dict):
    # Extract the true labels and the predicted labels from the input data
    y_true_raw = data["y_true"]
    y_pred = data["y_pred"]

    def parse_cell(cell: str) -> List[str]:
        # cell = str(cell).strip()
        # if cell == "" or cell.lower() == "none":
        #     return []
        cell_splitted = []
        if "|" in cell:
            cell_splitted = [t.strip() for t in cell.split("|") if t.strip()]
        else:
            cell_splitted = [cell]
        return cell_splitted

    y_true = []
    for i in range(len(y_true_raw)):
        label_true_list = parse_cell(y_true_raw[i])
        label_pred = y_pred[i]
        if label_pred in label_true_list:
            y_true.append(label_pred)
        else:
            y_true.append(y_true_raw[i])

    # Get the unique labels from the combined list of true and predicted labels
    labels = np.unique(y_true + y_pred)
    # labels = np.unique(y_true)

    # Calculate the confusion matrix using the unique labels to ensure proper order
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Convert the confusion matrix to a pandas DataFrame with row and column headers
    cm_df = pd.DataFrame(
        cm, index=[f"Actual: {label}" for label in labels], columns=[f"Predicted: {label}" for label in labels]
    )
    print("Confusion Matrix with Headers:\n", cm_df, "\n")

    # Calculate TP, FP, FN, TN for each class:
    # - True Positives (TP): correctly predicted for the class (diagonal of confusion matrix)
    # - False Positives (FP): predicted as the class but actually not that class
    # - False Negatives (FN): actual class but predicted as another class
    # - True Negatives (TN): all other correct predictions not related to this class
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    tn = np.sum(cm) - (tp + fp + fn)

    # Print the metrics for each class
    for i, label in enumerate(labels):
        print(f"Label '{label}':\tTP={tp[i]}, FP={fp[i]}, FN={fn[i]}, TN={tn[i]}")

    # Calculate micro-averaged metrics:
    # Micro-average aggregates the contributions of all classes to compute the average metric.
    micro_precision = precision_score(y_true, y_pred, average="micro", zero_division=0)
    micro_recall = recall_score(y_true, y_pred, average="micro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    print("\nMicro-averaged metrics:")
    print(f"Precision: {micro_precision:.4f}")
    print(f"Recall:    {micro_recall:.4f}")
    print(f"F1 Score:  {micro_f1:.4f}")

    # Calculate macro-averaged metrics:
    # Macro-average computes the metric independently for each class and then takes the average.
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\nMacro-averaged metrics:")
    print(f"Precision: {macro_precision:.4f}")
    print(f"Recall:    {macro_recall:.4f}")
    print(f"F1 Score:  {macro_f1:.4f}")

    # print a detailed classification report
    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4, zero_division=0))
    class_report = classification_report(y_true, y_pred, digits=4, zero_division=0, output_dict=True)
    # print(class_report)
    # class_report_df = pd.DataFrame(class_report).transpose()

    # if save_path:
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     cm_df.to_excel(save_path + f"/confusion_matrix_{task_type}.xlsx", index=True)
    #     class_report_df.to_excel(save_path + f"/classification_report_{task_type}.xlsx", index=True)

    return class_report, cm_df, cm, labels


def calc_confusion_matrix_and_metrics_new(data: dict):
    """
    计算多标签分类任务的各类指标。
    参数：
        data: dict，包括 y_true: list[list[str]], y_pred: list[list[str]]
    """
    y_true = data["y_true"]
    y_pred = data["y_pred"]

    # ① 使用 MultiLabelBinarizer 统一标签集合
    mlb = MultiLabelBinarizer()
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)
    classes = mlb.classes_

    # ② 混淆矩阵（每类一个 2x2）
    cm = multilabel_confusion_matrix(y_true_bin, y_pred_bin)

    # ③ 常用 F1 分数
    f1_micro = f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    f1_macro = f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true_bin, y_pred_bin, average="weighted", zero_division=0)

    # ④ 输出分类报告（可选）
    report = classification_report(y_true_bin, y_pred_bin, target_names=classes, zero_division=0)

    # ⑤ 打印每个标签的混淆矩阵
    print("Per-label confusion matrix:")
    for i, cls in enumerate(classes):
        tn, fp, fn, tp = cm[i].ravel()
        print(f"  [{cls}] TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # 返回指标和矩阵
    return {
        "confusion_matrices": cm,
        "classes": classes,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "report": report,
    }


def generate_report_and_table(excel_file_path: str, datasheet_index_list: list = [1, 2, 3, 5]):
    """
    excel_file_path is the one for the recognition collection, which matches the template.
    """
    output_folder_path = os.path.dirname(excel_file_path)

    result_table_prototypeData, result_table_unit, result_table_quantity = {}, {}, {}
    for ds_index in datasheet_index_list:
        if ds_index != 5:
            test_name = f"Test-{ds_index}"
        else:
            test_name = "merged"
        save_path = output_folder_path + "/" + test_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        data_prototypeData = read_data_from_excel(excel_file_path, ds_index, "C", "D", "E")
        [report_pd, cm_pd_df, cm_pd, labels_pd] = calc_confusion_matrix_and_metrics(data_prototypeData)
        result_table_prototypeData[test_name] = extract_classification_report(report_pd)

        data_unit = read_data_from_excel(excel_file_path, ds_index, "H", "I", "J")
        [report_un, cm_un_df, cm_un, labels_un] = calc_confusion_matrix_and_metrics(data_unit)
        result_table_unit[test_name] = extract_classification_report(report_un)

        data_quantity = read_data_from_excel(excel_file_path, ds_index, "L", "M", "N")
        [report_qu, cm_qu_df, cm_qu, labels_qu] = calc_confusion_matrix_and_metrics(data_quantity)
        result_table_quantity[test_name] = extract_classification_report(report_qu)

        heatmap_drawing = {
            "PrototypeData": [cm_pd, labels_pd],
            "Unit": [cm_un, labels_un],
            "Quantity": [cm_qu, labels_qu],
        }
        for target, value in heatmap_drawing.items():
            disp = ConfusionMatrixDisplay(confusion_matrix=value[0], display_labels=value[1])
            fig, ax = plt.subplots(figsize=(30, 30))
            disp.plot(ax=ax, cmap="Blues", colorbar=True)
            plt.xticks(rotation=90)
            plt.savefig(save_path + f"/heatmap_{target}.pdf", format="pdf", bbox_inches="tight")

        with pd.ExcelWriter(save_path + "/confusion_matrix.xlsx") as writer:
            cm_pd_df.to_excel(writer, index=True, sheet_name="prototypeData")
            cm_un_df.to_excel(writer, index=True, sheet_name="unit")
            cm_qu_df.to_excel(writer, index=True, sheet_name="quantity")

        with pd.ExcelWriter(save_path + "/classification_report.xlsx") as writer:
            cr_pd_df = pd.DataFrame(report_pd).transpose()
            cr_pd_df.to_excel(writer, index=True, sheet_name="prototypeData")
            cr_un_df = pd.DataFrame(report_un).transpose()
            cr_un_df.to_excel(writer, index=True, sheet_name="unit")
            cr_qu_df = pd.DataFrame(report_qu).transpose()
            cr_qu_df.to_excel(writer, index=True, sheet_name="quantity")

    with pd.ExcelWriter(output_folder_path + "/result_table.xlsx") as writer:
        result_table_df = pd.DataFrame(result_table_prototypeData)
        result_table_df.to_excel(writer, index=True, sheet_name="prototypeData")
        result_table_df = pd.DataFrame(result_table_unit)
        result_table_df.to_excel(writer, index=True, sheet_name="unit")
        result_table_df = pd.DataFrame(result_table_quantity)
        result_table_df.to_excel(writer, index=True, sheet_name="quantity")
        all_f1_score = extract_f1_score(result_table_quantity, result_table_unit, result_table_prototypeData)
        result_table_df = pd.DataFrame(all_f1_score)
        result_table_df.to_excel(writer, index=True, sheet_name="all_f1_score")


def extract_classification_report(class_report: dict):
    line = {
        "macro-precision (%)": round(100 * class_report["macro avg"]["precision"], 1),
        "macro-recall (%)": round(100 * class_report["macro avg"]["recall"], 1),
        "macro-f1-score (%)": round(100 * class_report["macro avg"]["f1-score"], 1),
        "macro-support": class_report["macro avg"]["support"],
        "weighted-precision (%)": round(100 * class_report["weighted avg"]["precision"], 1),
        "weighted-recall (%)": round(100 * class_report["weighted avg"]["recall"], 1),
        "weighted-f1-score (%)": round(100 * class_report["weighted avg"]["f1-score"], 1),
        "weighted-support": class_report["weighted avg"]["support"],
    }
    return line


def extract_f1_score(result_table_quantity: dict, result_table_unit: dict, result_table_prototypeData: dict):
    def update_f1_score(f1_score, source_dict, prefix):
        for key, value in source_dict.items():
            if key not in f1_score:
                f1_score[key] = {}
            f1_score[key].update(
                {
                    f"{prefix}-macro-f1-score (%)": value["macro-f1-score (%)"],
                    f"{prefix}-weighted-f1-score (%)": value["weighted-f1-score (%)"],
                }
            )

    f1_score = {}

    # 更新 f1_score 字典
    update_f1_score(f1_score, result_table_quantity, "quantity")
    update_f1_score(f1_score, result_table_unit, "unit")
    update_f1_score(f1_score, result_table_prototypeData, "prototypeData")

    return f1_score


def collect_3test_results(test_folder_path_list: list, output_path: str, template_excel_path: str = None):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if template_excel_path is None:
        template_excel_path = output_path + "/recognition_collection.xlsx"

    # 读取 template_xlsx 时禁用列名自动修改
    template_xls = pd.ExcelFile(template_excel_path)
    template_sheets = {
        sheet: pd.read_excel(template_excel_path, sheet_name=sheet, keep_default_na=False)
        for sheet in template_xls.sheet_names
    }

    result_sheets = []
    for test_path in test_folder_path_list:
        normalized_path = test_path.replace("\\", "/")  # 替换 \ 为 /
        project_id = os.path.basename(normalized_path)
        data_df = pd.read_excel(test_path + "/table_for_validation.xlsx", keep_default_na=False)

        target_sheet = ""
        for sheet_name in template_xls.sheet_names:
            if sheet_name in project_id:
                target_sheet = sheet_name
                result_sheets.append(target_sheet)

        if target_sheet != "":
            # 确保列名不会被改动
            data_df.columns = [col.split(".")[0] for col in data_df.columns]
            template_sheets[target_sheet].columns = [
                col.split(".")[0] for col in template_sheets[target_sheet].columns
            ]
            # 直接进行列替换
            columns_to_replace = [
                "PrototypeData_class (ddhub:)",
                "PrototypeData_class_candidates (ddhub:)",
                "Unit_class (ddhub:)",
                "Unit_class_candidates (ddhub:)",
                "Quantity_class (ddhub:)",
            ]
            template_sheets[target_sheet][columns_to_replace] = data_df[columns_to_replace]

        else:
            print(f"Error: Sheet '{project_id}' not found in {template_excel_path}")

    # merge the 3 tests results into one sheet
    merged_df = pd.concat([template_sheets[sheet] for sheet in result_sheets], ignore_index=True)
    # merged_df = merged_df.drop_duplicates(subset=["Mnemonic"], keep="last")
    template_sheets["merged"] = merged_df

    # 将修改后的数据保存回 Excel
    output_excel_path = output_path + "/recognition_collection.xlsx"
    with pd.ExcelWriter(output_excel_path) as writer:
        for sheet_name, df in template_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def run_evaluations(input_path: str, project_id_list: list, output_path: str, template_excel_path: str):
    project_folder_path_list = [f"{input_path}/{prj_id}" for prj_id in project_id_list]

    collect_3test_results(project_folder_path_list, output_path, template_excel_path)
    excel_file_path = output_path + "/recognition_collection.xlsx"
    generate_report_and_table(excel_file_path)


if __name__ == "__main__":
    currentFolder = os.path.dirname(os.path.realpath(__file__))

    project_package_folder_path_list = [
        currentFolder + r"/tasks/3Tests_no_pruning_20250516",
        currentFolder + r"/tasks/3Tests_baseline_20250518",
        currentFolder + r"/tasks/3Tests_cot_20250518",
        currentFolder + r"/tasks/3Tests_pc_20250515",
        currentFolder + r"/tasks/3Tests_pc_cot_20250515",
        currentFolder + r"/tasks/3Tests_baseline_20250518_gpt41mini",
        currentFolder + r"/tasks/3Tests_cot_20250518_gpt41mini",
        currentFolder + r"/tasks/3Tests_pc_20250518_gpt41mini",
    ]
    project_id_list = fix_project_id_list(
        [
            r"Norway-NA-15_$47$_9-F-9 A/1/log/1/1/1/00001",
            r"Norway-NA-15_$47$_9-F-9 A/1/log/2/1/1/00001",
            r"Norway-NA-15_$47$_9-F-9 A/1/log/2/2/1/00001",
        ]
    )
    evaluation_output_path_list = [
        currentFolder + r"/documents/data_for_paper/no_pruning_20250516",
        currentFolder + r"/documents/data_for_paper/baseline_20250518",
        currentFolder + r"/documents/data_for_paper/cot_20250518",
        currentFolder + r"/documents/data_for_paper/pc_20250515",
        currentFolder + r"/documents/data_for_paper/pc_cot_20250515",
        currentFolder + r"/documents/data_for_paper/baseline_20250518_gpt41mini",
        currentFolder + r"/documents/data_for_paper/cot_20250518_gpt41mini",
        currentFolder + r"/documents/data_for_paper/pc_20250518_gpt41mini",
    ]
    template_excel_path = currentFolder + r"/documents/data_for_paper/recognition_collection_template.xlsx"

    for i in range(len(project_package_folder_path_list)):
        project_package_folder_path = project_package_folder_path_list[i]
        output_path = evaluation_output_path_list[i]

        generate_table_batch(project_package_folder_path, project_id_list)
        run_evaluations(project_package_folder_path, project_id_list, output_path, template_excel_path)
