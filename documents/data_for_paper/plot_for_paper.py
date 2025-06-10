import math
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_excel(excel_path: str):
    df = pd.read_excel(
        excel_path,
        sheet_name="all_f1_score",
        usecols="A:E",
        header=0,
        keep_default_na=False,
    )
    df.columns = ["metric", "Test-1", "Test-2", "Test-3", "merged"]
    return df


def read_results_for_one_method():
    result_dict = {}
    currentFolder = os.path.dirname(os.path.realpath(__file__))
    excel_path_dict = {
        "Naive_v1-r1": currentFolder + r"/Naive_v1_20250408/result_table.xlsx",
        "Naive_v1-r2": currentFolder + r"/Naive_v1_20250410/result_table.xlsx",
        "Naive_v2-r1": currentFolder + r"/Naive_v2_20250409/result_table.xlsx",
        "Naive_v2-r2": currentFolder + r"/Naive_v2_20250410/result_table.xlsx",
        "CoT_v1-r1": currentFolder + r"/CoT_v1_20250404/result_table.xlsx",
        "CoT_v1-r2": currentFolder + r"/CoT_v1_20250410/result_table.xlsx",
        "CoT_v2-r1": currentFolder + r"/CoT_v2_20250403/result_table.xlsx",
        "CoT_v2-r2": currentFolder + r"/CoT_v2_20250409/result_table.xlsx",
    }
    for name, ep in excel_path_dict.items():
        result = read_excel(ep)
        result_dict[name] = result
    print(result_dict)
    return result_dict


def read_results_new():
    result_dict = {}
    currentFolder = os.path.dirname(os.path.realpath(__file__))
    excel_path_dict = {
        "no_pruning": currentFolder + r"/no_pruning_20250516/result_table.xlsx",
        "baseline": currentFolder + r"/baseline_20250518/result_table.xlsx",
        "cot": currentFolder + r"/cot_20250518/result_table.xlsx",
        "pc": currentFolder + r"/pc_20250515/result_table.xlsx",
        "pc_cot": currentFolder + r"/pc_cot_20250515/result_table.xlsx",
        "baseline_gpt41mini": currentFolder
        + r"/baseline_20250518_gpt41mini/result_table.xlsx",
        "cot_gpt41mini": currentFolder + r"/cot_20250518_gpt41mini/result_table.xlsx",
        "pc_gpt41mini": currentFolder + r"/pc_20250518_gpt41mini/result_table.xlsx",
    }
    for name, ep in excel_path_dict.items():
        result = read_excel(ep)
        result_dict[name] = result
    print(result_dict)
    return result_dict


def reshape_results(input_dict):
    # 初始化 f1_score_dict
    f1_score_dict = {
        "quantity": {"Test-1": {}, "Test-2": {}, "Test-3": {}, "merged": {}},
        "unit": {"Test-1": {}, "Test-2": {}, "Test-3": {}, "merged": {}},
        "prototypeData": {"Test-1": {}, "Test-2": {}, "Test-3": {}, "merged": {}},
    }

    # 遍历 input_dict
    for key, df in input_dict.items():
        # 提取方法名和运行编号
        method_name, run_id = key.split("-")
        # 将 'metric' 列设置为索引
        df.set_index("metric", inplace=True)
        # 遍历每个测试集
        for test in ["Test-1", "Test-2", "Test-3", "merged"]:
            # 提取 macro_f1 和 weighted_f1 分数
            # macro_f1 = df.loc['quantity-macro-f1-score (%)', test]
            # weighted_f1 = df.loc['quantity-weighted-f1-score (%)', test]
            # # 将数据添加到 f1_score_dict
            # f1_score_dict['quantity'][test].setdefault(f'macro_f1_{run_id}', []).append(macro_f1)
            # f1_score_dict['quantity'][test].setdefault(f'weighted_f1_{run_id}', []).append(weighted_f1)
            # 对于 'unit' 和 'prototypeData'，假设它们的顺序与 'quantity' 相同
            for i, target in enumerate(["quantity", "unit", "prototypeData"]):
                macro_f1 = df.loc[f"{target}-macro-f1-score (%)", test]
                weighted_f1 = df.loc[f"{target}-weighted-f1-score (%)", test]
                f1_score_dict[target][test].setdefault(f"macro_f1_{run_id}", []).append(
                    macro_f1
                )
                f1_score_dict[target][test].setdefault(
                    f"weighted_f1_{run_id}", []
                ).append(weighted_f1)
    print(f1_score_dict)
    return f1_score_dict


def reshape_results_new(input_dict):
    # 初始化 f1_score_dict
    f1_score_dict = {
        "quantity": {"Test-1": {}, "Test-2": {}, "Test-3": {}, "merged": {}},
        "unit": {"Test-1": {}, "Test-2": {}, "Test-3": {}, "merged": {}},
        "prototypeData": {"Test-1": {}, "Test-2": {}, "Test-3": {}, "merged": {}},
    }

    # 遍历 input_dict
    for key, df in input_dict.items():
        # 提取方法名和运行编号
        # method_name, run_id = key.split("-")
        # 将 'metric' 列设置为索引
        df.set_index("metric", inplace=True)
        # 遍历每个测试集
        for test in ["Test-1", "Test-2", "Test-3", "merged"]:
            for i, target in enumerate(["quantity", "unit", "prototypeData"]):
                macro_f1 = df.loc[f"{target}-macro-f1-score (%)", test]
                weighted_f1 = df.loc[f"{target}-weighted-f1-score (%)", test]
                f1_score_dict[target][test].setdefault(f"macro_f1", []).append(macro_f1)
                f1_score_dict[target][test].setdefault(f"weighted_f1", []).append(
                    weighted_f1
                )
    print(f1_score_dict)
    return f1_score_dict


def draw_errorbar(f1_score_dict):
    # 方法名称
    methods = ["Naive_v1", "Naive_v2", "CoT_v1", "CoT_v2"]

    for target in ["quantity", "unit", "prototypeData"]:
        # 创建一个包含3个子图（对应3个测试集）的图形，每个子图包含Macro F1和Weighted F1的误差条
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

        f1_score = f1_score_dict[target]
        # 遍历每个测试集
        for i, test_name in enumerate(["Test-1", "Test-2", "Test-3"]):
            # 生成示例数据：每种方法两次运行的 Macro F1 和 Weighted F1 分数
            macro_f1_run1 = np.array(f1_score[test_name]["macro_f1_r1"])
            macro_f1_run2 = np.array(f1_score[test_name]["macro_f1_r2"])
            weighted_f1_run1 = np.array(f1_score[test_name]["weighted_f1_r1"])
            weighted_f1_run2 = np.array(f1_score[test_name]["weighted_f1_r2"])

            # 计算平均值和误差（两次运行之间的差异）
            macro_f1_mean = (macro_f1_run1 + macro_f1_run2) / 2
            macro_f1_error = np.abs(macro_f1_run1 - macro_f1_run2) / 2
            weighted_f1_mean = (weighted_f1_run1 + weighted_f1_run2) / 2
            weighted_f1_error = np.abs(weighted_f1_run1 - weighted_f1_run2) / 2

            # 设置x轴位置
            x = np.arange(len(methods))

            # 绘制Macro F1的误差条
            axes[i].errorbar(
                # x - 0.2,
                x,
                macro_f1_mean,
                yerr=macro_f1_error,
                fmt="o",
                capsize=5,
                ecolor="blue",
                elinewidth=2,
                linestyle="-",
                label="Macro F1",
            )

            # 绘制Weighted F1的误差条
            axes[i].errorbar(
                # x + 0.2,
                x,
                weighted_f1_mean,
                yerr=weighted_f1_error,
                fmt="D",
                capsize=5,
                ecolor="green",
                elinewidth=2,
                linestyle="--",
                label="Weighted F1",
            )

            # 设置图表标题和标签
            axes[i].set_title(f"{test_name} Results")
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(methods)
            # axes[i].set_xlabel("Methods")
            axes[i].set_ylabel("F1 Score")
            axes[i].legend()

        # 自动调整布局
        figure_title = target[0].upper() + target[1:]
        fig.suptitle(figure_title, fontsize=14)
        plt.tight_layout()
        # plt.show()
        currentFolder = os.path.dirname(os.path.realpath(__file__))
        plt.savefig(currentFolder + f"/{figure_title}_errorbar.pdf")


def draw_barchart(f1_score_dict):
    methods = ["Naive", "Naive_diff_order", "CoT_direct", "CoT_show_analysis"]

    for target in ["quantity", "unit", "prototypeData"]:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharey=True)

        f1_score = f1_score_dict[target]
        for i, test_name in enumerate(["Test-1", "Test-2", "Test-3"]):
            macro_f1_run1 = np.array(f1_score[test_name]["macro_f1_r1"])
            macro_f1_run2 = np.array(f1_score[test_name]["macro_f1_r2"])
            weighted_f1_run1 = np.array(f1_score[test_name]["weighted_f1_r1"])
            weighted_f1_run2 = np.array(f1_score[test_name]["weighted_f1_r2"])

            macro_f1_mean = (macro_f1_run1 + macro_f1_run2) / 2
            macro_f1_error = np.abs(macro_f1_run1 - macro_f1_run2) / 2
            weighted_f1_mean = (weighted_f1_run1 + weighted_f1_run2) / 2
            weighted_f1_error = np.abs(weighted_f1_run1 - weighted_f1_run2) / 2

            x = np.arange(len(methods))
            width = 0.35
            min_f1_est = 0
            fontsize_min = 17

            axes[i].bar(
                x,
                macro_f1_mean - min_f1_est,
                width,
                bottom=min_f1_est,
                yerr=macro_f1_error,
                label="Macro F1",
                alpha=0.8,
            )
            axes[i].bar(
                x + width,
                weighted_f1_mean - min_f1_est,
                width,
                bottom=min_f1_est,
                yerr=weighted_f1_error,
                label="Weighted F1",
                alpha=0.8,
            )

            axes[i].set_title(f"{test_name} Results", fontsize=fontsize_min + 2)
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(methods, fontsize=fontsize_min, rotation=18)
            axes[i].set_ylim(min_f1_est, 100)
            axes[i].tick_params(axis="y", labelsize=fontsize_min)
            axes[i].tick_params(axis="x", labelsize=fontsize_min)

            # 只给第一个 subplot 设置 y label
            if i == 0:
                axes[i].set_ylabel("F1 Score", fontsize=fontsize_min)
            else:
                axes[i].tick_params(labelleft=False)

        # 设置总标题
        figure_title = target[0].upper() + target[1:]
        fig.suptitle(figure_title, fontsize=fontsize_min + 3, y=0.95)

        # 添加统一图例（避免每个子图都有）
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", fontsize=fontsize_min)

        # 调整布局，避免 suptitle 被覆盖
        plt.tight_layout(rect=[0, 0, 1.02, 0.95])

        # 保存图像
        currentFolder = os.path.dirname(os.path.realpath(__file__))
        plt.savefig(
            os.path.join(currentFolder, f"{figure_title}_bar.pdf"), transparent=True
        )
        plt.close()


def draw_bar_merged(f1_score_dict):
    methods = ["Naive", "Naive_diff_order", "CoT_direct", "CoT_show_analysis"]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharey=True)
    for i, target in enumerate(["quantity", "unit", "prototypeData"]):
        f1_score = f1_score_dict[target]
        # for i, test_name in enumerate(["merged"]):
        #     ...
        test_name = "merged"
        macro_f1_run1 = np.array(f1_score[test_name]["macro_f1_r1"])
        macro_f1_run2 = np.array(f1_score[test_name]["macro_f1_r2"])
        weighted_f1_run1 = np.array(f1_score[test_name]["weighted_f1_r1"])
        weighted_f1_run2 = np.array(f1_score[test_name]["weighted_f1_r2"])

        macro_f1_mean = (macro_f1_run1 + macro_f1_run2) / 2
        macro_f1_error = np.abs(macro_f1_run1 - macro_f1_run2) / 2
        weighted_f1_mean = (weighted_f1_run1 + weighted_f1_run2) / 2
        weighted_f1_error = np.abs(weighted_f1_run1 - weighted_f1_run2) / 2

        x = np.arange(len(methods))
        width = 0.35
        min_f1_est = 0
        fontsize_min = 17

        axes[i].bar(
            x,
            macro_f1_mean - min_f1_est,
            width,
            bottom=min_f1_est,
            yerr=macro_f1_error,
            label="Macro F1",
            alpha=0.8,
        )
        axes[i].bar(
            x + width,
            weighted_f1_mean - min_f1_est,
            width,
            bottom=min_f1_est,
            yerr=weighted_f1_error,
            label="Weighted F1",
            alpha=0.8,
        )
        subplot_title = target[0].upper() + target[1:]
        axes[i].set_title(subplot_title, fontsize=fontsize_min + 2)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(methods, fontsize=fontsize_min, rotation=18)
        axes[i].set_ylim(min_f1_est, 100)
        axes[i].tick_params(axis="y", labelsize=fontsize_min)
        axes[i].tick_params(axis="x", labelsize=fontsize_min)

        # 只给第一个 subplot 设置 y label
        if i == 0:
            axes[i].set_ylabel("F1 Score", fontsize=fontsize_min)
        else:
            axes[i].tick_params(labelleft=False)

    # 设置总标题
    # figure_title = target[0].upper() + target[1:]
    # fig.suptitle(figure_title, fontsize=fontsize_min + 3, y=0.95)

    # 添加统一图例（避免每个子图都有）
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=fontsize_min)

    # 调整布局，避免 suptitle 被覆盖
    plt.tight_layout(rect=[0, 0, 1.02, 0.95])

    # 保存图像
    currentFolder = os.path.dirname(os.path.realpath(__file__))
    plt.savefig(os.path.join(currentFolder, "merged_bar.pdf"), transparent=True)
    plt.close()


def draw_bar_merged_new(f1_score_dict):
    methods = [
        "No_pruning",
        "Baseline",
        "CoT",
        "PC",
        "PC_CoT",
        "Baseline_h",
        "CoT_h",
        "PC_h",
    ]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharey=True)
    for i, target in enumerate(["quantity", "unit", "prototypeData"]):
        f1_score = f1_score_dict[target]
        # for i, test_name in enumerate(["merged"]):
        #     ...
        test_name = "merged"
        macro_f1 = np.array(f1_score[test_name]["macro_f1"])
        weighted_f1 = np.array(f1_score[test_name]["weighted_f1"])

        x = np.arange(len(methods))
        width = 0.35
        min_f1_est = 0
        fontsize_min = 17

        axes[i].bar(
            x,
            macro_f1 - min_f1_est,
            width,
            bottom=min_f1_est,
            # yerr=macro_f1_error,
            label="Macro F1",
            alpha=0.8,
        )
        axes[i].bar(
            x + width,
            weighted_f1 - min_f1_est,
            width,
            bottom=min_f1_est,
            # yerr=weighted_f1_error,
            label="Weighted F1",
            alpha=0.8,
        )
        subplot_title = target[0].upper() + target[1:]
        axes[i].set_title(subplot_title, fontsize=fontsize_min + 2)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(methods, fontsize=fontsize_min, rotation=30)
        axes[i].set_ylim(min_f1_est, 100)
        axes[i].tick_params(axis="y", labelsize=fontsize_min)
        axes[i].tick_params(axis="x", labelsize=fontsize_min)

        # 只给第一个 subplot 设置 y label
        if i == 0:
            axes[i].set_ylabel("F1 Score (%)", fontsize=fontsize_min)
        else:
            axes[i].tick_params(labelleft=False)

    # 设置总标题
    # figure_title = target[0].upper() + target[1:]
    # fig.suptitle(figure_title, fontsize=fontsize_min + 3, y=0.95)

    # 添加统一图例（避免每个子图都有）
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=fontsize_min, ncol=2)

    # 调整布局，避免 suptitle 被覆盖
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # 保存图像
    currentFolder = os.path.dirname(os.path.realpath(__file__))
    plt.savefig(os.path.join(currentFolder, "merged_bar.pdf"), transparent=True)
    plt.close()


if __name__ == "__main__":
    # result = read_results_for_one_method()
    # f1_score_dict = reshape_results(result)
    # # draw_errorbar(f1_score_dict)
    # # draw_barchart(f1_score_dict)
    # draw_bar_merged(f1_score_dict)

    result = read_results_new()
    f1_score_dict = reshape_results_new(result)
    draw_bar_merged_new(f1_score_dict)
