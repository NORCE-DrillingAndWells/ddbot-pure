import requests, os, json, typing
import preprocessor as pp
import rag_task_manager as rag
import ddhub_model_builder as mb
import task_manager_pc as tm_pc
import task_manager_pc_cot as tm_pc_cot
import task_manager_no_pruning as tm_no_pruning
import task_manager_baseline as tm_baseline
import task_manager_cot as tm_cot
import utils as utils

# import log_config


def semanticize_raw_metadata_batch():
    currentFolder = os.path.dirname(os.path.realpath(__file__))
    metadata_profile = utils.metadata_profile_dict["Volve open data"]
    file_path_list = utils.fix_project_id_list(
        [
            # r"C:\Datasets\Volve open data\WITSML Realtime drilling data\Norway-NA-15_$47$_9-F-9 A\1\log\1\1\1\00001.xml",
            # r"C:\Datasets\Volve open data\WITSML Realtime drilling data\Norway-NA-15_$47$_9-F-9 A\1\log\1\2\1\00001.xml",
            r"C:\Datasets\Volve open data\WITSML Realtime drilling data\Norway-NA-15_$47$_9-F-9 A\1\log\2\1\1\00001.xml",
            r"C:\Datasets\Volve open data\WITSML Realtime drilling data\Norway-NA-15_$47$_9-F-9 A\1\log\2\2\1\00001.xml",
            # r"C:\Datasets\Volve open data\WITSML Realtime drilling data\Norway-NA-15_$47$_9-F-1\1\log\1\2\1\00001.xml",
            r"C:\Datasets\Volve open data\WITSML Realtime drilling data\Norway-Statoil-NO 15_$47$_9-F-9\1\log\1\1\1\00001.xml",
            # r"C:\Datasets\Volve open data\WITSML Realtime drilling data\Norway-Statoil-NO 15_$47$_9-F-9\1\log\1\2\1\00001.xml",
            # r"C:\Datasets\Volve open data\WITSML Realtime drilling data\Norway-Statoil-NO 15_$47$_9-F-9\1\log\1\5\1\00001.xml",
            # r"C:\Datasets\Volve open data\WITSML Realtime drilling data\Norway-Statoil-NO 15_$47$_9-F-9\1\log\1\7\1\00001.xml",
            # r"C:\Datasets\Volve open data\WITSML Realtime drilling data\Norway-Statoil-NO 15_$47$_9-F-9\1\log\1\9\1\00001.xml",
        ]
    )
    project_id_list = [
        # r"Norway-NA-15_$47$_9-F-9 A\1\log\1\1\1\00001",
        # r"Norway-NA-15_$47$_9-F-9 A\1\log\1\2\1\00001",
        r"Norway-NA-15_$47$_9-F-9 A\1\log\2\1\1\00001",
        r"Norway-NA-15_$47$_9-F-9 A\1\log\2\2\1\00001",
        # r"Norway-NA-15_$47$_9-F-1\1\log\1\2\1\00001",
        r"Norway-Statoil-NO 15_$47$_9-F-9\1\log\1\1\1\00001",
        # r"Norway-Statoil-NO 15_$47$_9-F-9\1\log\1\2\1\00001",
        # r"Norway-Statoil-NO 15_$47$_9-F-9\1\log\1\5\1\00001",
        # r"Norway-Statoil-NO 15_$47$_9-F-9\1\log\1\7\1\00001",
        # r"Norway-Statoil-NO 15_$47$_9-F-9\1\log\1\9\1\00001",
    ]

    for i in range(len(file_path_list)):
        # Fix project_id and generate namespace
        project_id = project_id_list[i]
        metadata_profile["Namespace"] = "http://ddhub.demo/zzz/<project_id>".replace("<project_id>", project_id)
        project_info = {
            "project_id": project_id,
            "output_path": currentFolder + "/tasks/3Tests_cot2/",
            "raw_metadata_file_path": file_path_list[i],
            "raw_metadata_file_reader": pp.read_from_xml_onsite,
            "prompt_templates_path": currentFolder + "/tasks/prompt_templates_pc_cot.yaml",
            "metadata_profile": metadata_profile,
            "user_interaction": False,
            "models_high_low": ["gpt-4o-mini", "gpt-4o-mini"],
        }
        tm_pc_cot.semanticize_raw_metadata(**project_info)
        # print(project_info)

        # ddhub_model_builder
        # project_folder_path = project_info["output_path"] + "/" + project_info["project_id"]
        # mb.generate_AllinOne(project_folder_path)


def semanticize_raw_metadata_batch_pc_cot_paper():
    currentFolder = os.path.dirname(os.path.realpath(__file__))
    metadata_profile = utils.metadata_profile_dict["Volve open data"]
    # path_dataset = "/home/AD.NORCERESEARCH.NO/lizh/Datasets"
    path_dataset = "/mnt/c/datasets/"
    file_path_list = [
        # r"C:\GitRepo\test\xml_files\Norway-NA-15__47__9-F-9_A+1+log+2+1+1+00001_selected-2.xml",
        # path_dataset
        # + r"/Volve open data/WITSML Realtime drilling data/Norway-NA-15_$47$_9-F-9 A/1/log/1/1/1/00001.xml",
        # path_dataset
        # + r"/Volve open data/WITSML Realtime drilling data/Norway-NA-15_$47$_9-F-9 A/1/log/2/1/1/00001.xml",
        # path_dataset
        # + r"/Volve open data/WITSML Realtime drilling data/Norway-NA-15_$47$_9-F-9 A/1/log/2/2/1/00001.xml",
        currentFolder
        + r"/data_store/mnemonic_onsite/00001_demo.xml"
    ]
    project_id_list = utils.fix_project_id_list(
        [
            # r"test_selected",
            # r"Norway-NA-15_$47$_9-F-9 A\1\log\1\1\1\00001",
            # r"Norway-NA-15_$47$_9-F-9 A\1\log\2\1\1\00001",
            # r"Norway-NA-15_$47$_9-F-9 A\1\log\2\2\1\00001",
            "00001_demo"
        ]
    )
    project_package_folder_path = currentFolder + "/tasks/3Tests_pc_cot_20250515/"

    for i in range(len(file_path_list)):
        # Fix project_id and generate namespace
        project_id = project_id_list[i]
        metadata_profile["Namespace"] = "http://ddhub.demo/zzz/<project_id>".replace("<project_id>", project_id)

        project_info = {
            "project_id": project_id,
            "output_path": project_package_folder_path,
            "raw_metadata_file_path": file_path_list[i],
            "raw_metadata_file_reader": pp.read_from_xml_onsite,
            "prompt_templates_path": currentFolder + "/tasks/prompt_templates_pc_cot.yaml",
            "metadata_profile": metadata_profile,
            "user_interaction": False,
            "models_high_low": ["gpt-4o-mini", "gpt-4o-mini"],
        }
        tm_pc_cot.semanticize_raw_metadata(**project_info)
        # print(project_info)

        # ddhub_model_builder
        # project_folder_path = project_info["output_path"] + "/" + project_info["project_id"]
        # mb.generate_AllinOne(project_folder_path)


def semanticize_raw_metadata_batch_pc_paper():
    currentFolder = os.path.dirname(os.path.realpath(__file__))
    metadata_profile = utils.metadata_profile_dict["Volve open data"]
    # path_dataset = "/home/AD.NORCERESEARCH.NO/lizh/Datasets"
    path_dataset = "/mnt/c/datasets/"
    file_path_list = [
        # r"C:\GitRepo\test\xml_files\Norway-NA-15__47__9-F-9_A+1+log+2+1+1+00001_selected-2.xml",
        path_dataset
        + r"/Volve open data/WITSML Realtime drilling data/Norway-NA-15_$47$_9-F-9 A/1/log/1/1/1/00001.xml",
        path_dataset
        + r"/Volve open data/WITSML Realtime drilling data/Norway-NA-15_$47$_9-F-9 A/1/log/2/1/1/00001.xml",
        path_dataset
        + r"/Volve open data/WITSML Realtime drilling data/Norway-NA-15_$47$_9-F-9 A/1/log/2/2/1/00001.xml",
        # currentFolder
        # + r"/data_store/mnemonic_onsite/00001_demo.xml"
    ]
    project_id_list = utils.fix_project_id_list(
        [
            # r"test_selected",
            r"Norway-NA-15_$47$_9-F-9 A\1\log\1\1\1\00001",
            r"Norway-NA-15_$47$_9-F-9 A\1\log\2\1\1\00001",
            r"Norway-NA-15_$47$_9-F-9 A\1\log\2\2\1\00001",
            # "00001_demo"
        ]
    )
    project_package_folder_path = currentFolder + "/tasks/3Tests_pc_20250530_gemini25f/"

    for i in range(len(file_path_list)):
        # Fix project_id and generate namespace
        project_id = project_id_list[i]
        metadata_profile["Namespace"] = "http://ddhub.demo/zzz/<project_id>".replace("<project_id>", project_id)

        project_info = {
            "project_id": project_id,
            "output_path": project_package_folder_path,
            "raw_metadata_file_path": file_path_list[i],
            "raw_metadata_file_reader": pp.read_from_xml_onsite,
            "prompt_templates_path": currentFolder + "/tasks/prompt_templates_pc.yaml",
            "metadata_profile": metadata_profile,
            "user_interaction": False,
            "models_high_low": [
                "gemini-2.5-flash-preview-05-20",
                "gemini-2.5-flash-preview-05-20",
                # "gpt-4o-mini",
                # "gpt-4o-mini",
            ],
        }
        tm_pc.semanticize_raw_metadata(**project_info)
        # print(project_info)

        # ddhub_model_builder
        # project_folder_path = project_info["output_path"] + "/" + project_info["project_id"]
        # mb.generate_AllinOne(project_folder_path)


def semanticize_raw_metadata_batch_no_pruning_paper():
    currentFolder = os.path.dirname(os.path.realpath(__file__))
    metadata_profile = utils.metadata_profile_dict["Volve open data"]
    # path_dataset = "/home/AD.NORCERESEARCH.NO/lizh/Datasets"
    path_dataset = "/mnt/c/datasets/"
    file_path_list = [
        # r"C:\GitRepo\test\xml_files\Norway-NA-15__47__9-F-9_A+1+log+2+1+1+00001_selected-2.xml",
        # r"C:\Datasets\Volve open data\WITSML Realtime drilling data\Norway-NA-15_$47$_9-F-9 A\1\log\1\1\1\00001.xml",
        # r"C:\Datasets\Volve open data\WITSML Realtime drilling data\Norway-NA-15_$47$_9-F-9 A\1\log\2\1\1\00001.xml",
        # r"C:\Datasets\Volve open data\WITSML Realtime drilling data\Norway-NA-15_$47$_9-F-9 A\1\log\2\2\1\00001.xml",
        # path_dataset + r"/Volve open data/WITSML Realtime drilling data/Norway-NA-15_$47$_9-F-9 A/1/log/1/1/1/00001.xml",
        # path_dataset + r"/Volve open data/WITSML Realtime drilling data/Norway-NA-15_$47$_9-F-9 A/1/log/2/1/1/00001.xml",
        # path_dataset + r"/Volve open data/WITSML Realtime drilling data/Norway-NA-15_$47$_9-F-9 A/1/log/2/2/1/00001.xml",
        currentFolder
        + r"/data_store/mnemonic_onsite/00001_demo.xml"
    ]
    project_id_list = utils.fix_project_id_list(
        [
            # "Norway-NA-15__47__9-F-9_A+1+log+2+1+1+00001_selected-2",
            # r"Norway-NA-15_$47$_9-F-9 A\1\log\1\1\1\00001",
            # r"Norway-NA-15_$47$_9-F-9 A\1\log\2\1\1\00001",
            # r"Norway-NA-15_$47$_9-F-9 A\1\log\2\2\1\00001",
            "00001_demo"
        ]
    )
    project_package_folder_path = currentFolder + "/tasks/3Tests_no_pruning_20250516/"
    for i in range(len(file_path_list)):
        # Fix project_id and generate namespace
        project_id = project_id_list[i]
        metadata_profile["Namespace"] = "http://ddhub.demo/zzz/<project_id>".replace("<project_id>", project_id)

        project_info = {
            "project_id": project_id,
            "output_path": project_package_folder_path,
            "raw_metadata_file_path": file_path_list[i],
            "raw_metadata_file_reader": pp.read_from_xml_onsite,
            "prompt_templates_path": currentFolder + "/tasks/prompt_templates_pc.yaml",
            "metadata_profile": metadata_profile,
            # "user_interaction": False,
            # "models_high_low": ["gpt-4o-mini", "gpt-4o-mini"],
        }
        tm_no_pruning.semanticize_raw_metadata(**project_info)
        # print(project_info)

        # ddhub_model_builder
        # project_folder_path = project_info["output_path"] + "/" + project_info["project_id"]
        # mb.generate_AllinOne(project_folder_path)


def semanticize_raw_metadata_batch_baseline_paper():
    currentFolder = os.path.dirname(os.path.realpath(__file__))
    metadata_profile = utils.metadata_profile_dict["Volve open data"]
    # path_dataset = "/home/AD.NORCERESEARCH.NO/lizh/Datasets"
    path_dataset = "/mnt/c/datasets/"
    file_path_list = [
        # r"C:\GitRepo\test\xml_files\Norway-NA-15__47__9-F-9_A+1+log+2+1+1+00001_selected-2.xml",
        # path_dataset
        # + r"/Volve open data/WITSML Realtime drilling data/Norway-NA-15_$47$_9-F-9 A/1/log/1/1/1/00001.xml",
        # path_dataset
        # + r"/Volve open data/WITSML Realtime drilling data/Norway-NA-15_$47$_9-F-9 A/1/log/2/1/1/00001.xml",
        # path_dataset
        # + r"/Volve open data/WITSML Realtime drilling data/Norway-NA-15_$47$_9-F-9 A/1/log/2/2/1/00001.xml",
        currentFolder
        + r"/data_store/mnemonic_onsite/00001_demo.xml"
    ]
    project_id_list = utils.fix_project_id_list(
        [
            # "Norway-NA-15__47__9-F-9_A+1+log+2+1+1+00001_selected-2",
            # r"Norway-NA-15_$47$_9-F-9 A\1\log\1\1\1\00001",
            # r"Norway-NA-15_$47$_9-F-9 A\1\log\2\1\1\00001",
            # r"Norway-NA-15_$47$_9-F-9 A\1\log\2\2\1\00001",
            "00001_demo"
        ]
    )
    project_package_folder_path = currentFolder + "/tasks/3Tests_baseline_20250518/"

    for i in range(len(file_path_list)):
        # Fix project_id and generate namespace
        project_id = project_id_list[i]
        metadata_profile["Namespace"] = "http://ddhub.demo/zzz/<project_id>".replace("<project_id>", project_id)

        project_info = {
            "project_id": project_id,
            "output_path": project_package_folder_path,
            "raw_metadata_file_path": file_path_list[i],
            "raw_metadata_file_reader": pp.read_from_xml_onsite,
            "prompt_templates_path": currentFolder + "/tasks/prompt_templates_pc.yaml",
            "metadata_profile": metadata_profile,
            # "user_interaction": False,
            "models_high_low": [
                # "gemini-2.5-flash-preview-05-20",
                # "gemini-2.5-flash-preview-05-20",
                "gpt-4o-mini",
                "gpt-4o-mini",
            ],
        }
        tm_baseline.semanticize_raw_metadata(**project_info)
        # print(project_info)

        # ddhub_model_builder
        # project_folder_path = project_info["output_path"] + "/" + project_info["project_id"]
        # mb.generate_AllinOne(project_folder_path)


def semanticize_raw_metadata_batch_cot_paper():
    currentFolder = os.path.dirname(os.path.realpath(__file__))
    metadata_profile = utils.metadata_profile_dict["Volve open data"]
    # path_dataset = "/home/AD.NORCERESEARCH.NO/lizh/Datasets"
    # path_dataset = "/mnt/c/datasets/"
    path_dataset = "/mnt/c/datasets/"
    file_path_list = [
        # r"C:\GitRepo\test\xml_files\Norway-NA-15__47__9-F-9_A+1+log+2+1+1+00001_selected-2.xml",
        # path_dataset
        # + r"/Volve open data/WITSML Realtime drilling data/Norway-NA-15_$47$_9-F-9 A/1/log/1/1/1/00001.xml",
        # path_dataset
        # + r"/Volve open data/WITSML Realtime drilling data/Norway-NA-15_$47$_9-F-9 A/1/log/2/1/1/00001.xml",
        # path_dataset
        # + r"/Volve open data/WITSML Realtime drilling data/Norway-NA-15_$47$_9-F-9 A/1/log/2/2/1/00001.xml",
        currentFolder
        + r"/data_store/mnemonic_onsite/00001_demo.xml"
    ]
    project_id_list = utils.fix_project_id_list(
        [
            # "Norway-NA-15__47__9-F-9_A+1+log+2+1+1+00001_selected-2",
            # r"Norway-NA-15_$47$_9-F-9 A\1\log\1\1\1\00001",
            # r"Norway-NA-15_$47$_9-F-9 A\1\log\2\1\1\00001",
            # r"Norway-NA-15_$47$_9-F-9 A\1\log\2\2\1\00001",
            "00001_demo"
        ]
    )
    project_package_folder_path = currentFolder + "/tasks/3Tests_cot_20250518/"

    for i in range(len(file_path_list)):
        # Fix project_id and generate namespace
        project_id = project_id_list[i]
        metadata_profile["Namespace"] = "http://ddhub.demo/zzz/<project_id>".replace("<project_id>", project_id)

        project_info = {
            "project_id": project_id,
            "output_path": project_package_folder_path,
            "raw_metadata_file_path": file_path_list[i],
            "raw_metadata_file_reader": pp.read_from_xml_onsite,
            "prompt_templates_path": currentFolder + "/tasks/prompt_templates_cot.yaml",
            "metadata_profile": metadata_profile,
            # "user_interaction": False,
            "models_high_low": [
                # "gemini-2.5-flash-preview-05-20",
                # "gemini-2.5-flash-preview-05-20",
                "gpt-4o-mini",
                "gpt-4o-mini",
            ],
        }
        tm_cot.semanticize_raw_metadata(**project_info)
        # print(project_info)

        # ddhub_model_builder
        # project_folder_path = project_info["output_path"] + "/" + project_info["project_id"]
        # mb.generate_AllinOne(project_folder_path)


if __name__ == "__main__":
    print("Choose a method to run:")
    print("1 - no_pruning")
    print("2 - baseline")
    print("3 - cot")
    print("4 - pc")
    print("5 - pc_cot")

    choice = input("Enter number (1-5): ").strip()

    if choice == "1":
        semanticize_raw_metadata_batch_no_pruning_paper()
    elif choice == "2":
        semanticize_raw_metadata_batch_baseline_paper()
    elif choice == "3":
        semanticize_raw_metadata_batch_cot_paper()
    elif choice == "4":
        semanticize_raw_metadata_batch_pc_paper()
    elif choice == "5":
        semanticize_raw_metadata_batch_pc_cot_paper()
    else:
        print("Invalid choice.")
