import os, json, yaml, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import chat_restapi as llm
import textdistance
import logging
from utils import filter_user_query

# import preprocessor as pp
# import retriever as rt

logger = logging.getLogger(__name__)


def sim_cos(s1: str, s2: str) -> float:
    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit and transform the sentences
    tfidf_matrix = tfidf_vectorizer.fit_transform([s1, s2])

    # Calculate the cosine similarity between the TF-IDF vectors of the two sentences
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])

    # Print the similarity score
    # print(f"Similarity between 's1' and 's2': {similarity[0][0]}")
    return similarity[0][0]


def sim_jaccard(s1: str, s2: str) -> float:
    similarity = textdistance.Jaccard.similarity(s1, s2)

    # print(f"Similarity between 's1' and 's2': {similarity[0][0]}")
    return similarity


# def filter_user_query(user_query: dict, selected_keys: list = ["Mnemonic", "Description"]) -> dict:
#     """
#     Reduce the distractions.
#     """
#     user_query_filtered = {}
#     for k in user_query:
#         if k in selected_keys:
#             user_query_filtered[k] = user_query[k]
#     return user_query_filtered


def mnemonic_split(mnemonic: str) -> str:
    description = mnemonic

    # E.g., GS_TV07, RPM30s, ROP2M, HKLD30s
    match = re.search(r"(\D+)(\d.*)", description)
    if match:
        description = f"{match.group(1)} {match.group(2)}"

    # The joint of keywords, e.g., ECD_MW_IN, GS_ROP
    description = description.replace("_", " ")

    # The joint of keywords, e.g., ECD_MW_IN
    keywords = ["WOB", "ROP", "SPP", "SPM", "RPM"]
    for keyword in keywords:
        if keyword in description:
            description = description.replace(keyword, f" {keyword}")
    return description


def retrieve_relatedContext(
    user_query: dict, context_toSelect: dict, topX: int = 4
) -> list:
    similarity_score = {}
    relatedContext = []
    if user_query["Description"] == "zzz:undefined":
        user_query["Description"] = mnemonic_split(user_query["Mnemonic"])

    user_query_filtered = filter_user_query(
        user_query, ["Mnemonic", "Description", "interpretation"]
    )
    user_query_str = str(user_query_filtered)

    for key, value_dict in context_toSelect.items():
        DDHubTerminology_str = str(value_dict)
        similarity_score.update({key: sim_cos(user_query_str, DDHubTerminology_str)})
    key_top_scores = sorted(similarity_score, key=similarity_score.get, reverse=True)[
        :topX
    ]
    for key in key_top_scores:
        relatedContext.append(context_toSelect[key])
    return relatedContext


def generate_query_and_context_batch(
    preprocessed_metadata_batch: dict, context_toSelect: dict
) -> dict:
    """
    Input:
    preprocessed_metadata_batch is processed from raw metadata.
    context_toSelect is a dict
    Output:
    query_and_context_batch = {"<DrillingDataPoint_name>": {"user_query": "", "relatedContext": ""}}
    """
    query_and_context_batch = {}
    for user_query in preprocessed_metadata_batch.values():
        # user_query = pp.preprocess_mnemonic_metadata(value_dict)
        mnemonic = user_query["Mnemonic"]
        user_query_filtered = filter_user_query(
            user_query,
            ["Mnemonic", "Description", "Unit", "DataType", "interpretation"],
        )
        relatedContext = retrieve_relatedContext(user_query_filtered, context_toSelect)
        query_and_context_batch.update(
            {
                mnemonic: {
                    "user_query": str(user_query_filtered),
                    "relatedContext": str(relatedContext),
                }
            }
        )
    return query_and_context_batch


def assemble_prompt_old(
    prompt_template: str, user_query: str, relatedContext: str
) -> str:
    prompt = prompt_template.replace("user_query", user_query).replace(
        "relatedContext", relatedContext
    )
    return prompt


def assemble_prompt(prompt_template: str, kvPairs: dict) -> str:
    prompt = prompt_template
    for keyword in kvPairs.keys():
        if (kvPairs[keyword] is None) or (len(kvPairs[keyword]) == 0):
            continue
        prompt = prompt.replace(keyword, str(kvPairs[keyword]))
    return prompt


def generate_rag_task_batch(
    query_and_context_batch: dict,
    prompt_template: str,
) -> dict:
    task_batch = {}
    for mnemonic, query_and_context in query_and_context_batch.items():
        user_query = query_and_context["user_query"]
        relatedContext = query_and_context["relatedContext"]
        prompt = assemble_prompt_old(prompt_template, user_query, relatedContext)
        task_batch.update(
            {
                mnemonic: {
                    "user_query": user_query,
                    "relatedContext": relatedContext,
                    "prompt": prompt,
                }
            }
        )
    return task_batch


def generate_rag_task_batch_new(
    query_and_context_batch: dict,
    prompt_template: str,
    interpreted_metadata_batch: dict,
) -> dict:
    task_batch = {}
    for mnemonic, kvPairs in query_and_context_batch.items():
        interpretation = interpreted_metadata_batch[mnemonic]["interpretation"]
        kvPairs.update({"Interpretation": interpretation})
        prompt = assemble_prompt(prompt_template, kvPairs)
        kvPairs.update({"prompt": prompt})
        task_batch.update({mnemonic: kvPairs})
    return task_batch


def run_rag_task_batch(rag_task_batch: dict) -> dict:
    for task in rag_task_batch.values():
        prompt = task["prompt"]
        # print(str(prompt))
        logger.info(f"LLM task prompt: {prompt}")
        # response = llm.chat_with_openai(prompt)
        # result = llm.result_extractor_old(response)
        response = llm.chat_with_llm(prompt)
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            logger.debug("Invalid JSON: ", response.text)
            raise json.JSONDecodeError
        result = llm.result_extractor_openai(response_data)
        result_list = result["content"].split(",")
        task["result_LLM"] = [s.strip() for s in result_list]
        # print(str(result["content"]) + "\n")
        logger.info(f"LLM task result: {result['content']}")
    return rag_task_batch


def run_rag_task_single(prompt: str, model: str, user_config: dict = None) -> dict:
    # prompt = assemble_prompt(prompt_template, kvPairs)
    response = llm.chat_with_llm(prompt, model, user_config)
    result = llm.result_extractor(response, model)
    logger.debug(f"prompt: {prompt}")
    logger.debug(f"model: {result["model"]}")
    logger.debug(f"content: {result["content"]}")
    logger.debug(f"prompt_tokens: {result["prompt_tokens"]}")
    logger.debug(f"completion_tokens: {result["completion_tokens"]}")
    return result


if __name__ == "__main__":
    pass
