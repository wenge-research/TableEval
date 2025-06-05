# -*- encoding: utf-8 -*-
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json_repair
import logging
import json
import numpy as np
import os
import time
from tqdm import tqdm
from openai_client import *
from utils import *


# Set the working directory to the current file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)


def extract_json(json_str: str) -> Optional[Any]:
    """
    Try to repair and parse a JSON string using json_repair.
    """
    try:
        return json_repair.loads(json_str)
    except Exception as e:
        logging.error(f"JSON repair error: {str(e)}")
        return None


def calculate_f1_sub_question(llm_judgement_dict: Dict[str, Any]) -> float:
    """
    Calculate the average F1 score for a sub question.
    """
    f1_scores = []
    for question_one in llm_judgement_dict.get("问题列表", []):
        if not question_one.get("是否正确", []):
            return 0.0

        # Calculate true positives (tp) if '是否正确' is a list
        is_true = question_one.get("是否正确")
        tp = sum(is_true) if isinstance(is_true, list) else 0
        golden_answer = question_one.get("参考答案")

        # Calculate precision: ratio of true positives to total predicted positives
        if not is_true:
            precision = 0
        elif not golden_answer:
            recall = 0
        else:
            precision = min(tp / len(is_true), 1.0) if isinstance(is_true, list) else 0
            # Calculate recall: ratio of true positives to total golden answers
            recall = min(tp / len(golden_answer), 1.0) if isinstance(golden_answer, list) else 0

        # Compute F1 score if precision and recall are positive
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1_score)

    f1_score_average = np.mean(f1_scores)
    return f1_score_average


def calculate_f1(llm_judgement_list: List[Dict[str, Any]]) -> Optional[float]:
    """
    Calculate the overall average F1 score from a question.
    """
    if not llm_judgement_list:
        return None
    f1_scores = []
    valid_judgements = [j for j in llm_judgement_list if isinstance(j, dict)]
    if not valid_judgements:
        return None
    for llm_judgement_dict in valid_judgements:
        f1_score = calculate_f1_sub_question(llm_judgement_dict)
        f1_scores.append(f1_score)
    f1_score_average = np.mean(f1_scores)
    return round(f1_score_average * 100, 2)


def get_model_answer(
    client: Any,
    model_prediction: str,
    evaluation_prompt: str,
    golden_answer_str: str
) -> Optional[Dict[str, Any]]:
    """
    Call the language model to get an evaluation dict of the model's prediction.
    """
    model_prediction_formatted = "### 大模型的回答\n{answer}".format(answer=model_prediction)
    evaluation_prompt = evaluation_prompt.format(answer=golden_answer_str)
    messages = [
        {"role": "system", "content": evaluation_prompt},
        {"role": "user", "content": model_prediction_formatted}
    ]

    # Generate response from the model
    model_response = client.generate_response(messages)

    # Extract JSON structure from the model's response
    llm_judgement_dict = extract_json(model_response)
    return llm_judgement_dict


def run_llm_evaluation_one(
    data: Dict[str, Any],
    client: Any,
    evaluation_prompt: str
) -> Dict[str, Any]:
    """
    Evaluate a single test data item, obtain LLM judgements for each prediction and calculate the F1 score.
    """
    try:
        # Return data immediately if already evaluated
        if data.get("llm_judgement_list") and data.get("f1_score"):
            return data

        # If no judgement list exists, compute it using the golden answers and predictions
        if not data.get("llm_judgement_list", []):
            golden_answer_list = data.get("golden_answer_list", [])
            prediction_list = data.get("prediction_list", [])

            llm_judgement_list: List[Any] = []
            # Iterate over each prediction and its corresponding golden answer
            if len(prediction_list) != len(golden_answer_list):
                logging.error(f"Length mismatch between prediction_list and golden_answer_list for data id {data['data']}")
                return data
            for model_prediction, golden_answer in zip(prediction_list, golden_answer_list):
                golden_answer_str = json.dumps(golden_answer, ensure_ascii=False, indent=2)
                llm_judgement_dict = get_model_answer(
                    client,
                    model_prediction,
                    evaluation_prompt,
                    golden_answer_str,
                )
                llm_judgement_list.append(llm_judgement_dict)

            data["llm_judgement_list"] = llm_judgement_list

        # Calculate F1 score from the LLM judgements
        if data.get("f1_score") is None:
            f1_score = calculate_f1(data["llm_judgement_list"])
            data["f1_score"] = f1_score
    except Exception as err:
        logging.error(f"Error evaluating data: {data['id']}")
    
    return data


def get_score_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate overall f1-score and f1-score for each task.
    """
    failed_ids = []
    sub_task_score_sum = defaultdict(int)
    sub_task_score_count = defaultdict(int)
    task_to_sub_tasks = defaultdict(set)

    # Process each entry to accumulate relevant information
    for item in results:
        if item.get("f1_score") is not None:
            sub_task = item["sub_task_name"]
            task = item["task_name"]

            sub_task_score_sum[sub_task] += item["f1_score"]
            sub_task_score_count[sub_task] += 1

            task_to_sub_tasks[task].add(sub_task)
        else:
            failed_ids.append(item["id"])

    f1_score_sub_task = {
        sub_task: sub_task_score_sum[sub_task] / sub_task_score_count[sub_task]
        for sub_task in sub_task_score_sum
    }

    f1_score_task = {
        task: round(np.mean([f1_score_sub_task[sub_task] for sub_task in task_to_sub_tasks[task]]), 2)
        for task in task_to_sub_tasks if task_to_sub_tasks[task]
    }

    f1_score_overall = round(np.mean(list(f1_score_task.values())), 2) if f1_score_task else 0


    f1_score_sub_task = {
        sub_task: round(f1_score_sub_task[sub_task],2) for sub_task in f1_score_sub_task
    }

    f1_score_task = {
        task: round(f1_score_task[task],2) for task in f1_score_task
    }

    return {
        "f1_score": f1_score_overall,  
        "f1_score_task": f1_score_task, 
        "f1_score_sub_task": f1_score_sub_task, 
        "failed_ids": failed_ids,  
    }
    
    
def split_path(path: str):
    """
    Given a file path, return its parent directory and the file name
    """
    abs_path = os.path.abspath(path)
    dirname = os.path.dirname(abs_path)
    filename = os.path.basename(abs_path)
    parent_of_dirname = os.path.dirname(dirname)
    return parent_of_dirname, filename


def run_llm_evaluation(
    llm_judge_name: str,
    prediction_file_path: str,
    config_file: str,
    prompt_file=str, 
    max_workers: int = 10
) -> None:
    """
    Run the LLM evaluation on a list of predictions.
    Loads prediction data from a JSON file, evaluates each item in parallel,
    computes the overall F1 score, and saves detailed and summary results.
    """

    start_time = time.time()
    output_dir, prediction_file_name = split_path(prediction_file_path)
    
    setup_logging(output_dir)
    logging.info("Evaluation process started.")
   
    prediction_data = load_json(prediction_file_path)

    if not prediction_data:
        return

    # Create output file paths based on the prediction file name and judge name
    
    evaluation_file_name = prediction_file_name[:-5] + f"_eval_by_{llm_judge_name}.json"
    score_file_name = prediction_file_name[:-5] + f"_eval_by_{llm_judge_name}_score.json"
    evaluation_dir = os.path.join(output_dir, "evaluation")
    score_dir = os.path.join(output_dir, "score")
    evaluation_file_path = os.path.join(evaluation_dir, evaluation_file_name)
    score_file_path = os.path.join(score_dir, score_file_name)
    os.makedirs(evaluation_dir, exist_ok=True)
    os.makedirs(score_dir, exist_ok=True)
    evaluation_prompt = load_yaml_safe(prompt_file)["evaluation_prompt"]["prompt"]

    client = OpenAIClient(llm_judge_name, config_file)

    results = []

    # Use ThreadPoolExecutor to run evaluations in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for item in prediction_data:
            future = executor.submit(run_llm_evaluation_one, item, client, evaluation_prompt)
            futures.append(future)

        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Evaluating"), 1):
            try:
                results.append(future.result())
            except Exception as e:
                logging.error(f"Thread failed: {str(e)}")
            if i % 50 == 0 and i > 0:
                save_json(results, evaluation_file_path)
    # Save results
    if results:
        results = sorted(results, key=lambda x: int(x["id"]))

    save_json(results, evaluation_file_path, "evaluation results")

    # F1 score summary
    score_summary = get_score_summary(results)
    save_json(score_summary, score_file_path, "evaluation score")
    elapsed_time = time.time() - start_time
    
    logging.info(f"Total Time Elapsed: {elapsed_time:.2f}s | Total number of predictions: {len(prediction_data)} | Finish processing: {len(results)}")
    logging.info("Evaluation process ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument("--llm_judge_name", type=str, default="gpt-4o-2024-11-20", help="Name of the LLM judge to use for evaluation")
    parser.add_argument("--prediction_file_path", type=str, help="File path containing prediction result in JSON format")
    parser.add_argument("--config_file", type=str, default="./config/api_config.yaml", help="YAML configuration file for API authentication parameters")
    parser.add_argument("--prompt_file", type=str,  default="./config/prompts.yaml", help="YAML file containing prompt templates for evaluation tasks")
    parser.add_argument("--max_workers", type=int, default=5, help="Maximum number of parallel worker processes")
    args = parser.parse_args()

    run_llm_evaluation(
        llm_judge_name = args.llm_judge_name,
        prediction_file_path = args.prediction_file_path,
        config_file=args.config_file, 
        prompt_file=args.prompt_file, 
        max_workers=args.max_workers
    )
