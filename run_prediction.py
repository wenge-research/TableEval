# -*- encoding: utf-8 -*-
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
from tqdm import tqdm 
from typing import List, Dict, Any, Optional, Union
from openai_client import *
from utils import *


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)


CONTEXT_MAPPING = {
    "html": "context_html",
    "latex": "context_latex",
    "markdown": "context_markdown"
}


def filter_dataset(data: List[Dict[str, Any]], 
                   task_filter: Optional[Union[str, List[str]]] = None, 
                   id_filter: Optional[Union[str, List[str]]] = None) -> List[Dict[str, Any]]:
    """
    Filter dataset based on task types and/or IDs.
    """
    if task_filter and isinstance(task_filter, str):
        task_filter = [task_filter]
    if id_filter and isinstance(id_filter, str):
        id_filter = [id_filter]


    if task_filter and id_filter:
        logging.info(f"Predicting data in {task_filter} and ID in {id_filter}")
    elif id_filter:
        logging.info(f"Predicting ID in {id_filter}")
    elif task_filter:
        logging.info(f"Predicting data in {task_filter}")

        
    filtered =  [
        d for d in data
        if (not task_filter or d["task_name"] in task_filter) and 
           (not id_filter or d["id"] in id_filter)
    ]
    logging.info(f"Number of prediction data: {len(filtered)}")
    
    return filtered
    

def run_prediction_one(
    client: OpenAIClient,
    data: Dict[str, Any],
    context_type: str,
    system_message: str
) -> Dict[str, Any]:
    """
    Run prediction for a single data point.
    """
    
    try:
        # pick context
        context_key = CONTEXT_MAPPING.get(context_type, "context_markdown")
        context = data['context'][context_key]
        
        prediction_list = []
        messages = [{"role": "system", "content": system_message}]
        for question_idx, question in enumerate(data['question_list']):
            if question_idx == 0:
                user_input = data['instruction'].format(context=context, question=question)
                messages.append({"role": "user", "content": user_input})
            else:
                messages.append({"role": "user", "content": question})

            model_response = client.generate_response(messages)
            messages.append({"role": "assistant", "content": model_response})
            prediction_list.append(model_response)

        return {"id": data["id"], 
                "task_name": data["task_name"],
                "sub_task_name": data["sub_task_name"],
                "golden_answer_list": data["golden_answer_list"], 
                "prediction_list": prediction_list}
    
    except Exception as e:
        # 记录错误日志（包含 ID 和错误信息）
        logging.error(f"❌ Failed | ID: {data['id']} | Error type: {type(e).__name__} | Details: {str(e)}")
        return {"id": data["id"], "prediction_list": [], "error": str(e)}



def run_prediction(
    model_name: str,
    test_data_path: str,
    output_dir: str,
    config_file: str,
    reasoning_type: str = "cot",
    context_type: Optional[str] = None,
    max_workers: int = 10,
    specific_tasks: Optional[List[str]] = None,
    specific_ids: Optional[List[str]] = None,
) -> None:
    """
    Run batch prediction pipeline with configurable parameters.
    """
    start_time = time.time()


    logging.info("Prediction process started.")
    

    # Load dataset

    test_data = load_json(test_data_path)

    # Filter dataset
    filtered_data = filter_dataset(test_data, task_filter=specific_tasks, id_filter=specific_ids)

    # Build output path
    prediction_file_path = generate_prediction_path(output_dir, model_name, reasoning_type, context_type, specific_tasks, specific_ids) 

    system_message = load_yaml_safe("config/prompts.yaml")["system_messages"].get(reasoning_type, "")
    
    results = []
    
    # Instantiate client
    client = OpenAIClient(model_name, config_file)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for item in filtered_data:
            future = executor.submit(run_prediction_one, client, item, context_type, system_message)
            futures.append(future)

        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Predicting"), 1):
            try:
                results.append(future.result())
            except Exception as e:
                logging.error(f"Thread failed: {str(e)}")

            if i % 50 == 0 and i > 0:
                save_json(results, prediction_file_path)

    
    
    # Save results
    if results:
        results = sorted(results, key=lambda x: int(x["id"]))
    save_json(results, prediction_file_path, "prediction results")

    elapsed_time = time.time() - start_time
    logging.info(f"Total Time Elapsed: {elapsed_time:.2f}s | Total number of data: {len(filtered_data)} | Finish processing: {len(results)}")

    logging.info("Prediction process ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model prediction")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="Name of the model to use for inference")
    parser.add_argument("--test_data_filepath", type=str, default="./data/TableEval-test.jsonl", help="File path containing evaluation dataset in JSONL format")
    parser.add_argument("--config_file", type=str, default="./config/api_config.yaml", help="YAML configuration file for API authentication parameters")
    parser.add_argument("--prompt_file", type=str, default="./config/prompts.yaml", help="YAML file containing prompt templates for evaluation tasks")
    parser.add_argument("--output_directory", type=str, default="./outputs", help="Directory path to store prediction and evaluation results")
    parser.add_argument("--max_workers", type=int, default=5, help="Maximum number of parallel worker processes")
    parser.add_argument("--context_type", type=str, default="markdown", help="Table formatting syntax for model input (markdown, html, latex | Default markdown)")
    parser.add_argument("--specific_tasks", nargs="+", type=str, default=None, help="Filter evaluation tasks by name (Default None=all tasks)")
    parser.add_argument("--specific_ids", nargs="+", type=str, default=None, help="Filter dataset samples by ID (Default None=all ids)")
    args = parser.parse_args()
    
    run_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_directory = os.path.join(args.output_directory, run_time)
    os.makedirs(args.output_directory, exist_ok=True)
    setup_logging(output_directory)
    prediction_dir = os.path.join(args.output_directory, run_time, "prediction")
    os.makedirs(prediction_dir)

    run_prediction(
        model_name=args.model_name, 
        test_data_path=args.test_data_filepath, 
        output_dir=prediction_dir, 
        config_file=args.config_file, 
        reasoning_type="cot",  
        context_type=args.context_type, 
        max_workers=args.max_workers, 
        specific_tasks=args.specific_tasks, 
        specific_ids=args.specific_ids
    )
