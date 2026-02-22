import json
import pandas as pd
from datasets import load_from_disk
from llama_cpp import Llama


def llm_inference(
        path_nbest_predictions: str,
        path_testset: str,
        path_wikidump: str,
        path_ragdata: str,
        path_system_msg: str,
        path_response: str
        ):

    with open(path_nbest_predictions, "r") as f:
        nbest_predictions = json.load(f)
    list_best_predictions = []
    for idx_problem, key in enumerate(nbest_predictions):
        list_best_predictions.append((nbest_predictions[key][0]['probability'], idx_problem))
    list_best_predictions.sort()
    list_low_qa_confidence = []
    for idx_problem in range(len(list_best_predictions)//5):
        list_low_qa_confidence.append(list_best_predictions[idx_problem][1])
    
    with open(path_wikidump, "r") as f:
        wikidump = json.load(f)
    
    with open(path_ragdata, "r") as f:
        tmp = json.load(f)
    ragdata = {
        'idx_problem': tmp['question_id'],
        'idx_document': tmp['document_id']
        }
    df_response = pd.DataFrame(ragdata)
    df_response.to_csv(path_response, index=False)

    testset = load_from_disk(path_testset, split='validation')


    llm = Llama.from_pretrained(
        repo_id="unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
        filename="Qwen3-30B-A3B-Instruct-2507-UD-Q6_K_XL.gguf",
        n_ctx=32768,
        n_gpu_layers=-1,
        flash_attn=True,
        verbose=False
    )

    with open(path_system_msg, 'r', encoding='utf-8') as f:
        system_msg = f.read()

    for idx_problem in list_low_qa_confidence:
        user_content = ""
        list_idx_document = ragdata['document_id'][idx_problem]
        for idx_document in range(len(list_idx_document)):
            user_content += f"텍스트 {idx_document}\n{wikidump[str(idx_document)]['text']}\n\n"
        user_content += f"질문\n{testset[idx_problem]['question']}"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content}
        ]
        out = llm.create_chat_completion(
            messages=messages,
            max_tokens=8192,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0
        )

        df_response.loc[idx_problem, f'input'] = user_content
        df_response.loc[idx_problem, f'output'] = out['choices'][0]['message']['content']
        df_response.to_csv(path_response, index=False)

    return None