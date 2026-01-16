import json
import glob
import os
from Reranker import Reranker
from convert_to_json import convert_to_json

TEST_SET_DIR = "./data/test_dataset/"
WIKI_DATA_PATH = './data/wikipedia_documents.json'

def data_load(DIR):
    dataset = load_from_disk(DIR)
    return dataset


def simple_json_compress():
    # search json
    json_files = glob.glob('./test_*.json')
    
    
    
    # rag json load
    file_number = len(json_files)
    all_data = []
    for i in range(file_number):
        with open(json_files[i], 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.append(data)
    # compression

    result = []
    total_rows = len(all_data[0]['document_id'])
    for i in range(total_rows):
        union_set = set()
        for data in all_data:
            union_set |= set(data['document_id'][i])
        result.append(list(union_set))
    json_dic = {'question_id':all_data[0]['question_id'], 'document_id':result}

    file_path = './test_union_simple.json'
    with open(file_path, 'w') as f:
        json.dump(json_dic, f)


    return json_dic

def top_k_json_compress(top_k):

    # search json
    json_files = glob.glob('./test_*.json')
    
    # data load
    test_dataset = data_load(TEST_SET_DIR)
    with open(WIKI_DATA_PATH) as f:
        wiki_data = json.load(f)

    # rag json load
    file_number = len(json_files)
    all_data = []
    for i in range(file_number):
        with open(json_files[i], 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.append(data)

    # compression
    result = []
    total_rows = len(all_data[0]['document_id'])
    for i in range(total_rows):
        union_set = set()
        for data in all_data:
            union_set |= set(data['document_id'][i])
        result.append(list(union_set))
    json_dic = {'question_id':all_data[0]['question_id'], 'document_id':result}

    # reranking
    reranked_result = []
    for i in tqdm.tqdm(range(600)):
        query = test_dataset['validation'][i]['question']
        q_id = test_dataset['validation'][i]['id']
        ids_for_rerank_validation = json_dic['document_id'][i]
        docs_for_rerank_validation = [wiki_data['%d'%k]['text'] for k in ids_for_rerank_validation]
        reranked_results_validation = reranker.rerank(query, docs_for_rerank_validation, ids_for_rerank_validation, top_k=top_k)
        result = list(dict.fromkeys((list(map(int, ((list(np.array(reranked_results_validation[1])[:,0].astype(int)))))))))[:10]
        reranked_result.append([q_id, result])
    # converting
    json_test = convert_to_json(reranked_result)

    # save json
    file_path = './test_union_top%d.json'%top_k
    with open(file_path, 'w') as f:
        json.dump(json_test, f)


    return json_test
