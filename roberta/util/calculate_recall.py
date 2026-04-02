from datasets import load_from_disk
import json

train_set_dir = "/data/ephemeral/home/data/train_dataset/"
dataset = load_from_disk(train_set_dir)
train_set = dataset['train']

# result: dense_only, dense_ner 등의 RAG 결과 json
def calculate(result, file_path):
    
    file_path = "/data/ephemeral/home/pro-nlp-mrc-nlp-05/document/train.json"

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    question_ids = data["question_id"]
    document_ids = data["document_id"]

    result = [
        {"question_id": qid, "document_ids": doc_ids}
        for qid, doc_ids in zip(question_ids, document_ids)
    ]
    
    # 정답 문서를 가져온 문제 인덱스
    correct = {'yes': [], 'no': []}
    # top 5 중, 중복된게 있는지
    dup = {'yes': [], 'no': []}
    total_len = len(train_set)
    
    for i, (train, res) in enumerate(zip(train_set, result)):
        # 같은 문제에 대해 체크하는게 맞는지 확인!
        if train['id'] != res['question_id']:
            print("ID가 맞지 않습니다.")
            break
        
        train_doc_id = train['document_id']
        rag_doc_ids = res['document_ids']  # 리스트
        
        is_correct = train_doc_id in rag_doc_ids
        
        if is_correct:
            correct['yes'].append(i)
        else:
            correct['no'].append(i)
            
        rag_ids_set = set(rag_doc_ids)
        is_dup = len(rag_ids_set) != len(rag_doc_ids)
        
        if is_dup:
            if is_correct:
                dup['yes'].append(i)
            else:
                dup['no'].append(i)
    
    print(f"top 5 내에 정답 문서가 있는 개수: {len(correct['yes'])}, 비율: {len(correct['yes'])/ total_len * 100 }")
    print(f"문서를 중복해서 가져온 개수: {len(dup['no']) + len(dup['yes'])}, 비율: {(len(dup['no'])+len(dup['yes'])) / total_len * 100 }")
    print(f"중복된 문서를 가져온 것 중 정답이 있는 개수: {len(dup['yes'])}")
    
    return correct
    