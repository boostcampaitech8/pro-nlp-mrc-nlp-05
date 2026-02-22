def convert_to_json(data):
    question_id = []
    document_list = []

    for q_id, doc_list in data:
        question_id.append(q_id)
        document_list.append(list(map(int, (doc_list))))
    result_dict = {
        "question_id": question_id,
        "document_id": document_list
    }
    return result_dict


