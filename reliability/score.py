import numpy as np
from scipy.spatial.distance import cosine
import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-multilingual-uncased')
model = torch.load('checkpoints/bert_multilingual.pt')  # 전체 모델을 통째로 불러옴
model.load_state_dict(torch.load('checkpoints/50k_ep50_241011.pt'))  # state_dict를 불러 온 후, 모델에 저장
model.to("cpu")

# 문장을 BERT 임베딩으로 변환하는 함수
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model.bert(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# 제목과 본문 간 일치도를 낚시성 분류로 평가하는 함수
def calculate_title_body_similarity(title, body):
    inputs = tokenizer.encode_plus(
        title + " " + body, 
        add_special_tokens=True, 
        return_tensors='pt', 
        truncation=True, 
        padding=True, 
        max_length=512
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    
    similarity_score = 1 if prediction == 1 else 0
    print(f"Title-Body Similarity: {similarity_score}")
    return similarity_score

# 문장 간 유사성 평가 함수
def evaluate_sentence_coherence(sentences):
    embeddings = [get_bert_embedding(sentence) for sentence in sentences]
    num_sentences = len(embeddings)
    
    # 유사성 행렬 계산
    similarity_matrix = np.zeros((num_sentences, num_sentences))

    for i in range(num_sentences):
        for j in range(num_sentences):
            if i != j:
                similarity_matrix[i, j] = 1 - cosine(embeddings[i], embeddings[j])
            else:
                similarity_matrix[i, j] = 1  # 자기 자신과의 유사도는 1

    sentence_similarity_scores = np.mean(similarity_matrix[np.triu_indices(len(sentences), k=1)])  # 상삼각 행렬의 유사도 값만 사용
    
    return sentence_similarity_scores

# 뉴스 신뢰도 계산 함수
def calculate_news_reliability(title, body):
    # 본문을 문장 단위로 분리
    sentences = body.split('. ')
    
    # 1. 제목과 본문 일치도 계산 
    title_body_similarity = calculate_title_body_similarity(title, body)
    
    # 2. 본문 내 문장 간 연관도 계산
    sentence_coherence = evaluate_sentence_coherence(sentences)
    
    # 신뢰도 점수 계산 (가중치를 설정, 필요에 따라 조정 가능)
    title_weight = 0.5  # 제목과 본문의 일치도 가중치
    body_weight = 0.5  # 본문 내 문장 간 연관도 가중치
    
    reliability_score = (title_body_similarity * title_weight) + (sentence_coherence * body_weight)
    
    # 0~100 사이로 변환
    reliability_score = reliability_score * 100
    return reliability_score

##### 사용 예시 #####
# title = '기사 제목'
# body = '기사 본문'
# reliability_score = calculate_news_reliability(title, body)
# print(f"News Reliability Score: {reliability_score:.2f}")
