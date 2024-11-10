import numpy as np
from scipy.spatial.distance import cosine
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
import nltk
from summarization.summary import summarize_article

tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-multilingual-uncased')
model = torch.load('reliability/checkpoints/bert_multilingual.pt', map_location=torch.device('cpu'))  # 전체 모델을 통째로 불러옴
model.load_state_dict(torch.load('reliability/checkpoints/50k_ep50_241011.pt', map_location=torch.device('cpu')))  # state_dict를 불러 온 후, 모델에 저장

# 문장을 BERT 임베딩으로 변환
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model.bert(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# 제목과 본문 간 일치도 계산
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
        # prediction = torch.argmax(outputs.logits, dim=1).item()
        probabilities = F.softmax(outputs.logits, dim=1)
        similarity_score = probabilities[0, 1].item() * 100
    
    return similarity_score

# 요약문 기반 문장 간 유사성 평가
def evaluate_sentence_coherence(sentences):    
    summary_text = summarize_article(text)
    summary_embedding = get_bert_embedding(summary_text)
    
    sentences = nltk.sent_tokenize(text)
    sentence_embeddings = [get_bert_embedding(sentence) for sentence in sentences]

    similarities = [1 - cosine(summary_embedding, sentence_emb) for sentence_emb in sentence_embeddings]
    coherence_score = np.mean(similarities) * 100
    return coherence_score
    
# 뉴스 신뢰도 계산
def calculate_news_reliability(title, body):
    sentences = body.split('. ')
    
    # 1. 제목과 본문 일치도 계산 
    title_body_similarity = calculate_title_body_similarity(title, body)
    
    # 2. 본문 내 문장 간 연관도 계산
    sentence_coherence = evaluate_sentence_coherence(sentences)
    
    # 신뢰도 점수 가중치 설정
    title_weight = 0.7  # 제목과 본문의 일치도 가중치
    body_weight = 0.3  # 본문 내 문장 간 연관도 가중치
    
    reliability_score = (title_body_similarity * title_weight) + (sentence_coherence * body_weight)
    return reliability_score

##### 사용 예시 #####
# title = '기사 제목'
# body = '기사 본문'
# reliability_score = calculate_news_reliability(title, body)
# print(f"News Reliability Score: {reliability_score:.2f}")
