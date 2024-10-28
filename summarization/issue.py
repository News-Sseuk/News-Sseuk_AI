from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
nltk.download('punkt_tab')

# 모델과 토크나이저 경로
save_directory = "summarization/t5-korean-issue-model-v1"
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = AutoModelForSeq2SeqLM.from_pretrained(save_directory)

# 이슈 추출 함수 정의
def extract_issue_article(text, max_input_length=512, min_length=10, max_length=100, num_beams=8):
    # 입력 텍스트를 모델의 입력 형식으로 변환
    inputs = tokenizer("summarize: " + text, max_length=max_input_length, truncation=True, return_tensors="pt")
    
    # 요약 생성
    output = model.generate(**inputs, num_beams=num_beams, do_sample=True, min_length=min_length, max_length=max_length)
    
    # 디코딩하고 첫 번째 문장 선택
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    issue = nltk.sent_tokenize(decoded_output.strip())[0]
    
    return issue
