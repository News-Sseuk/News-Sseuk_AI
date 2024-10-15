# 뉴스 기사 신뢰도 평가 모델 (BERT 기반)

## 프로젝트 개요

이 프로젝트는 뉴스 기사의 신뢰도를 평가하기 위해 두 가지 주요 지표를 추출하는 것을 목표로 합니다:

1. **제목-본문 일치도**: 기사의 제목과 본문 내용 간의 의미적 유사성을 평가합니다.
2. **문장 간 일관성**: 본문 내 문장 간의 의미적 연관성을 측정합니다.

이 구현은 BERT(Bidirectional Encoder Representations from Transformers) 모델을 사용하여 이러한 평가를 수행합니다.

## 사용 기술

- Python
- PyTorch
- Transformers (Hugging Face)
- NumPy
- SciPy
