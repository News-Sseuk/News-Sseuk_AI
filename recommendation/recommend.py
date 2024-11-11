import nltk
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
from keyword import extract_keywords, preprocess_text

# 이슈 기반 유사 기사 추천
def recommend_similar_articles(issue_text: str, articles_df: pd.DataFrame) -> List[str]:
    """
    Parameters:
        issue_text (str): 이슈 텍스트
        articles_df (pd.DataFrame): 'article_id' 및 'content' 열이 포함된 기사 데이터프레임

    Returns:
        List[str]: 유사한 기사 ID 목록 (최대 5개)
    """
    # 이슈 텍스트에서 키워드 추출
    issue_keywords = extract_keywords(issue_text, top_n=3)
    issue_keywords_str = " ".join(issue_keywords)

    # 기사 본문 TF-IDF 적용
    tfidf = TfidfVectorizer()
    article_tfidf = tfidf.fit_transform(articles_df['content'].apply(preprocess_text))
    
    # 이슈 텍스트 TF-IDF 적용
    issue_tfidf = tfidf.transform([issue_keywords_str])

    # 코사인 유사도 계산
    similarities = cosine_similarity(issue_tfidf, article_tfidf).flatten()
    
    # 유사도가 높은 상위 5개 기사 선택
    similar_articles_idx = similarities.argsort()[::-1][:5]
    similar_article_ids = articles_df.iloc[similar_articles_idx]['article_id'].tolist()
    
    return similar_article_ids


# 사용자 관심 키워드 기반 추천
def recommend_articles_based_on_keywords(keywords: List[str], articles_df: pd.DataFrame) -> List[str]:
    """
    Parameters:
        keywords (List[str]): 사용자 관심 키워드 목록 (3개)
        articles_df (pd.DataFrame): 'article_id' 및 'content' 열이 포함된 기사 데이터프레임

    Returns:
        List[str]: 유사한 기사 ID 목록 (최대 5개)
    """
    # 키워드 문자열 생성
    keywords_str = " ".join(keywords)
    
    # 기사 본문 TF-IDF 적용
    tfidf = TfidfVectorizer()
    article_tfidf = tfidf.fit_transform(articles_df['content'].apply(preprocess_text))
    
    # 키워드 TF-IDF 적용
    keywords_tfidf = tfidf.transform([keywords_str])

    # 코사인 유사도 계산
    similarities = cosine_similarity(keywords_tfidf, article_tfidf).flatten()
    
    # 유사도가 높은 상위 5개 기사 선택
    relevant_articles_idx = similarities.argsort()[::-1][:5]
    relevant_article_ids = articles_df.iloc[relevant_articles_idx]['article_id'].tolist()
    
    return relevant_article_ids
