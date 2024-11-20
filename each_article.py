from flask import request, jsonify
from flask_restx import Resource, Api, Namespace
from db_connection import Mongo, Mysql
from bson.objectid import ObjectId
from reliability.score import calculate_news_reliability
from summarization.summary import summarize_article
from summarization.issue import extract_issue_article
from recommendation.keyword import extract_keywords
from datetime import datetime, timedelta
from recommendation.recommend import recommend_similar_articles, recommend_articles_based_on_keywords
import pandas as pd

article = Namespace('Article')

@article.route('/detail')
class save_each_article_detail(Resource):
    def post(self):
        db_connection = Mysql.get_mysql_connection(self)
        cursor = db_connection.cursor()
        article_id = request.args.get("id")
        # GET
        query = 'SELECT a.id, a.nosql_id FROM article a WHERE a.id= %s'
        cursor.execute(query, (article_id,))
        # return
        db_object = Mongo.get_mongo_connection(self)
        for row in cursor.fetchall():
            article = db_object.find_one({'_id': ObjectId(row[1])}, {"title": True, "content": True})

            score = calculate_news_reliability(article["title"], article["content"])
            keyword_list = extract_keywords(article["content"])
            summarized_text = summarize_article(article["content"])
            issue_text = extract_issue_article(article["content"])

            update_query = "UPDATE article SET reliability = %s, summary = %s, issue = %s WHERE id = %s"
            # 1. 키워드가 존재하지 않으면 HashTag 테이블에 추가
            insert_query = ("""INSERT IGNORE INTO hash_tag (name) VALUES {}"""
                            .format(", ".join(f"('{keyword}')" for keyword in keyword_list)))
            # 2. ArticleHashTag에 매핑을 추가하기 위한 SQL
            map_sql = ("""
            INSERT INTO article_hash_tag (article_id, hashtag_id, created_time)
            SELECT %s, id, NOW() FROM hash_tag WHERE name IN ({})"""
                       .format(", ".join(f"'{keyword}'" for keyword in keyword_list)))

            cursor.execute(update_query, (int(score), summarized_text, issue_text, row[0],))
            cursor.execute(insert_query)
            cursor.execute(map_sql, (row[0],))

            db_connection.commit()

        cursor.close()
        Mysql.close_mysql_connection(self, cursor, db_connection)

class today_articles:
    def get_dataframe(self):
        db_object = Mongo.get_mongo_connection(self)

        # 오늘 날짜와 7일 전 날짜 계산
        end_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=7)

        # MongoDB에서 날짜 필터링하여 관련 기사 검색
        relevant_articles = db_object.find(
            {
                "publishedDate": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            },
            {"_id": True, "content": True}  # 필요한 필드만 반환
        )

        # 결과를 DataFrame 형태로 변환
        articles_data = [
            {"article_id": str(article["_id"]), "content": article["content"]}
            for article in relevant_articles
        ]
        articles_df = pd.DataFrame(articles_data)
        return articles_df


@article.route('/each')
class recommend_article_by_issue(Resource):
    def get(self):
        db_connection = Mysql.get_mysql_connection(self)
        cursor = db_connection.cursor()
        article_id = request.args.get("nosql_id")

        query = 'SELECT a.issue FROM article a WHERE a.nosql_id= %s'
        cursor.execute(query, (article_id,))
        row = cursor.fetchall()

        articles_df = today_articles.get_dataframe(self)

        cursor.close()
        Mysql.close_mysql_connection(self, cursor, db_connection)
        return recommend_similar_articles(row[0][0], articles_df)  # List[str] 형태로 기사의 nosql_id 리스트 반환


@article.route('/personalizing')
class recommend_article_by_userhistory(Resource):
    def post(self):
        data = request.get_json()
        keyword_list = data["keywords_list"]
        articles_df = today_articles.get_dataframe(self)
        return recommend_articles_based_on_keywords(keyword_list, articles_df)