from flask import request, jsonify
from flask_restx import Resource, Api, Namespace
from db_connection import Mongo, Mysql
from bson.objectid import ObjectId
from reliability.score import calculate_news_reliability
from summarization.summary import summarize_article
from recommendation.keyword import extract_keywords

article = Namespace('Article')

@article.route('')
class save_each_article_detail(Resource):
    def post(self):
        db_connection = Mysql.get_mysql_connection(self)
        cursor = db_connection.cursor()
        # GET
        query = 'SELECT a.id, a.nosql_id FROM article a WHERE a.reliability IS NULL'
        cursor.execute(query)
        # return
        db_object = Mongo.get_mongo_connection(self)
        for row in cursor.fetchall():
            article = db_object.find_one({'_id': ObjectId(row[1])}, {"title": True, "content": True})

            score = calculate_news_reliability(article["title"], article["content"])
            keyword_list = extract_keywords(article["content"])
            summarized_text = summarize_article(article["content"])

            update_query = "UPDATE article SET reliability = %s, summary = %s WHERE id = %s"
            # 1. 키워드가 존재하지 않으면 HashTag 테이블에 추가
            insert_query = ("""INSERT IGNORE INTO Hash_Tag (name) VALUES {}"""
                            .format(", ".join(f"('{keyword}')" for keyword in keyword_list)))
            # 2. ArticleHashTag에 매핑을 추가하기 위한 SQL
            map_sql = ("""
            INSERT INTO Article_Hash_Tag (article_id, hashtag_id)
            SELECT %s, id FROM Hash_Tag WHERE name IN ({})"""
                       .format(", ".join(f"'{keyword}'" for keyword in keyword_list)))

            cursor.execute(update_query, (int(score), summarized_text, row[0],))
            cursor.execute(insert_query)
            cursor.execute(map_sql, (row[0],))

            db_connection.commit()  # 이걸 해야 적용됨.

        cursor.close()
        Mysql.close_mysql_connection(self, cursor, db_connection)