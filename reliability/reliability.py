from flask import request, jsonify
from flask_restx import Resource, Api, Namespace
from db_connection import Mongo, Mysql
from bson.objectid import ObjectId
from reliability.score import calculate_news_reliability

reliability = Namespace('Reliability')

@reliability.route('')
class save_reliability_score(Resource):
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

            update_query = "UPDATE article SET reliability = %s WHERE id = %s"
            cursor.execute(update_query, (int(score), row[0],))
            db_connection.commit()  # 이걸 해야 적용됨.

        cursor.close()
        Mysql.close_mysql_connection(self, cursor, db_connection)