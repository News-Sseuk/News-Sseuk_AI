import mysql.connector
from config import Mysql_Config, Mongo_Config
from pymongo import MongoClient
import certifi

class Mysql:
    def get_mysql_connection(self):
        return mysql.connector.connect(
            host=Mysql_Config.host,
            user=Mysql_Config.user,
            password=Mysql_Config.password,
            database=Mysql_Config.database
        )
    def close_mysql_connection(self, cursor, db_connection):
        return cursor.close() ; db_connection.close()

class Mongo:
    def get_mongo_connection(self):
        client = MongoClient(Mongo_Config.mongo_connect, tlsCAFile=certifi.where())
        connection = client["ArticleDB"]
        collection = connection.Articles
        return collection
