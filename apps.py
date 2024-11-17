from flask import Flask  # 서버 구현을 위한 Flask 객체 import
from flask_restx import Api, Resource  # Api 구현을 위한 Api 객체 import
from config import Mongo_Config, Mysql_Config
from each_article import article

app = Flask(__name__)  # Flask 객체 선언, 파라미터로 어플리케이션 패키지의 이름을 넣어줌.
app.config.from_object(Mongo_Config)
app.config.from_object(Mysql_Config)
api = Api(app)  # Flask 객체에 Api 객체 등록

api.add_namespace(article, '/article')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)