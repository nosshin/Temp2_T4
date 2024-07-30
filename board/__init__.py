from flask import Flask
from flask_restx import Api
from flask_session import Session

from board.pages import bp as pages_bp
from board.video import bp_video

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'supersecretkey'
    app.config['SESSION_TYPE'] = 'filesystem'

    app.register_blueprint(pages_bp)
    app.register_blueprint(bp_video)

    # Flask-RestX API 객체 초기화
    api = Api(
        app,
        version='1.0',
        title='Lucky Vicky Geti Viti API',
        description='APIs for Lucky Vicky Geti Viti operations',
        doc='/api-docs'  # API 문서 경로 설정
    )

    Session(app)
    
    return app

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
    #app.run(ssl_context=('board/cert.pem', 'board/key.pem'), host='0.0.0.0', port=8000)
