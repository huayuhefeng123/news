import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent  # backend 目录
sys.path.append(str(backend_dir))

from flask import Flask
from flask_cors import CORS
import os
from app.extensions import db, migrate
from app.models.user import User

def create_app():
    app = Flask(__name__)
    
    # 基本配置
    app.config['UPLOAD_FOLDER'] = 'uploads'
    # 修改数据库路径为绝对路径，确保在项目根目录
    db_path = Path(__file__).resolve().parent.parent / 'database' / 'app.db'
    # 创建数据库目录
    db_path.parent.mkdir(parents=True, exist_ok=True)
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = 'your-secret-key'
    
    # 初始化扩展
    db.init_app(app)
    migrate.init_app(app, db)
    
    # CORS 配置
    CORS(app, 
         origins=["http://localhost:5173", "http://localhost:5174"],
         allow_headers=["Content-Type", "Authorization"],
         supports_credentials=True
    )
    
    # 创建上传目录
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # 注册蓝图
    from app.controllers.verification_controller import verification_bp
    from app.controllers.auth_controller import auth_bp
    
    app.register_blueprint(verification_bp, url_prefix='/api/verify')
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    
    # 确保数据库表存在并创建初始用户
    with app.app_context():
        db.create_all()
        # 检查是否存在管理员用户
        if not User.query.filter_by(username='admin').first():
            admin = User(
                username='admin',
                email='admin@example.com',
                role='admin'
            )
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("已创建管理员用户")
    
    return app

def main():
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main() 