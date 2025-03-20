from app.main import create_app
from app.extensions import db
from app.models.user import User

def init_db():
    app = create_app()
    with app.app_context():
        # 清空数据库
        db.drop_all()
        db.create_all()
        
        # 创建管理员用户
        admin = User(
            username='admin',
            email='admin@example.com',
            role='admin'
        )
        admin.set_password('admin123')  # 设置初始密码
        
        # 添加到数据库
        db.session.add(admin)
        db.session.commit()
        
        print("数据库初始化完成！")
        print("管理员账号：admin")
        print("管理员密码：admin123")

if __name__ == '__main__':
    init_db() 