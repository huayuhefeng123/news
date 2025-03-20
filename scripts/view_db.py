from app.main import create_app
from app.models.user import User
from app.extensions import db

def view_db():
    app = create_app()
    with app.app_context():
        print("\n=== 数据库中的用户列表 ===")
        users = User.query.all()
        if not users:
            print("数据库中没有用户记录！")
        for user in users:
            print(f"\n用户ID: {user.id}")
            print(f"用户名: {user.username}")
            print(f"邮箱: {user.email}")
            print(f"角色: {user.role}")
            print("="*30)

if __name__ == '__main__':
    view_db() 