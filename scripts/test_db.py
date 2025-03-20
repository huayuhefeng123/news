from app.main import create_app
from app.extensions import db
from app.models.user import User
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_db():
    app = create_app()
    with app.app_context():
        try:
            # 测试数据库连接
            db.session.execute('SELECT 1')
            logger.info("数据库连接成功！")
            
            # 删除所有现有用户
            User.query.delete()
            db.session.commit()
            logger.info("清除所有现有用户")
            
            # 创建测试用户
            test_user = User(
                username='admin',
                email='admin@example.com',
                role='admin'
            )
            test_user.set_password('admin123')
            
            db.session.add(test_user)
            db.session.commit()
            logger.info("创建测试用户成功")
            
            # 验证用户是否创建成功
            user = User.query.filter_by(username='admin').first()
            if user:
                logger.info(f"找到用户: {user.username}")
                logger.info(f"密码验证测试: {'成功' if user.check_password('admin123') else '失败'}")
            else:
                logger.error("未找到创建的用户！")
                
        except Exception as e:
            logger.error(f"测试过程中出错: {str(e)}")
            db.session.rollback()

if __name__ == '__main__':
    test_db() 