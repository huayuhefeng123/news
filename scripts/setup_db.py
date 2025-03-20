from pathlib import Path
from app.main import create_app
from app.extensions import db
from app.models.user import User
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_database():
    logger.info("开始设置数据库...")
    
    app = create_app()
    
    with app.app_context():
        try:
            # 确保数据库目录存在
            db_path = Path(app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', ''))
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 重新创建所有表
            logger.info("重新创建数据库表...")
            db.drop_all()
            db.create_all()
            
            # 创建管理员用户
            admin = User(
                username='admin',
                email='admin@example.com',
                role='admin'
            )
            admin.set_password('admin123')
            
            db.session.add(admin)
            db.session.commit()
            
            # 验证用户创建
            admin_check = User.query.filter_by(username='admin').first()
            if admin_check and admin_check.check_password('admin123'):
                logger.info("管理员用户创建成功！")
                logger.info(f"用户名: admin")
                logger.info(f"密码: admin123")
                logger.info(f"邮箱: admin@example.com")
            else:
                logger.error("管理员用户验证失败！")
            
        except Exception as e:
            logger.error(f"设置数据库时出错: {str(e)}", exc_info=True)
            raise

if __name__ == '__main__':
    setup_database() 