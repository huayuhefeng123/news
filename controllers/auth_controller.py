from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from app.models.user import User
from app.extensions import db
import jwt
import datetime
from app.config import Config
import logging
import re

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        logger.info(f"Received registration request: {data}")
        
        # 验证必要字段
        required_fields = ['username', 'password', 'email']
        missing_fields = [field for field in required_fields if not data.get(field)]
        if missing_fields:
            return jsonify({
                'error': f'缺少必要字段: {", ".join(missing_fields)}'
            }), 400
        
        # 验证用户名长度
        if len(data['username']) < 3 or len(data['username']) > 20:
            return jsonify({
                'error': '用户名长度必须在3-20个字符之间'
            }), 400
            
        # 验证密码长度
        if len(data['password']) < 6:
            return jsonify({
                'error': '密码长度不能小于6个字符'
            }), 400
            
        # 验证邮箱格式
        if not re.match(r"[^@]+@[^@]+\.[^@]+", data['email']):
            return jsonify({
                'error': '请输入有效的邮箱地址'
            }), 400
        
        # 检查用户名是否已存在
        if User.query.filter_by(username=data['username']).first():
            return jsonify({
                'error': '用户名已存在'
            }), 400
        
        # 检查邮箱是否已存在
        if User.query.filter_by(email=data['email']).first():
            return jsonify({
                'error': '邮箱已被注册'
            }), 400
        
        # 创建新用户
        new_user = User(
            username=data['username'],
            email=data['email'],
            role='user'  # 默认角色
        )
        new_user.set_password(data['password'])
        
        try:
            db.session.add(new_user)
            db.session.commit()
            logger.info(f"Successfully registered user: {data['username']}")
            return jsonify({
                'message': '注册成功',
                'user': {
                    'username': new_user.username,
                    'email': new_user.email,
                    'role': new_user.role
                }
            }), 201
        except Exception as e:
            db.session.rollback()
            logger.error(f"Database error during registration: {str(e)}")
            return jsonify({
                'error': '注册失败，请稍后重试'
            }), 500
            
    except Exception as e:
        logger.error(f"Error during registration: {str(e)}")
        return jsonify({
            'error': '注册失败，请稍后重试'
        }), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        logger.info(f"Received login request for user: {username}")
        
        # 查找用户
        user = User.query.filter_by(username=username).first()
        
        if not user:
            logger.warning(f"Login failed: User not found: {username}")
            return jsonify({
                'success': False,
                'message': '用户名或密码错误'
            }), 401
        
        # 验证密码
        if user.check_password(password):
            logger.info(f"Login successful for user: {username}")
            
            # 生成 JWT token
            token = jwt.encode({
                'user_id': user.id,
                'username': user.username,
                'role': user.role,
                'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1)
            }, Config.SECRET_KEY, algorithm='HS256')
            
            # 返回更多用户信息和 token
            return jsonify({
                'success': True,
                'message': '登录成功',
                'token': token,
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'role': user.role
                }
            })
        else:
            logger.warning(f"Login failed: Invalid password for user: {username}")
            return jsonify({
                'success': False,
                'message': '用户名或密码错误'
            }), 401
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': '登录失败，请稍后重试'
        }), 500

@auth_bp.route('/logout', methods=['POST'])
def logout():
    # 由于使用JWT，服务器端不需要处理登出
    return jsonify({'message': '登出成功'})

@auth_bp.route('/user', methods=['GET'])
def get_user_info():
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({'error': '未提供认证令牌'}), 401
    
    try:
        token = token.split(' ')[1]  # Bearer token
        payload = jwt.decode(token, Config.SECRET_KEY, algorithms=['HS256'])
        user = User.query.get(payload['user_id'])
        
        if not user:
            return jsonify({'error': '用户不存在'}), 404
        
        return jsonify({
            'id': user.id,
            'username': user.username,
            'email': user.email
        })
    except jwt.ExpiredSignatureError:
        return jsonify({'error': '令牌已过期'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'error': '无效的令牌'}), 401 