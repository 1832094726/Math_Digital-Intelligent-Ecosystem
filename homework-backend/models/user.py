# -*- coding: utf-8 -*-
"""
用户模型
"""
import bcrypt
import re
from datetime import datetime
from models.database import db
from config import Config

class User:
    """用户模型类"""
    
    def __init__(self, user_data=None):
        if user_data:
            self.id = user_data.get('id')
            self.username = user_data.get('username')
            self.email = user_data.get('email')
            self.password_hash = user_data.get('password_hash')
            self.role = user_data.get('role', 'student')
            self.real_name = user_data.get('real_name')
            self.grade = user_data.get('grade')
            self.school = user_data.get('school')
            self.class_name = user_data.get('class_name')
            self.student_id = user_data.get('student_id')
            self.phone = user_data.get('phone')
            self.avatar = user_data.get('avatar')
            self.profile = user_data.get('profile')
            self.learning_preferences = user_data.get('learning_preferences')
            self.is_active = user_data.get('is_active', True)
            self.last_login_time = user_data.get('last_login_time')
            self.created_at = user_data.get('created_at')
            self.updated_at = user_data.get('updated_at')
    
    @staticmethod
    def validate_password(password):
        """验证密码强度"""
        if len(password) < Config.PASSWORD_MIN_LENGTH:
            return False, f"密码长度至少{Config.PASSWORD_MIN_LENGTH}位"
        
        # 检查是否包含字母和数字
        if not re.search(r'[a-zA-Z]', password):
            return False, "密码必须包含字母"
        
        if not re.search(r'\d', password):
            return False, "密码必须包含数字"
        
        # 如果要求特殊字符
        if Config.PASSWORD_REQUIRE_SPECIAL:
            if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
                return False, "密码必须包含特殊字符"
        
        return True, "密码格式正确"
    
    @staticmethod
    def validate_email(email):
        """验证邮箱格式"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            return False, "邮箱格式不正确"
        return True, "邮箱格式正确"
    
    @staticmethod
    def hash_password(password):
        """加密密码"""
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
        return password_hash.decode('utf-8')
    
    @staticmethod
    def verify_password(password, password_hash):
        """验证密码"""
        return bcrypt.checkpw(
            password.encode('utf-8'), 
            password_hash.encode('utf-8')
        )
    
    @classmethod
    def create(cls, user_data):
        """创建新用户"""
        # 验证必需字段
        required_fields = ['username', 'email', 'password', 'real_name', 'role']
        for field in required_fields:
            if not user_data.get(field):
                raise ValueError(f"缺少必需字段: {field}")
        
        # 验证邮箱格式
        is_valid, msg = cls.validate_email(user_data['email'])
        if not is_valid:
            raise ValueError(msg)
        
        # 验证密码强度
        is_valid, msg = cls.validate_password(user_data['password'])
        if not is_valid:
            raise ValueError(msg)
        
        # 检查用户名是否已存在
        if cls.get_by_username(user_data['username']):
            raise ValueError("用户名已存在")
        
        # 检查邮箱是否已存在
        if cls.get_by_email(user_data['email']):
            raise ValueError("邮箱已被使用")
        
        # 加密密码
        password_hash = cls.hash_password(user_data['password'])
        
        # 构建插入数据
        insert_data = {
            'username': user_data['username'],
            'email': user_data['email'],
            'password_hash': password_hash,
            'role': user_data['role'],
            'real_name': user_data['real_name'],
            'grade': user_data.get('grade'),
            'school': user_data.get('school'),
            'class_name': user_data.get('class_name'),
            'student_id': user_data.get('student_id'),
            'phone': user_data.get('phone'),
            'avatar': user_data.get('avatar'),
            'profile': user_data.get('profile'),
            'learning_preferences': user_data.get('learning_preferences'),
            'is_active': True,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
        
        # 构建SQL语句
        columns = list(insert_data.keys())
        placeholders = ', '.join(['%s'] * len(columns))
        sql = f"INSERT INTO users ({', '.join(columns)}) VALUES ({placeholders})"
        
        # 执行插入
        user_id = db.execute_insert(sql, list(insert_data.values()))
        
        # 返回创建的用户
        return cls.get_by_id(user_id)
    
    @classmethod
    def get_by_id(cls, user_id):
        """根据ID获取用户"""
        sql = "SELECT * FROM users WHERE id = %s"
        result = db.execute_query_one(sql, (user_id,))
        if result:
            return cls(result)
        return None
    
    @classmethod
    def get_by_username(cls, username):
        """根据用户名获取用户"""
        sql = "SELECT * FROM users WHERE username = %s"
        result = db.execute_query_one(sql, (username,))
        if result:
            return cls(result)
        return None
    
    @classmethod
    def get_by_email(cls, email):
        """根据邮箱获取用户"""
        sql = "SELECT * FROM users WHERE email = %s"
        result = db.execute_query_one(sql, (email,))
        if result:
            return cls(result)
        return None
    
    @classmethod
    def authenticate(cls, username_or_email, password):
        """用户认证"""
        # 尝试用户名登录
        user = cls.get_by_username(username_or_email)
        if not user:
            # 尝试邮箱登录
            user = cls.get_by_email(username_or_email)
        
        if user and user.is_active and cls.verify_password(password, user.password_hash):
            # 更新最后登录时间
            user.update_last_login()
            return user
        
        return None
    
    def update_last_login(self):
        """更新最后登录时间"""
        sql = "UPDATE users SET last_login_time = %s WHERE id = %s"
        db.execute_update(sql, (datetime.now(), self.id))
        self.last_login_time = datetime.now()
    
    def update_profile(self, profile_data):
        """更新用户信息"""
        allowed_fields = [
            'real_name', 'phone', 'avatar', 'profile', 
            'learning_preferences', 'grade', 'school', 'class_name'
        ]
        
        update_fields = []
        update_values = []
        
        for field in allowed_fields:
            if field in profile_data:
                update_fields.append(f"{field} = %s")
                # 如果是JSON字段，需要序列化
                if field in ['profile', 'learning_preferences']:
                    import json
                    if isinstance(profile_data[field], dict):
                        update_values.append(json.dumps(profile_data[field]))
                    else:
                        update_values.append(profile_data[field])
                else:
                    update_values.append(profile_data[field])
        
        if update_fields:
            update_fields.append("updated_at = %s")
            update_values.append(datetime.now())
            update_values.append(self.id)
            
            sql = f"UPDATE users SET {', '.join(update_fields)} WHERE id = %s"
            db.execute_update(sql, update_values)
            
            # 更新实例属性
            for field in allowed_fields:
                if field in profile_data:
                    setattr(self, field, profile_data[field])
            self.updated_at = datetime.now()
    
    def get_permissions(self):
        """获取用户权限"""
        permission_map = {
            'admin': ['*'],
            'teacher': [
                'homework.create', 'homework.edit', 'homework.grade', 
                'student.view', 'class.manage'
            ],
            'student': [
                'homework.view', 'homework.submit', 'profile.edit'
            ],
            'parent': [
                'child.view', 'homework.view', 'report.view'
            ]
        }
        return permission_map.get(self.role, [])
    
    def to_dict(self, include_sensitive=False):
        """转换为字典"""
        data = {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'real_name': self.real_name,
            'grade': self.grade,
            'school': self.school,
            'class_name': self.class_name,
            'student_id': self.student_id,
            'phone': self.phone,
            'avatar': self.avatar,
            'profile': self.profile,
            'learning_preferences': self.learning_preferences,
            'is_active': self.is_active,
            'last_login_time': self.last_login_time.isoformat() if self.last_login_time else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_sensitive:
            data['password_hash'] = self.password_hash
        
        return data

