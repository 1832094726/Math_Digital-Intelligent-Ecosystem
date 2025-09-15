# -*- coding: utf-8 -*-
"""
数据库连接和基础操作
"""
import pymysql
import pymysql.cursors
from config import Config
import logging
from contextlib import contextmanager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        self.config = Config.DATABASE_CONFIG
        
    @contextmanager
    def get_connection(self):
        """获取数据库连接上下文管理器"""
        connection = None
        try:
            connection = pymysql.connect(
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['user'],
                password=self.config['password'],
                database=self.config['database'],
                charset=self.config['charset'],
                cursorclass=pymysql.cursors.DictCursor,
                autocommit=False
            )
            yield connection
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"数据库连接错误: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    def execute_query(self, sql, params=None):
        """执行查询语句"""
        with self.get_connection() as conn:
            try:
                with conn.cursor() as cursor:
                    cursor.execute(sql, params)
                    result = cursor.fetchall()
                    return result
            except Exception as e:
                logger.error(f"查询执行错误: {e}")
                raise
    
    def execute_query_one(self, sql, params=None):
        """执行查询语句，返回单行结果"""
        with self.get_connection() as conn:
            try:
                with conn.cursor() as cursor:
                    cursor.execute(sql, params)
                    result = cursor.fetchone()
                    return result
            except Exception as e:
                logger.error(f"查询执行错误: {e}")
                raise
    
    def execute_insert(self, sql, params=None):
        """执行插入语句"""
        with self.get_connection() as conn:
            try:
                with conn.cursor() as cursor:
                    cursor.execute(sql, params)
                    conn.commit()
                    return cursor.lastrowid
            except Exception as e:
                conn.rollback()
                logger.error(f"插入执行错误: {e}")
                raise
    
    def execute_update(self, sql, params=None):
        """执行更新语句"""
        with self.get_connection() as conn:
            try:
                with conn.cursor() as cursor:
                    affected_rows = cursor.execute(sql, params)
                    conn.commit()
                    return affected_rows
            except Exception as e:
                conn.rollback()
                logger.error(f"更新执行错误: {e}")
                raise
    
    def execute_delete(self, sql, params=None):
        """执行删除语句"""
        with self.get_connection() as conn:
            try:
                with conn.cursor() as cursor:
                    affected_rows = cursor.execute(sql, params)
                    conn.commit()
                    return affected_rows
            except Exception as e:
                conn.rollback()
                logger.error(f"删除执行错误: {e}")
                raise

# 全局数据库管理器实例
db = DatabaseManager()

def get_db_connection():
    """获取数据库连接 - 兼容旧接口"""
    config = Config.DATABASE_CONFIG
    return pymysql.connect(
        host=config['host'],
        port=config['port'],
        user=config['user'],
        password=config['password'],
        database=config['database'],
        charset=config['charset'],
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False
    )


