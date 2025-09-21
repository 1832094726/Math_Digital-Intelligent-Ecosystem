# -*- coding: utf-8 -*-
"""
数据库连接和基础操作
"""
import pymysql
import pymysql.cursors
from config import Config
import logging
from contextlib import contextmanager
import threading
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        self.config = Config.DATABASE_CONFIG
        self._connection_pool = []
        self._pool_lock = threading.Lock()
        self._max_connections = 5
        self._connection_timeout = 30
        
    def _create_connection(self):
        """创建新的数据库连接"""
        try:
            connection = pymysql.connect(
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['user'],
                password=self.config['password'],
                database=self.config['database'],
                charset=self.config['charset'],
                cursorclass=pymysql.cursors.DictCursor,
                autocommit=False,
                connect_timeout=10,
                read_timeout=30,
                write_timeout=30
            )
            return connection
        except Exception as e:
            logger.error(f"创建数据库连接失败: {e}")
            raise

    def _get_connection_from_pool(self):
        """从连接池获取连接"""
        with self._pool_lock:
            # 清理过期连接
            current_time = time.time()
            self._connection_pool = [
                conn for conn in self._connection_pool 
                if hasattr(conn, '_last_used') and 
                current_time - conn._last_used < self._connection_timeout
            ]
            
            # 尝试获取可用连接
            for i, conn in enumerate(self._connection_pool):
                try:
                    # 测试连接是否有效
                    conn.ping(reconnect=False)
                    conn._last_used = current_time
                    return self._connection_pool.pop(i)
                except:
                    continue
            
            # 如果池中没有可用连接且未达到最大连接数，创建新连接
            if len(self._connection_pool) < self._max_connections:
                return self._create_connection()
            
            # 如果达到最大连接数，等待或创建临时连接
            return self._create_connection()

    def _return_connection_to_pool(self, connection):
        """将连接返回到连接池"""
        if connection and not connection.open:
            return
            
        with self._pool_lock:
            if len(self._connection_pool) < self._max_connections:
                connection._last_used = time.time()
                self._connection_pool.append(connection)
            else:
                connection.close()

    @contextmanager
    def get_connection(self):
        """获取数据库连接上下文管理器（使用连接池）"""
        connection = None
        try:
            connection = self._get_connection_from_pool()
            yield connection
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"数据库连接错误: {e}")
            raise
        finally:
            if connection:
                try:
                    self._return_connection_to_pool(connection)
                except Exception as e:
                    logger.error(f"返回连接到池时出错: {e}")
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


