#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统维护脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import db
from datetime import datetime, timedelta
import json

def system_health_check():
    """系统健康检查"""
    print("🏥 系统健康检查...")
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 1. 数据库连接检查
                cursor.execute("SELECT 1 as test")
                print("✅ 数据库连接正常")
                
                # 2. 核心表检查
                core_tables = ['users', 'homeworks', 'homework_assignments', 'homework_submissions']
                for table in core_tables:
                    cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                    count = cursor.fetchone()['count']
                    print(f"✅ {table}: {count} 条记录")
                
                # 3. 检查最近活动
                cursor.execute("""
                    SELECT COUNT(*) as count 
                    FROM homework_submissions 
                    WHERE submitted_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                """)
                recent_submissions = cursor.fetchone()['count']
                print(f"📊 最近7天提交: {recent_submissions} 份")
                
                # 4. 检查系统性能指标
                cursor.execute("SELECT COUNT(*) as count FROM users WHERE role='student'")
                student_count = cursor.fetchone()['count']
                
                cursor.execute("SELECT COUNT(*) as count FROM users WHERE role='teacher'")
                teacher_count = cursor.fetchone()['count']
                
                print(f"👥 用户统计: {student_count} 学生, {teacher_count} 教师")
                
                return True
                
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        return False

def cleanup_old_data():
    """清理旧数据"""
    print("\n🧹 清理旧数据...")
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 1. 清理过期的会话令牌
                cursor.execute("""
                    DELETE FROM user_sessions 
                    WHERE expires_at < NOW()
                """)
                deleted_sessions = cursor.rowcount
                print(f"🗑️ 清理过期会话: {deleted_sessions} 条")
                
                # 2. 清理旧的日志记录（如果有的话）
                cursor.execute("""
                    DELETE FROM system_logs 
                    WHERE created_at < DATE_SUB(NOW(), INTERVAL 30 DAY)
                """)
                deleted_logs = cursor.rowcount
                print(f"🗑️ 清理旧日志: {deleted_logs} 条")
                
                conn.commit()
                print("✅ 数据清理完成")
                
    except Exception as e:
        print(f"⚠️ 数据清理警告: {e}")

def update_statistics():
    """更新统计信息"""
    print("\n📈 更新统计信息...")
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 1. 更新作业完成率统计
                cursor.execute("""
                    SELECT 
                        h.id,
                        h.title,
                        COUNT(ha.id) as assigned_count,
                        COUNT(hs.id) as submitted_count,
                        ROUND(COUNT(hs.id) * 100.0 / COUNT(ha.id), 2) as completion_rate
                    FROM homeworks h
                    LEFT JOIN homework_assignments ha ON h.id = ha.homework_id
                    LEFT JOIN homework_submissions hs ON ha.id = hs.assignment_id
                    WHERE h.is_published = 1
                    GROUP BY h.id, h.title
                """)
                
                homework_stats = cursor.fetchall()
                print("📚 作业完成率统计:")
                for stat in homework_stats:
                    print(f"  {stat['title']}: {stat['completion_rate']}% ({stat['submitted_count']}/{stat['assigned_count']})")
                
                # 2. 更新学生活跃度统计
                cursor.execute("""
                    SELECT 
                        u.real_name,
                        COUNT(hs.id) as submission_count,
                        AVG(hs.score) as avg_score,
                        MAX(hs.submitted_at) as last_activity
                    FROM users u
                    LEFT JOIN homework_submissions hs ON u.id = hs.student_id
                    WHERE u.role = 'student'
                    GROUP BY u.id, u.real_name
                    ORDER BY submission_count DESC
                """)
                
                student_stats = cursor.fetchall()
                print("\n👨‍🎓 学生活跃度统计:")
                for stat in student_stats[:5]:  # 显示前5名
                    avg_score = stat['avg_score'] or 0
                    last_activity = stat['last_activity'] or '从未提交'
                    print(f"  {stat['real_name']}: {stat['submission_count']} 份作业, 平均分 {avg_score:.1f}")
                
                print("✅ 统计信息更新完成")
                
    except Exception as e:
        print(f"❌ 统计更新失败: {e}")

def check_system_alerts():
    """检查系统警报"""
    print("\n🚨 检查系统警报...")
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                alerts = []
                
                # 1. 检查未提交作业
                cursor.execute("""
                    SELECT COUNT(*) as count
                    FROM homework_assignments ha
                    JOIN homeworks h ON ha.homework_id = h.id
                    LEFT JOIN homework_submissions hs ON ha.id = hs.assignment_id
                    WHERE h.due_date < NOW() AND hs.id IS NULL
                """)
                overdue_count = cursor.fetchone()['count']
                if overdue_count > 0:
                    alerts.append(f"⚠️ {overdue_count} 份作业已过期未提交")
                
                # 2. 检查系统错误
                cursor.execute("""
                    SELECT COUNT(*) as count
                    FROM system_logs
                    WHERE level = 'ERROR' AND created_at >= DATE_SUB(NOW(), INTERVAL 1 DAY)
                """)
                error_count = cursor.fetchone()['count']
                if error_count > 10:
                    alerts.append(f"⚠️ 最近24小时有 {error_count} 个系统错误")
                
                # 3. 检查数据库性能
                cursor.execute("SHOW PROCESSLIST")
                processes = cursor.fetchall()
                long_queries = [p for p in processes if p.get('Time', 0) > 30]
                if long_queries:
                    alerts.append(f"⚠️ 发现 {len(long_queries)} 个长时间运行的查询")
                
                if alerts:
                    print("发现以下警报:")
                    for alert in alerts:
                        print(f"  {alert}")
                else:
                    print("✅ 无系统警报")
                
    except Exception as e:
        print(f"⚠️ 警报检查失败: {e}")

def backup_critical_data():
    """备份关键数据"""
    print("\n💾 备份关键数据...")
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 1. 导出用户数据
                cursor.execute("SELECT * FROM users")
                users = cursor.fetchall()
                
                # 2. 导出作业数据
                cursor.execute("SELECT * FROM homeworks")
                homeworks = cursor.fetchall()
                
                # 3. 导出提交数据
                cursor.execute("SELECT * FROM homework_submissions")
                submissions = cursor.fetchall()
                
                # 创建备份目录
                backup_dir = "backups"
                if not os.path.exists(backup_dir):
                    os.makedirs(backup_dir)
                
                # 保存备份文件
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = f"{backup_dir}/backup_{timestamp}.json"
                
                backup_data = {
                    'timestamp': timestamp,
                    'users': users,
                    'homeworks': homeworks,
                    'submissions': submissions
                }
                
                with open(backup_file, 'w', encoding='utf-8') as f:
                    json.dump(backup_data, f, ensure_ascii=False, indent=2, default=str)
                
                print(f"✅ 数据备份完成: {backup_file}")
                
    except Exception as e:
        print(f"❌ 数据备份失败: {e}")

def optimize_database():
    """优化数据库"""
    print("\n⚡ 优化数据库...")
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 1. 分析表
                tables = ['users', 'homeworks', 'homework_assignments', 'homework_submissions']
                for table in tables:
                    cursor.execute(f"ANALYZE TABLE {table}")
                    print(f"✅ 分析表 {table}")
                
                # 2. 优化表
                for table in tables:
                    cursor.execute(f"OPTIMIZE TABLE {table}")
                    print(f"✅ 优化表 {table}")
                
                print("✅ 数据库优化完成")
                
    except Exception as e:
        print(f"❌ 数据库优化失败: {e}")

def main():
    """主函数"""
    print("🔧 系统维护开始...")
    print(f"⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 执行维护任务
    system_health_check()
    cleanup_old_data()
    update_statistics()
    check_system_alerts()
    backup_critical_data()
    optimize_database()
    
    print("\n✅ 系统维护完成!")

if __name__ == "__main__":
    main()
