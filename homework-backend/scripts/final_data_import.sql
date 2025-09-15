-- 最终数据导入脚本
USE homework_system;

-- 1. 清理并创建questions表
DROP TABLE IF EXISTS questions;
CREATE TABLE questions (
  id bigint(20) NOT NULL AUTO_INCREMENT COMMENT '题目ID',
  homework_id bigint(20) NOT NULL COMMENT '作业ID',
  content text NOT NULL COMMENT '题目内容',
  question_type enum('single_choice','multiple_choice','fill_blank','calculation','proof','application') NOT NULL DEFAULT 'single_choice' COMMENT '题目类型',
  options json DEFAULT NULL COMMENT '选择题选项(JSON格式)',
  correct_answer text NOT NULL COMMENT '正确答案',
  score decimal(5,2) NOT NULL DEFAULT '0.00' COMMENT '题目分值',
  difficulty int(11) NOT NULL DEFAULT '1' COMMENT '难度等级(1-5)',
  order_index int(11) NOT NULL DEFAULT '1' COMMENT '题目顺序',
  knowledge_points json DEFAULT NULL COMMENT '关联知识点ID列表',
  explanation text DEFAULT NULL COMMENT '题目解析',
  created_at timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  updated_at timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_homework_id (homework_id),
  KEY idx_question_type (question_type),
  KEY idx_difficulty (difficulty)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='题目表';

-- 2. 插入教师数据
INSERT IGNORE INTO users (id, username, email, password_hash, real_name, role, grade, school, created_at, updated_at) VALUES
(10, 'teacher_wang', 'wang@school.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', '王老师', 'teacher', 7, '北京市第一中学', NOW(), NOW()),
(11, 'teacher_li', 'li@school.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', '李老师', 'teacher', 7, '北京市第一中学', NOW(), NOW()),
(12, 'teacher_zhang', 'zhang@school.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', '张老师', 'teacher', 7, '北京市第一中学', NOW(), NOW());

-- 3. 插入学生数据
INSERT IGNORE INTO users (id, username, email, password_hash, real_name, role, grade, school, created_at, updated_at) VALUES
(20, 'student_001', 'student001@school.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', '张三', 'student', 7, '北京市第一中学', NOW(), NOW()),
(21, 'student_002', 'student002@school.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', '李四', 'student', 7, '北京市第一中学', NOW(), NOW()),
(22, 'student_003', 'student003@school.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', '王五', 'student', 7, '北京市第一中学', NOW(), NOW()),
(23, 'student_004', 'student004@school.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', '赵六', 'student', 7, '北京市第一中学', NOW(), NOW()),
(24, 'student_005', 'student005@school.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', '钱七', 'student', 7, '北京市第一中学', NOW(), NOW()),
(25, 'student_006', 'student006@school.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', '孙八', 'student', 7, '北京市第一中学', NOW(), NOW());

-- 4. 插入班级数据
INSERT IGNORE INTO classes (id, name, grade, school, teacher_id, created_at, updated_at) VALUES
(1, '七年级1班', 7, '北京市第一中学', 10, NOW(), NOW()),
(2, '七年级2班', 7, '北京市第一中学', 11, NOW(), NOW());

-- 5. 插入学生班级关系
INSERT IGNORE INTO student_classes (student_id, class_id, created_at) VALUES
(20, 1, NOW()), (21, 1, NOW()), (22, 1, NOW()),  -- 1班学生
(23, 2, NOW()), (24, 2, NOW()), (25, 2, NOW());  -- 2班学生

-- 6. 插入作业数据
INSERT IGNORE INTO homeworks (id, title, subject, description, grade, difficulty_level, created_by, due_date, start_date, time_limit, max_score, max_attempts, is_published, auto_grade, show_answers, instructions, tags, category, created_at, updated_at) VALUES
(1, '有理数运算练习 - 七年级1班', '数学', '练习有理数的加减乘除运算', 7, 2, 10, DATE_ADD(NOW(), INTERVAL 7 DAY), NOW(), 60, 50, 2, 1, 1, 0, '请仔细作答，注意运算顺序', '["练习", "有理数"]', '课后练习', NOW(), NOW()),
(2, '有理数运算练习 - 七年级2班', '数学', '练习有理数的加减乘除运算', 7, 2, 10, DATE_ADD(NOW(), INTERVAL 7 DAY), NOW(), 60, 50, 2, 1, 1, 0, '请仔细作答，注意运算顺序', '["练习", "有理数"]', '课后练习', NOW(), NOW()),
(3, '代数式化简 - 七年级1班', '数学', '练习代数式的化简和求值', 7, 3, 11, DATE_ADD(NOW(), INTERVAL 7 DAY), NOW(), 60, 50, 2, 1, 1, 0, '请仔细作答，注意化简步骤', '["练习", "代数式"]', '课后练习', NOW(), NOW()),
(4, '代数式化简 - 七年级2班', '数学', '练习代数式的化简和求值', 7, 3, 11, DATE_ADD(NOW(), INTERVAL 7 DAY), NOW(), 60, 50, 2, 1, 1, 0, '请仔细作答，注意化简步骤', '["练习", "代数式"]', '课后练习', NOW(), NOW()),
(5, '几何图形认识 - 七年级1班', '数学', '认识基本几何图形和性质', 7, 2, 12, DATE_ADD(NOW(), INTERVAL 7 DAY), NOW(), 60, 50, 2, 1, 1, 0, '请仔细观察图形特征', '["练习", "几何"]', '课后练习', NOW(), NOW()),
(6, '几何图形认识 - 七年级2班', '数学', '认识基本几何图形和性质', 7, 2, 12, DATE_ADD(NOW(), INTERVAL 7 DAY), NOW(), 60, 50, 2, 1, 1, 0, '请仔细观察图形特征', '["练习", "几何"]', '课后练习', NOW(), NOW());

-- 7. 插入题目数据
INSERT IGNORE INTO questions (id, homework_id, content, question_type, options, correct_answer, score, difficulty, order_index, knowledge_points, explanation) VALUES
-- 有理数运算题目 (作业1)
(1, 1, '计算：(-3) + 5 - 2 = ?', 'single_choice', '["0", "2", "4", "-10"]', '0', 25, 2, 1, '[1]', '根据有理数运算法则：(-3) + 5 - 2 = 2 - 2 = 0'),
(2, 1, '计算：(-2) × 3 ÷ (-1) = ?', 'fill_blank', NULL, '6', 25, 2, 2, '[1]', '根据有理数运算法则：(-2) × 3 ÷ (-1) = -6 ÷ (-1) = 6'),
-- 有理数运算题目 (作业2)
(3, 2, '计算：(-3) + 5 - 2 = ?', 'single_choice', '["0", "2", "4", "-10"]', '0', 25, 2, 1, '[1]', '根据有理数运算法则：(-3) + 5 - 2 = 2 - 2 = 0'),
(4, 2, '计算：(-2) × 3 ÷ (-1) = ?', 'fill_blank', NULL, '6', 25, 2, 2, '[1]', '根据有理数运算法则：(-2) × 3 ÷ (-1) = -6 ÷ (-1) = 6'),
-- 代数式题目 (作业3)
(5, 3, '化简：3x + 2x - x = ?', 'fill_blank', NULL, '4x', 30, 3, 1, '[2]', '合并同类项：3x + 2x - x = (3+2-1)x = 4x'),
(6, 3, '当x=2时，代数式2x+1的值是多少？', 'single_choice', '["3", "4", "5", "6"]', '5', 20, 3, 2, '[2]', '将x=2代入：2×2+1 = 4+1 = 5'),
-- 代数式题目 (作业4)
(7, 4, '化简：3x + 2x - x = ?', 'fill_blank', NULL, '4x', 30, 3, 1, '[2]', '合并同类项：3x + 2x - x = (3+2-1)x = 4x'),
(8, 4, '当x=2时，代数式2x+1的值是多少？', 'single_choice', '["3", "4", "5", "6"]', '5', 20, 3, 2, '[2]', '将x=2代入：2×2+1 = 4+1 = 5'),
-- 几何题目 (作业5)
(9, 5, '一个三角形的三个内角分别是60°、60°、60°，这是什么三角形？', 'single_choice', '["直角三角形", "等腰三角形", "等边三角形", "钝角三角形"]', '等边三角形', 25, 2, 1, '[3]', '三个角都是60°的三角形是等边三角形'),
(10, 5, '正方形有几条对称轴？', 'fill_blank', NULL, '4', 25, 2, 2, '[3]', '正方形有4条对称轴：2条对角线和2条中线'),
-- 几何题目 (作业6)
(11, 6, '一个三角形的三个内角分别是60°、60°、60°，这是什么三角形？', 'single_choice', '["直角三角形", "等腰三角形", "等边三角形", "钝角三角形"]', '等边三角形', 25, 2, 1, '[3]', '三个角都是60°的三角形是等边三角形'),
(12, 6, '正方形有几条对称轴？', 'fill_blank', NULL, '4', 25, 2, 2, '[3]', '正方形有4条对称轴：2条对角线和2条中线');

-- 8. 插入作业分配
INSERT IGNORE INTO homework_assignments (id, homework_id, assigned_to_type, assigned_to_id, assigned_by, assigned_at, is_active) VALUES
(1, 1, 'class', 1, 10, NOW(), 1),  -- 王老师给1班布置有理数作业
(2, 2, 'class', 2, 10, NOW(), 1),  -- 王老师给2班布置有理数作业
(3, 3, 'class', 1, 11, NOW(), 1),  -- 李老师给1班布置代数式作业
(4, 4, 'class', 2, 11, NOW(), 1),  -- 李老师给2班布置代数式作业
(5, 5, 'class', 1, 12, NOW(), 1),  -- 张老师给1班布置几何作业
(6, 6, 'class', 2, 12, NOW(), 1);  -- 张老师给2班布置几何作业

-- 9. 插入学生作业提交记录（每个学生完成所有作业）
INSERT IGNORE INTO homework_submissions (assignment_id, student_id, answers, score, time_spent, status, submitted_at) VALUES
-- 学生20 (张三, 1班)
(1, 20, '{"1": "0", "2": "6"}', 50, 25, 'submitted', NOW()),  -- 有理数作业 - 全对
(3, 20, '{"5": "4x", "6": "5"}', 50, 30, 'submitted', NOW()),  -- 代数式作业 - 全对
(5, 20, '{"9": "等边三角形", "10": "4"}', 50, 20, 'submitted', NOW()),  -- 几何作业 - 全对
-- 学生21 (李四, 1班)
(1, 21, '{"1": "2", "2": "6"}', 25, 28, 'submitted', NOW()),  -- 有理数作业 - 第一题错
(3, 21, '{"5": "4x", "6": "4"}', 30, 35, 'submitted', NOW()),  -- 代数式作业 - 第二题错
(5, 21, '{"9": "等边三角形", "10": "4"}', 50, 22, 'submitted', NOW()),  -- 几何作业 - 全对
-- 学生22 (王五, 1班)
(1, 22, '{"1": "0", "2": "错误"}', 25, 32, 'submitted', NOW()),  -- 有理数作业 - 第二题错
(3, 22, '{"5": "5x", "6": "5"}', 20, 40, 'submitted', NOW()),  -- 代数式作业 - 第一题错
(5, 22, '{"9": "直角三角形", "10": "4"}', 25, 25, 'submitted', NOW()),  -- 几何作业 - 第一题错
-- 学生23 (赵六, 2班)
(2, 23, '{"3": "0", "4": "6"}', 50, 26, 'submitted', NOW()),  -- 有理数作业 - 全对
(4, 23, '{"7": "4x", "8": "5"}', 50, 32, 'submitted', NOW()),  -- 代数式作业 - 全对
(6, 23, '{"11": "等边三角形", "12": "4"}', 50, 21, 'submitted', NOW()),  -- 几何作业 - 全对
-- 学生24 (钱七, 2班)
(2, 24, '{"3": "4", "4": "6"}', 25, 30, 'submitted', NOW()),  -- 有理数作业 - 第一题错
(4, 24, '{"7": "4x", "8": "3"}', 30, 38, 'submitted', NOW()),  -- 代数式作业 - 第二题错
(6, 24, '{"11": "等腰三角形", "12": "4"}', 25, 27, 'submitted', NOW()),  -- 几何作业 - 第一题错
-- 学生25 (孙八, 2班)
(2, 25, '{"3": "0", "4": "错误"}', 25, 35, 'submitted', NOW()),  -- 有理数作业 - 第二题错
(4, 25, '{"7": "3x", "8": "5"}', 20, 42, 'submitted', NOW()),  -- 代数式作业 - 第一题错
(6, 25, '{"11": "等边三角形", "12": "3"}', 25, 28, 'submitted', NOW());  -- 几何作业 - 第二题错

-- 10. 插入知识点数据
INSERT IGNORE INTO knowledge_points (id, name, description, subject, grade, parent_id, created_at, updated_at) VALUES
(1, '有理数运算', '有理数的加减乘除运算法则', '数学', 7, NULL, NOW(), NOW()),
(2, '代数式', '代数式的化简和求值', '数学', 7, NULL, NOW(), NOW()),
(3, '几何图形', '基本几何图形的认识和性质', '数学', 7, NULL, NOW(), NOW());

-- 11. 显示导入结果
SELECT '✅ 数据导入完成！' as message;
SELECT COUNT(*) as '教师数量' FROM users WHERE role='teacher';
SELECT COUNT(*) as '学生数量' FROM users WHERE role='student';
SELECT COUNT(*) as '作业数量' FROM homeworks;
SELECT COUNT(*) as '题目数量' FROM questions;
SELECT COUNT(*) as '提交数量' FROM homework_submissions;
SELECT COUNT(*) as '知识点数量' FROM knowledge_points;
