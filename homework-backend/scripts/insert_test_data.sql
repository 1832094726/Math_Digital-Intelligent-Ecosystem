-- 插入测试数据SQL脚本
-- 用于快速创建完整的测试数据集

-- 1. 插入学校数据
INSERT IGNORE INTO schools (id, school_name, school_code, school_type, address, phone, principal, established_year, description, created_at, updated_at)
VALUES (1, '北京市第一中学', 'BJ001', 'public', '北京市朝阳区教育路123号', '010-12345678', '张校长', 1950, '北京市重点中学，数学教育特色学校', NOW(), NOW());

-- 2. 插入年级数据
INSERT IGNORE INTO grades (id, school_id, grade_name, grade_level, academic_year, grade_director, created_at, updated_at)
VALUES (1, 1, '七年级', 7, '2024-2025', '李主任', NOW(), NOW());

-- 3. 插入老师数据
INSERT IGNORE INTO users (id, username, email, password_hash, real_name, role, grade, school, phone, profile, created_at, updated_at) VALUES
(10, 'teacher_wang', 'wang@school.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', '王老师', 'teacher', 7, '北京市第一中学', '13800001001', '{"subject": "数学", "teaching_years": 10, "specialty": "代数教学"}', NOW(), NOW()),
(11, 'teacher_li', 'li@school.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', '李老师', 'teacher', 7, '北京市第一中学', '13800001002', '{"subject": "数学", "teaching_years": 8, "specialty": "几何教学"}', NOW(), NOW()),
(12, 'teacher_zhang', 'zhang@school.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', '张老师', 'teacher', 7, '北京市第一中学', '13800001003', '{"subject": "数学", "teaching_years": 12, "specialty": "应用题教学"}', NOW(), NOW());

-- 4. 插入班级数据
INSERT IGNORE INTO classes (id, school_id, grade_id, class_name, class_code, head_teacher_id, class_size, classroom, created_at, updated_at) VALUES
(1, 1, 1, '七年级1班', 'G7C1', 10, 3, '教学楼A101', NOW(), NOW()),
(2, 1, 1, '七年级2班', 'G7C2', 11, 3, '教学楼A102', NOW(), NOW());

-- 5. 插入学生数据
INSERT IGNORE INTO users (id, username, email, password_hash, real_name, role, grade, school, class_name, student_id, profile, created_at, updated_at) VALUES
-- 1班学生
(20, 'student_1_1', 'student11@school.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', '小明', 'student', 7, '北京市第一中学', '七年级1班', '20240101', '{"interests": ["数学", "科学"], "learning_style": "视觉型"}', NOW(), NOW()),
(21, 'student_1_2', 'student12@school.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', '小红', 'student', 7, '北京市第一中学', '七年级1班', '20240102', '{"interests": ["数学", "科学"], "learning_style": "听觉型"}', NOW(), NOW()),
(22, 'student_1_3', 'student13@school.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', '小刚', 'student', 7, '北京市第一中学', '七年级1班', '20240103', '{"interests": ["数学", "科学"], "learning_style": "动手型"}', NOW(), NOW()),
-- 2班学生
(23, 'student_2_1', 'student21@school.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', '小华', 'student', 7, '北京市第一中学', '七年级2班', '20240201', '{"interests": ["数学", "科学"], "learning_style": "视觉型"}', NOW(), NOW()),
(24, 'student_2_2', 'student22@school.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', '小丽', 'student', 7, '北京市第一中学', '七年级2班', '20240202', '{"interests": ["数学", "科学"], "learning_style": "听觉型"}', NOW(), NOW()),
(25, 'student_2_3', 'student23@school.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', '小强', 'student', 7, '北京市第一中学', '七年级2班', '20240203', '{"interests": ["数学", "科学"], "learning_style": "动手型"}', NOW(), NOW());

-- 6. 插入班级学生关系
INSERT IGNORE INTO class_students (class_id, student_id, enrollment_date, is_active) VALUES
(1, 20, CURDATE(), 1), (1, 21, CURDATE(), 1), (1, 22, CURDATE(), 1),
(2, 23, CURDATE(), 1), (2, 24, CURDATE(), 1), (2, 25, CURDATE(), 1);

-- 7. 插入知识点数据
INSERT IGNORE INTO knowledge_points (id, name, description, subject, grade, difficulty, parent_id, created_at, updated_at) VALUES
(1, '有理数运算', '有理数的加减乘除运算', '数学', 7, 2, NULL, NOW(), NOW()),
(2, '代数式', '代数式的基本概念和运算', '数学', 7, 3, NULL, NOW(), NOW()),
(3, '几何图形', '平面几何基础图形', '数学', 7, 2, NULL, NOW(), NOW());

-- 8. 插入作业数据
INSERT IGNORE INTO homeworks (id, title, subject, description, grade, difficulty_level, created_by, due_date, start_date, time_limit, max_score, max_attempts, is_published, auto_grade, show_answers, instructions, tags, category, created_at, updated_at) VALUES
-- 王老师的作业
(1, '有理数运算练习 - 七年级1班', '数学', '练习有理数的加减乘除运算', 7, 2, 10, DATE_ADD(NOW(), INTERVAL 7 DAY), NOW(), 60, 50, 2, 1, 1, 0, '请仔细阅读题目，认真作答。', '["基础", "练习"]', '课后练习', NOW(), NOW()),
(2, '有理数运算练习 - 七年级2班', '数学', '练习有理数的加减乘除运算', 7, 2, 10, DATE_ADD(NOW(), INTERVAL 7 DAY), NOW(), 60, 50, 2, 1, 1, 0, '请仔细阅读题目，认真作答。', '["基础", "练习"]', '课后练习', NOW(), NOW()),
-- 李老师的作业
(3, '代数式化简 - 七年级1班', '数学', '学习代数式的化简方法', 7, 3, 11, DATE_ADD(NOW(), INTERVAL 7 DAY), NOW(), 60, 50, 2, 1, 1, 0, '请仔细阅读题目，认真作答。', '["基础", "练习"]', '课后练习', NOW(), NOW()),
(4, '代数式化简 - 七年级2班', '数学', '学习代数式的化简方法', 7, 3, 11, DATE_ADD(NOW(), INTERVAL 7 DAY), NOW(), 60, 50, 2, 1, 1, 0, '请仔细阅读题目，认真作答。', '["基础", "练习"]', '课后练习', NOW(), NOW()),
-- 张老师的作业
(5, '几何图形认识 - 七年级1班', '数学', '认识基本的几何图形', 7, 2, 12, DATE_ADD(NOW(), INTERVAL 7 DAY), NOW(), 60, 50, 2, 1, 1, 0, '请仔细阅读题目，认真作答。', '["基础", "练习"]', '课后练习', NOW(), NOW()),
(6, '几何图形认识 - 七年级2班', '数学', '认识基本的几何图形', 7, 2, 12, DATE_ADD(NOW(), INTERVAL 7 DAY), NOW(), 60, 50, 2, 1, 1, 0, '请仔细阅读题目，认真作答。', '["基础", "练习"]', '课后练习', NOW(), NOW());

-- 9. 确保questions表存在并插入题目数据
-- 删除不需要的测试表
DROP TABLE IF EXISTS `simple_questions`;
DROP TABLE IF EXISTS `test_questions`;

-- 确保questions表结构正确
DROP TABLE IF EXISTS `questions`;
CREATE TABLE `questions` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '题目ID',
  `homework_id` bigint(20) NOT NULL COMMENT '作业ID',
  `content` text NOT NULL COMMENT '题目内容',
  `question_type` enum('single_choice','multiple_choice','fill_blank','calculation','proof','application') NOT NULL DEFAULT 'single_choice' COMMENT '题目类型',
  `options` json DEFAULT NULL COMMENT '选择题选项(JSON格式)',
  `correct_answer` text NOT NULL COMMENT '正确答案',
  `score` decimal(5,2) NOT NULL DEFAULT '0.00' COMMENT '题目分值',
  `difficulty` int(11) NOT NULL DEFAULT '1' COMMENT '难度等级(1-5)',
  `order_index` int(11) NOT NULL DEFAULT '1' COMMENT '题目顺序',
  `knowledge_points` json DEFAULT NULL COMMENT '关联知识点ID列表',
  `explanation` text DEFAULT NULL COMMENT '题目解析',
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_homework_id` (`homework_id`),
  KEY `idx_question_type` (`question_type`),
  KEY `idx_difficulty` (`difficulty`),
  CONSTRAINT `fk_questions_homework_id` FOREIGN KEY (`homework_id`) REFERENCES `homeworks` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='题目表';

-- 插入题目数据
INSERT INTO questions (id, homework_id, content, question_type, options, correct_answer, score, difficulty, order_index, knowledge_points, explanation) VALUES
-- 有理数运算题目 (作业1,2)
(1, 1, '计算：(-3) + 5 - 2 = ?', 'single_choice', '["0", "2", "4", "-10"]', '0', 25, 2, 1, '[1]', '请根据有理数运算法则计算。'),
(2, 1, '计算：(-2) × 3 ÷ (-1) = ?', 'fill_blank', NULL, '6', 25, 2, 2, '[1]', '请根据有理数运算法则计算。'),
(3, 2, '计算：(-3) + 5 - 2 = ?', 'single_choice', '["0", "2", "4", "-10"]', '0', 25, 2, 1, '[1]', '请根据有理数运算法则计算。'),
(4, 2, '计算：(-2) × 3 ÷ (-1) = ?', 'fill_blank', NULL, '6', 25, 2, 2, '[1]', '请根据有理数运算法则计算。'),
-- 代数式题目 (作业3,4)
(5, 3, '化简：3x + 2x - x = ?', 'fill_blank', NULL, '4x', 30, 3, 1, '[2]', '请根据代数式化简法则计算。'),
(6, 3, '当x=2时，代数式2x+1的值是多少？', 'single_choice', '["3", "4", "5", "6"]', '5', 20, 3, 2, '[2]', '请将x=2代入代数式计算。'),
(7, 4, '化简：3x + 2x - x = ?', 'fill_blank', NULL, '4x', 30, 3, 1, '[2]', '请根据代数式化简法则计算。'),
(8, 4, '当x=2时，代数式2x+1的值是多少？', 'single_choice', '["3", "4", "5", "6"]', '5', 20, 3, 2, '[2]', '请将x=2代入代数式计算。'),
-- 几何题目 (作业5,6)
(9, 5, '一个三角形的三个内角分别是60°、60°、60°，这是什么三角形？', 'single_choice', '["直角三角形", "等腰三角形", "等边三角形", "钝角三角形"]', '等边三角形', 25, 2, 1, '[3]', '三个角都是60°的三角形是等边三角形。'),
(10, 5, '正方形有几条对称轴？', 'fill_blank', NULL, '4', 25, 2, 2, '[3]', '正方形有4条对称轴。'),
(11, 6, '一个三角形的三个内角分别是60°、60°、60°，这是什么三角形？', 'single_choice', '["直角三角形", "等腰三角形", "等边三角形", "钝角三角形"]', '等边三角形', 25, 2, 1, '[3]', '三个角都是60°的三角形是等边三角形。'),
(12, 6, '正方形有几条对称轴？', 'fill_blank', NULL, '4', 25, 2, 2, '[3]', '正方形有4条对称轴。');

-- 10. 插入作业分配数据
INSERT IGNORE INTO homework_assignments (id, homework_id, assigned_to_type, assigned_to_id, assigned_by, assigned_at, due_date_override, is_active) VALUES
-- 王老师的作业分配
(1, 1, 'class', 1, 10, NOW(), DATE_ADD(NOW(), INTERVAL 7 DAY), 1),
(2, 2, 'class', 2, 10, NOW(), DATE_ADD(NOW(), INTERVAL 7 DAY), 1),
-- 李老师的作业分配
(3, 3, 'class', 1, 11, NOW(), DATE_ADD(NOW(), INTERVAL 7 DAY), 1),
(4, 4, 'class', 2, 11, NOW(), DATE_ADD(NOW(), INTERVAL 7 DAY), 1),
-- 张老师的作业分配
(5, 5, 'class', 1, 12, NOW(), DATE_ADD(NOW(), INTERVAL 7 DAY), 1),
(6, 6, 'class', 2, 12, NOW(), DATE_ADD(NOW(), INTERVAL 7 DAY), 1);

-- 11. 插入学生提交数据
INSERT IGNORE INTO homework_submissions (id, assignment_id, student_id, answers, score, time_spent, status, submitted_at, graded_at, graded_by) VALUES
-- 1班学生提交 (assignment_id: 1,3,5)
-- 小明的提交
(1, 1, 20, '{"1": "0", "2": "6"}', 50, 35, 'graded', DATE_SUB(NOW(), INTERVAL 2 DAY), DATE_SUB(NOW(), INTERVAL 1 DAY), 10),
(2, 3, 20, '{"5": "4x", "6": "5"}', 50, 28, 'graded', DATE_SUB(NOW(), INTERVAL 3 DAY), DATE_SUB(NOW(), INTERVAL 2 DAY), 11),
(3, 5, 20, '{"9": "等边三角形", "10": "4"}', 50, 22, 'graded', DATE_SUB(NOW(), INTERVAL 1 DAY), NOW(), 12),
-- 小红的提交
(4, 1, 21, '{"1": "0", "2": "错误答案"}', 25, 42, 'graded', DATE_SUB(NOW(), INTERVAL 2 DAY), DATE_SUB(NOW(), INTERVAL 1 DAY), 10),
(5, 3, 21, '{"5": "4x", "6": "5"}', 50, 31, 'graded', DATE_SUB(NOW(), INTERVAL 3 DAY), DATE_SUB(NOW(), INTERVAL 2 DAY), 11),
(6, 5, 21, '{"9": "等边三角形", "10": "4"}', 50, 26, 'graded', DATE_SUB(NOW(), INTERVAL 1 DAY), NOW(), 12),
-- 小刚的提交
(7, 1, 22, '{"1": "2", "2": "6"}', 25, 38, 'graded', DATE_SUB(NOW(), INTERVAL 2 DAY), DATE_SUB(NOW(), INTERVAL 1 DAY), 10),
(8, 3, 22, '{"5": "错误答案", "6": "5"}', 20, 45, 'graded', DATE_SUB(NOW(), INTERVAL 3 DAY), DATE_SUB(NOW(), INTERVAL 2 DAY), 11),
(9, 5, 22, '{"9": "等边三角形", "10": "错误答案"}', 25, 33, 'graded', DATE_SUB(NOW(), INTERVAL 1 DAY), NOW(), 12),

-- 2班学生提交 (assignment_id: 2,4,6)
-- 小华的提交
(10, 2, 23, '{"3": "0", "4": "6"}', 50, 29, 'graded', DATE_SUB(NOW(), INTERVAL 2 DAY), DATE_SUB(NOW(), INTERVAL 1 DAY), 10),
(11, 4, 23, '{"7": "4x", "8": "5"}', 50, 34, 'graded', DATE_SUB(NOW(), INTERVAL 3 DAY), DATE_SUB(NOW(), INTERVAL 2 DAY), 11),
(12, 6, 23, '{"11": "等边三角形", "12": "4"}', 50, 27, 'graded', DATE_SUB(NOW(), INTERVAL 1 DAY), NOW(), 12),
-- 小丽的提交
(13, 2, 24, '{"3": "0", "4": "6"}', 50, 36, 'graded', DATE_SUB(NOW(), INTERVAL 2 DAY), DATE_SUB(NOW(), INTERVAL 1 DAY), 10),
(14, 4, 24, '{"7": "4x", "8": "错误答案"}', 30, 41, 'graded', DATE_SUB(NOW(), INTERVAL 3 DAY), DATE_SUB(NOW(), INTERVAL 2 DAY), 11),
(15, 6, 24, '{"11": "等边三角形", "12": "4"}', 50, 24, 'graded', DATE_SUB(NOW(), INTERVAL 1 DAY), NOW(), 12),
-- 小强的提交
(16, 2, 25, '{"3": "错误答案", "4": "6"}', 25, 47, 'graded', DATE_SUB(NOW(), INTERVAL 2 DAY), DATE_SUB(NOW(), INTERVAL 1 DAY), 10),
(17, 4, 25, '{"7": "4x", "8": "5"}', 50, 32, 'graded', DATE_SUB(NOW(), INTERVAL 3 DAY), DATE_SUB(NOW(), INTERVAL 2 DAY), 11),
(18, 6, 25, '{"11": "直角三角形", "12": "4"}', 25, 39, 'graded', DATE_SUB(NOW(), INTERVAL 1 DAY), NOW(), 12);

-- 12. 插入练习题数据
INSERT IGNORE INTO exercises (id, title, content, question_type, options, correct_answer, difficulty, subject, grade, knowledge_points, explanation, created_at, updated_at) VALUES
(1, '有理数加法练习', '计算：(-5) + 3 = ?', 'single_choice', '["2", "-2", "8", "-8"]', '-2', 2, '数学', 7, '[1]', '负数加正数，绝对值大的数决定符号。', NOW(), NOW()),
(2, '代数式求值练习', '当a=3时，2a-1的值是？', 'fill_blank', NULL, '5', 3, '数学', 7, '[2]', '将a=3代入2a-1得到2×3-1=5。', NOW(), NOW()),
(3, '角度计算练习', '直角等于多少度？', 'single_choice', '["45°", "60°", "90°", "180°"]', '90°', 1, '数学', 7, '[3]', '直角等于90度。', NOW(), NOW());

-- 数据插入完成
SELECT '✅ 测试数据插入完成！' as message;

-- 显示统计信息
SELECT
    '📊 数据统计' as info,
    (SELECT COUNT(*) FROM users WHERE role='teacher') as 教师数量,
    (SELECT COUNT(*) FROM users WHERE role='student') as 学生数量,
    (SELECT COUNT(*) FROM schools) as 学校数量,
    (SELECT COUNT(*) FROM classes) as 班级数量,
    (SELECT COUNT(*) FROM homeworks) as 作业数量,
    (SELECT COUNT(*) FROM questions) as 题目数量,
    (SELECT COUNT(*) FROM homework_assignments) as 作业分配数量,
    (SELECT COUNT(*) FROM homework_submissions) as 学生提交数量,
    (SELECT COUNT(*) FROM knowledge_points) as 知识点数量,
    (SELECT COUNT(*) FROM exercises) as 练习题数量;
