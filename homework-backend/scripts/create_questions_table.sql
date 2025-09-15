-- 创建questions表和相关数据
USE homework_system;

-- 删除不需要的测试表
DROP TABLE IF EXISTS `simple_questions`;
DROP TABLE IF EXISTS `test_questions`;

-- 确保questions表存在
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

-- 插入测试数据
-- 首先确保有作业数据
INSERT IGNORE INTO homeworks (id, title, subject, description, grade, difficulty_level, created_by, due_date, start_date, time_limit, max_score, max_attempts, is_published, auto_grade, show_answers, instructions, tags, category, created_at, updated_at) VALUES
(1, '有理数运算练习 - 七年级1班', '数学', '练习有理数的加减乘除运算', 7, 2, 10, DATE_ADD(NOW(), INTERVAL 7 DAY), NOW(), 60, 50, 2, 1, 1, 0, '请仔细阅读题目，认真作答。', '["基础", "练习"]', '课后练习', NOW(), NOW()),
(2, '有理数运算练习 - 七年级2班', '数学', '练习有理数的加减乘除运算', 7, 2, 10, DATE_ADD(NOW(), INTERVAL 7 DAY), NOW(), 60, 50, 2, 1, 1, 0, '请仔细阅读题目，认真作答。', '["基础", "练习"]', '课后练习', NOW(), NOW()),
(3, '代数式化简 - 七年级1班', '数学', '学习代数式的化简方法', 7, 3, 11, DATE_ADD(NOW(), INTERVAL 7 DAY), NOW(), 60, 50, 2, 1, 1, 0, '请仔细阅读题目，认真作答。', '["基础", "练习"]', '课后练习', NOW(), NOW()),
(4, '代数式化简 - 七年级2班', '数学', '学习代数式的化简方法', 7, 3, 11, DATE_ADD(NOW(), INTERVAL 7 DAY), NOW(), 60, 50, 2, 1, 1, 0, '请仔细阅读题目，认真作答。', '["基础", "练习"]', '课后练习', NOW(), NOW()),
(5, '几何图形认识 - 七年级1班', '数学', '认识基本的几何图形', 7, 2, 12, DATE_ADD(NOW(), INTERVAL 7 DAY), NOW(), 60, 50, 2, 1, 1, 0, '请仔细阅读题目，认真作答。', '["基础", "练习"]', '课后练习', NOW(), NOW()),
(6, '几何图形认识 - 七年级2班', '数学', '认识基本的几何图形', 7, 2, 12, DATE_ADD(NOW(), INTERVAL 7 DAY), NOW(), 60, 50, 2, 1, 1, 0, '请仔细阅读题目，认真作答。', '["基础", "练习"]', '课后练习', NOW(), NOW());

-- 插入题目数据
INSERT IGNORE INTO questions (id, homework_id, content, question_type, options, correct_answer, score, difficulty, order_index, knowledge_points, explanation) VALUES
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

-- 显示结果
SELECT '✅ questions表创建完成，测试题目数据插入完成！' as message;
SELECT COUNT(*) as '题目总数' FROM questions;
SELECT homework_id, COUNT(*) as '题目数量' FROM questions GROUP BY homework_id;
