-- 创建评分相关表
USE homework_system;

-- 评分结果表
DROP TABLE IF EXISTS `grading_results`;
CREATE TABLE `grading_results` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '评分结果ID',
  `submission_id` bigint(20) NOT NULL COMMENT '提交记录ID',
  `result_data` json NOT NULL COMMENT '详细评分结果(JSON格式)',
  `total_score` decimal(5,2) NOT NULL DEFAULT '0.00' COMMENT '总得分',
  `total_possible` decimal(5,2) NOT NULL DEFAULT '0.00' COMMENT '总分',
  `accuracy` decimal(5,2) NOT NULL DEFAULT '0.00' COMMENT '正确率(%)',
  `grading_method` enum('auto','manual','hybrid') NOT NULL DEFAULT 'auto' COMMENT '评分方式',
  `graded_by` bigint(20) DEFAULT NULL COMMENT '评分教师ID',
  `graded_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '评分时间',
  `reviewed_at` timestamp NULL DEFAULT NULL COMMENT '复核时间',
  `review_notes` text DEFAULT NULL COMMENT '复核备注',
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_submission_id` (`submission_id`),
  KEY `idx_grading_method` (`grading_method`),
  KEY `idx_graded_at` (`graded_at`),
  KEY `idx_accuracy` (`accuracy`),
  CONSTRAINT `fk_grading_results_submission` FOREIGN KEY (`submission_id`) REFERENCES `homework_submissions` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='评分结果表';

-- 评分规则配置表
DROP TABLE IF EXISTS `grading_rules`;
CREATE TABLE `grading_rules` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '规则ID',
  `homework_id` bigint(20) NOT NULL COMMENT '作业ID',
  `question_type` enum('single_choice','multiple_choice','fill_blank','calculation','proof','application') NOT NULL COMMENT '题目类型',
  `rule_config` json NOT NULL COMMENT '规则配置(JSON格式)',
  `is_active` tinyint(1) NOT NULL DEFAULT '1' COMMENT '是否启用',
  `created_by` bigint(20) NOT NULL COMMENT '创建者ID',
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_homework_id` (`homework_id`),
  KEY `idx_question_type` (`question_type`),
  KEY `idx_is_active` (`is_active`),
  CONSTRAINT `fk_grading_rules_homework` FOREIGN KEY (`homework_id`) REFERENCES `homeworks` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='评分规则配置表';

-- 错误类型分析表
DROP TABLE IF EXISTS `error_analysis`;
CREATE TABLE `error_analysis` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '分析ID',
  `submission_id` bigint(20) NOT NULL COMMENT '提交记录ID',
  `question_id` bigint(20) NOT NULL COMMENT '题目ID',
  `error_type` varchar(50) NOT NULL COMMENT '错误类型',
  `error_description` text DEFAULT NULL COMMENT '错误描述',
  `student_answer` text DEFAULT NULL COMMENT '学生答案',
  `correct_answer` text DEFAULT NULL COMMENT '正确答案',
  `suggestions` json DEFAULT NULL COMMENT '改进建议',
  `confidence_score` decimal(3,2) DEFAULT '0.00' COMMENT '分析置信度(0-1)',
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`id`),
  KEY `idx_submission_id` (`submission_id`),
  KEY `idx_question_id` (`question_id`),
  KEY `idx_error_type` (`error_type`),
  CONSTRAINT `fk_error_analysis_submission` FOREIGN KEY (`submission_id`) REFERENCES `homework_submissions` (`id`) ON DELETE CASCADE,
  CONSTRAINT `fk_error_analysis_question` FOREIGN KEY (`question_id`) REFERENCES `questions` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='错误类型分析表';

-- 评分统计表
DROP TABLE IF EXISTS `grading_statistics`;
CREATE TABLE `grading_statistics` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '统计ID',
  `homework_id` bigint(20) NOT NULL COMMENT '作业ID',
  `question_id` bigint(20) DEFAULT NULL COMMENT '题目ID(NULL表示整个作业)',
  `total_submissions` int(11) NOT NULL DEFAULT '0' COMMENT '总提交数',
  `correct_count` int(11) NOT NULL DEFAULT '0' COMMENT '正确数量',
  `average_score` decimal(5,2) NOT NULL DEFAULT '0.00' COMMENT '平均分',
  `accuracy_rate` decimal(5,2) NOT NULL DEFAULT '0.00' COMMENT '正确率(%)',
  `common_errors` json DEFAULT NULL COMMENT '常见错误统计',
  `difficulty_analysis` json DEFAULT NULL COMMENT '难度分析',
  `last_updated` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '最后更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_homework_question` (`homework_id`, `question_id`),
  KEY `idx_homework_id` (`homework_id`),
  KEY `idx_question_id` (`question_id`),
  KEY `idx_accuracy_rate` (`accuracy_rate`),
  CONSTRAINT `fk_grading_stats_homework` FOREIGN KEY (`homework_id`) REFERENCES `homeworks` (`id`) ON DELETE CASCADE,
  CONSTRAINT `fk_grading_stats_question` FOREIGN KEY (`question_id`) REFERENCES `questions` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='评分统计表';

-- 插入默认评分规则
INSERT INTO grading_rules (homework_id, question_type, rule_config, created_by) VALUES
(1, 'single_choice', '{"exact_match": true, "case_sensitive": false, "partial_credit": false}', 10),
(1, 'fill_blank', '{"exact_match": false, "fuzzy_threshold": 0.8, "partial_credit": true, "partial_ratio": 0.5}', 10),
(2, 'single_choice', '{"exact_match": true, "case_sensitive": false, "partial_credit": false}', 10),
(2, 'fill_blank', '{"exact_match": false, "fuzzy_threshold": 0.8, "partial_credit": true, "partial_ratio": 0.5}', 10),
(3, 'fill_blank', '{"exact_match": false, "fuzzy_threshold": 0.8, "partial_credit": true, "partial_ratio": 0.5}', 11),
(3, 'single_choice', '{"exact_match": true, "case_sensitive": false, "partial_credit": false}', 11),
(4, 'fill_blank', '{"exact_match": false, "fuzzy_threshold": 0.8, "partial_credit": true, "partial_ratio": 0.5}', 11),
(4, 'single_choice', '{"exact_match": true, "case_sensitive": false, "partial_credit": false}', 11),
(5, 'single_choice', '{"exact_match": true, "case_sensitive": false, "partial_credit": false}', 12),
(5, 'fill_blank', '{"exact_match": false, "fuzzy_threshold": 0.8, "partial_credit": true, "partial_ratio": 0.5}', 12),
(6, 'single_choice', '{"exact_match": true, "case_sensitive": false, "partial_credit": false}', 12),
(6, 'fill_blank', '{"exact_match": false, "fuzzy_threshold": 0.8, "partial_credit": true, "partial_ratio": 0.5}', 12);

-- 显示创建结果
SELECT '✅ 评分相关表创建完成！' as message;
SELECT COUNT(*) as '评分规则数量' FROM grading_rules;
