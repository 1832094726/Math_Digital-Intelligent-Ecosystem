<template>
  <div class="homework-feedback">
    <!-- 加载状态 -->
    <div v-if="loading" class="loading-container">
      <div class="loading-spinner"></div>
      <p>正在加载反馈信息...</p>
    </div>

    <!-- 反馈内容 -->
    <div v-else-if="feedbackData" class="feedback-content">
      <!-- 作业信息头部 -->
      <div class="feedback-header">
        <h2>{{ feedbackData.homework_info.title }}</h2>
        <div class="homework-meta">
          <span class="subject">{{ feedbackData.homework_info.subject }}</span>
          <span class="grade">{{ feedbackData.homework_info.grade_level }}</span>
        </div>
      </div>

      <!-- 成绩概览 -->
      <div class="score-overview">
        <div class="score-card main-score">
          <div class="score-value">{{ feedbackData.personal_performance.total_score }}</div>
          <div class="score-max">/ {{ feedbackData.personal_performance.max_score }}</div>
          <div class="score-percentage">{{ feedbackData.personal_performance.percentage }}%</div>
          <div class="score-label">我的得分</div>
        </div>

        <div class="score-card class-stats">
          <div class="stat-item">
            <span class="stat-label">班级平均分</span>
            <span class="stat-value">{{ feedbackData.class_statistics.class_average }}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">班级排名</span>
            <span class="stat-value">{{ feedbackData.class_statistics.student_rank }}/{{ feedbackData.class_statistics.total_students }}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">超越同学</span>
            <span class="stat-value">{{ feedbackData.class_statistics.percentile }}%</span>
          </div>
        </div>
      </div>

      <!-- 学习建议 -->
      <div class="learning-suggestions">
        <h3>学习建议</h3>
        <div class="suggestions-list">
          <div 
            v-for="suggestion in feedbackData.learning_suggestions" 
            :key="suggestion.type"
            :class="['suggestion-item', `priority-${suggestion.priority}`]"
          >
            <div class="suggestion-icon">
              <i :class="getSuggestionIcon(suggestion.type)"></i>
            </div>
            <div class="suggestion-content">
              <h4>{{ suggestion.title }}</h4>
              <p>{{ suggestion.content }}</p>
            </div>
          </div>
        </div>
      </div>

      <!-- 题目详细反馈 -->
      <div class="question-feedback">
        <h3>题目详细反馈</h3>
        <div class="questions-list">
          <div 
            v-for="question in feedbackData.question_feedback" 
            :key="question.question_id"
            :class="['question-item', question.is_correct ? 'correct' : 'incorrect']"
          >
            <div class="question-header">
              <span class="question-number">第{{ question.question_order }}题</span>
              <span class="question-type">{{ getQuestionTypeLabel(question.question_type) }}</span>
              <span class="question-score">{{ question.score_earned }}/{{ question.max_score }}分</span>
              <span :class="['question-status', question.is_correct ? 'correct' : 'incorrect']">
                {{ question.is_correct ? '✓ 正确' : '✗ 错误' }}
              </span>
            </div>

            <div class="question-content">
              <div class="question-text">{{ question.question_content }}</div>
              
              <div class="answer-comparison">
                <div class="student-answer">
                  <label>你的答案：</label>
                  <div class="answer-text">{{ question.student_answer || '未作答' }}</div>
                </div>
                
                <div v-if="!question.is_correct && question.correct_answer" class="correct-answer">
                  <label>参考答案：</label>
                  <div class="answer-text">{{ question.correct_answer }}</div>
                </div>
              </div>

              <div v-if="question.explanation" class="explanation">
                <label>解题思路：</label>
                <div class="explanation-text">{{ question.explanation }}</div>
              </div>

              <div v-if="question.error_analysis && !question.is_correct" class="error-analysis">
                <label>错误分析：</label>
                <div class="error-types">
                  <span 
                    v-for="errorType in question.error_analysis.error_types" 
                    :key="errorType"
                    class="error-tag"
                  >
                    {{ errorType }}
                  </span>
                </div>
                <div class="error-suggestions">
                  <ul>
                    <li v-for="suggestion in question.error_analysis.suggestions" :key="suggestion">
                      {{ suggestion }}
                    </li>
                  </ul>
                </div>
              </div>

              <div v-if="question.knowledge_points.length > 0" class="knowledge-points">
                <label>相关知识点：</label>
                <div class="knowledge-tags">
                  <span 
                    v-for="point in question.knowledge_points" 
                    :key="point"
                    class="knowledge-tag"
                  >
                    {{ point }}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 错误分析总结 -->
      <div v-if="feedbackData.error_analysis && Object.keys(feedbackData.error_analysis).length > 0" class="error-summary">
        <h3>错误分析总结</h3>
        <div class="error-overview">
          <div class="error-rate">
            <span class="label">错误率：</span>
            <span class="value">{{ feedbackData.error_analysis.error_rate }}%</span>
          </div>
          <div class="performance-level">
            <span class="label">整体表现：</span>
            <span :class="['value', feedbackData.error_analysis.overall_performance]">
              {{ getPerformanceLabel(feedbackData.error_analysis.overall_performance) }}
            </span>
          </div>
        </div>

        <div v-if="feedbackData.error_analysis.main_issues.length > 0" class="main-issues">
          <h4>主要问题</h4>
          <ul>
            <li v-for="issue in feedbackData.error_analysis.main_issues" :key="issue">
              {{ issue }}
            </li>
          </ul>
        </div>

        <div v-if="feedbackData.error_analysis.improvement_areas.length > 0" class="improvement-areas">
          <h4>改进建议</h4>
          <ul>
            <li v-for="area in feedbackData.error_analysis.improvement_areas" :key="area">
              {{ area }}
            </li>
          </ul>
        </div>
      </div>

      <!-- 操作按钮 -->
      <div class="feedback-actions">
        <button @click="printFeedback" class="btn btn-secondary">
          <i class="fas fa-print"></i> 打印反馈
        </button>
        <button @click="shareFeedback" class="btn btn-primary">
          <i class="fas fa-share"></i> 分享反馈
        </button>
        <button @click="$emit('close')" class="btn btn-outline">
          <i class="fas fa-times"></i> 关闭
        </button>
      </div>
    </div>

    <!-- 错误状态 -->
    <div v-else-if="error" class="error-container">
      <div class="error-icon">⚠️</div>
      <h3>加载失败</h3>
      <p>{{ error }}</p>
      <button @click="loadFeedback" class="btn btn-primary">重试</button>
    </div>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'HomeworkFeedback',
  props: {
    homeworkId: {
      type: [String, Number],
      required: true
    }
  },
  data() {
    return {
      loading: false,
      feedbackData: null,
      error: null
    }
  },
  mounted() {
    this.loadFeedback()
  },
  methods: {
    async loadFeedback() {
      this.loading = true
      this.error = null
      
      try {
        const response = await axios.get(`/api/feedback/homework/${this.homeworkId}`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        })
        
        if (response.data.success) {
          this.feedbackData = response.data.data
        } else {
          this.error = response.data.message || '获取反馈失败'
        }
      } catch (error) {
        console.error('加载反馈失败:', error)
        this.error = error.response?.data?.message || '网络错误，请稍后重试'
      } finally {
        this.loading = false
      }
    },

    getSuggestionIcon(type) {
      const icons = {
        encouragement: 'fas fa-star',
        improvement: 'fas fa-arrow-up',
        remediation: 'fas fa-book',
        comparison: 'fas fa-chart-line',
        knowledge_review: 'fas fa-graduation-cap'
      }
      return icons[type] || 'fas fa-lightbulb'
    },

    getQuestionTypeLabel(type) {
      const labels = {
        choice: '选择题',
        fill_blank: '填空题',
        calculation: '计算题',
        application: '应用题',
        proof: '证明题'
      }
      return labels[type] || type
    },

    getPerformanceLabel(performance) {
      const labels = {
        excellent: '优秀',
        good: '良好',
        needs_improvement: '需要改进'
      }
      return labels[performance] || performance
    },

    printFeedback() {
      window.print()
    },

    async shareFeedback() {
      try {
        const response = await axios.post(`/api/feedback/homework/${this.homeworkId}/share`, {
          type: 'link'
        }, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        })
        
        if (response.data.success) {
          // 复制分享链接到剪贴板
          const shareUrl = window.location.origin + response.data.share_url
          await navigator.clipboard.writeText(shareUrl)
          alert('分享链接已复制到剪贴板')
        } else {
          alert('分享失败：' + response.data.message)
        }
      } catch (error) {
        console.error('分享失败:', error)
        alert('分享失败，请稍后重试')
      }
    }
  }
}
</script>

<style scoped>
.homework-feedback {
  max-width: 1000px;
  margin: 0 auto;
  padding: 20px;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.loading-container, .error-container {
  text-align: center;
  padding: 60px 20px;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #007bff;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 20px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.feedback-header {
  text-align: center;
  margin-bottom: 30px;
  padding-bottom: 20px;
  border-bottom: 2px solid #eee;
}

.feedback-header h2 {
  color: #333;
  margin-bottom: 10px;
}

.homework-meta {
  display: flex;
  justify-content: center;
  gap: 20px;
}

.homework-meta span {
  padding: 4px 12px;
  background: #f8f9fa;
  border-radius: 20px;
  font-size: 14px;
  color: #666;
}

.score-overview {
  display: grid;
  grid-template-columns: 1fr 2fr;
  gap: 30px;
  margin-bottom: 30px;
}

.score-card {
  padding: 20px;
  border-radius: 8px;
  background: #f8f9fa;
}

.main-score {
  text-align: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.score-value {
  font-size: 48px;
  font-weight: bold;
  display: inline-block;
}

.score-max {
  font-size: 24px;
  opacity: 0.8;
  display: inline-block;
}

.score-percentage {
  font-size: 20px;
  margin: 10px 0;
  opacity: 0.9;
}

.score-label {
  font-size: 16px;
  opacity: 0.8;
}

.class-stats .stat-item {
  display: flex;
  justify-content: space-between;
  margin-bottom: 10px;
  padding: 8px 0;
  border-bottom: 1px solid #eee;
}

.stat-label {
  color: #666;
}

.stat-value {
  font-weight: bold;
  color: #333;
}

.learning-suggestions, .question-feedback, .error-summary {
  margin-bottom: 30px;
}

.learning-suggestions h3, .question-feedback h3, .error-summary h3 {
  color: #333;
  margin-bottom: 20px;
  padding-bottom: 10px;
  border-bottom: 2px solid #007bff;
}

.suggestion-item {
  display: flex;
  align-items: flex-start;
  padding: 15px;
  margin-bottom: 15px;
  border-radius: 8px;
  border-left: 4px solid #007bff;
}

.suggestion-item.priority-high {
  border-left-color: #dc3545;
  background: #fff5f5;
}

.suggestion-item.priority-medium {
  border-left-color: #ffc107;
  background: #fffbf0;
}

.suggestion-item.priority-low {
  border-left-color: #28a745;
  background: #f8fff8;
}

.suggestion-icon {
  margin-right: 15px;
  font-size: 20px;
  color: #007bff;
}

.suggestion-content h4 {
  margin: 0 0 8px 0;
  color: #333;
}

.suggestion-content p {
  margin: 0;
  color: #666;
  line-height: 1.5;
}

.question-item {
  border: 1px solid #ddd;
  border-radius: 8px;
  margin-bottom: 20px;
  overflow: hidden;
}

.question-item.correct {
  border-left: 4px solid #28a745;
}

.question-item.incorrect {
  border-left: 4px solid #dc3545;
}

.question-header {
  display: flex;
  align-items: center;
  gap: 15px;
  padding: 15px 20px;
  background: #f8f9fa;
  border-bottom: 1px solid #ddd;
}

.question-number {
  font-weight: bold;
  color: #333;
}

.question-type {
  padding: 2px 8px;
  background: #e9ecef;
  border-radius: 12px;
  font-size: 12px;
  color: #666;
}

.question-score {
  font-weight: bold;
  color: #007bff;
}

.question-status.correct {
  color: #28a745;
  font-weight: bold;
}

.question-status.incorrect {
  color: #dc3545;
  font-weight: bold;
}

.question-content {
  padding: 20px;
}

.question-text {
  margin-bottom: 15px;
  line-height: 1.6;
  color: #333;
}

.answer-comparison {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin-bottom: 15px;
}

.student-answer, .correct-answer {
  padding: 10px;
  border-radius: 4px;
}

.student-answer {
  background: #f8f9fa;
}

.correct-answer {
  background: #e8f5e8;
}

.answer-comparison label {
  font-weight: bold;
  color: #666;
  font-size: 14px;
  display: block;
  margin-bottom: 5px;
}

.answer-text {
  color: #333;
  min-height: 20px;
}

.explanation, .error-analysis, .knowledge-points {
  margin-top: 15px;
  padding-top: 15px;
  border-top: 1px solid #eee;
}

.explanation label, .error-analysis label, .knowledge-points label {
  font-weight: bold;
  color: #666;
  font-size: 14px;
  display: block;
  margin-bottom: 8px;
}

.error-tag, .knowledge-tag {
  display: inline-block;
  padding: 2px 8px;
  margin: 2px 4px 2px 0;
  background: #fff3cd;
  color: #856404;
  border-radius: 12px;
  font-size: 12px;
}

.knowledge-tag {
  background: #d1ecf1;
  color: #0c5460;
}

.error-suggestions ul {
  margin: 10px 0 0 0;
  padding-left: 20px;
}

.error-suggestions li {
  margin-bottom: 5px;
  color: #666;
}

.error-summary .error-overview {
  display: flex;
  gap: 30px;
  margin-bottom: 20px;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 8px;
}

.error-rate .value, .performance-level .value {
  font-weight: bold;
  margin-left: 5px;
}

.performance-level .value.excellent {
  color: #28a745;
}

.performance-level .value.good {
  color: #007bff;
}

.performance-level .value.needs_improvement {
  color: #dc3545;
}

.main-issues, .improvement-areas {
  margin-bottom: 20px;
}

.main-issues h4, .improvement-areas h4 {
  color: #333;
  margin-bottom: 10px;
}

.main-issues ul, .improvement-areas ul {
  margin: 0;
  padding-left: 20px;
}

.main-issues li, .improvement-areas li {
  margin-bottom: 8px;
  color: #666;
  line-height: 1.5;
}

.feedback-actions {
  display: flex;
  justify-content: center;
  gap: 15px;
  padding-top: 30px;
  border-top: 2px solid #eee;
}

.btn {
  padding: 10px 20px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  transition: all 0.3s ease;
}

.btn-primary {
  background: #007bff;
  color: white;
}

.btn-primary:hover {
  background: #0056b3;
}

.btn-secondary {
  background: #6c757d;
  color: white;
}

.btn-secondary:hover {
  background: #545b62;
}

.btn-outline {
  background: transparent;
  color: #666;
  border: 1px solid #ddd;
}

.btn-outline:hover {
  background: #f8f9fa;
}

@media (max-width: 768px) {
  .score-overview {
    grid-template-columns: 1fr;
  }
  
  .answer-comparison {
    grid-template-columns: 1fr;
  }
  
  .question-header {
    flex-wrap: wrap;
    gap: 10px;
  }
  
  .error-overview {
    flex-direction: column;
    gap: 15px;
  }
  
  .feedback-actions {
    flex-direction: column;
    align-items: center;
  }
}
</style>
