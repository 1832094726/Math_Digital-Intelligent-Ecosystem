<template>
  <div class="homework-analytics">
    <!-- 加载状态 -->
    <div v-if="loading" class="loading-container">
      <div class="loading-spinner"></div>
      <p>正在生成分析报告...</p>
    </div>

    <!-- 分析报告内容 -->
    <div v-else-if="analyticsData" class="analytics-content">
      <!-- 报告头部 -->
      <div class="analytics-header">
        <h2>{{ analyticsData.homework_info.title }} - 分析报告</h2>
        <div class="report-meta">
          <span class="subject">{{ analyticsData.homework_info.subject }}</span>
          <span class="grade">{{ analyticsData.homework_info.grade_level }}</span>
          <span class="generated-time">生成时间：{{ formatDateTime(analyticsData.generated_at) }}</span>
        </div>
      </div>

      <!-- 基础统计概览 -->
      <div class="statistics-overview">
        <h3>基础统计</h3>
        <div class="stats-grid">
          <div class="stat-card">
            <div class="stat-value">{{ analyticsData.basic_statistics.total_assignments }}</div>
            <div class="stat-label">总分配数</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{{ analyticsData.basic_statistics.completed_count }}</div>
            <div class="stat-label">已完成</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{{ analyticsData.basic_statistics.completion_rate }}%</div>
            <div class="stat-label">完成率</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{{ analyticsData.basic_statistics.average_score }}</div>
            <div class="stat-label">平均分</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{{ analyticsData.basic_statistics.average_completion_time }}min</div>
            <div class="stat-label">平均用时</div>
          </div>
        </div>
      </div>

      <!-- 分数分布分析 -->
      <div class="score-distribution">
        <h3>分数分布分析</h3>
        <div class="distribution-chart">
          <div class="chart-container">
            <canvas ref="scoreChart" width="400" height="200"></canvas>
          </div>
          <div class="distribution-stats">
            <div class="stat-item">
              <span class="label">平均分：</span>
              <span class="value">{{ analyticsData.score_distribution.statistics.mean }}</span>
            </div>
            <div class="stat-item">
              <span class="label">中位数：</span>
              <span class="value">{{ analyticsData.score_distribution.statistics.median }}</span>
            </div>
            <div class="stat-item">
              <span class="label">标准差：</span>
              <span class="value">{{ analyticsData.score_distribution.statistics.std_dev }}</span>
            </div>
            <div class="stat-item">
              <span class="label">参与人数：</span>
              <span class="value">{{ analyticsData.score_distribution.statistics.total_students }}</span>
            </div>
          </div>
        </div>
        
        <div class="grade-distribution">
          <div 
            v-for="grade in analyticsData.score_distribution.distribution" 
            :key="grade.range"
            class="grade-bar"
          >
            <div class="grade-label">{{ grade.label }}</div>
            <div class="grade-progress">
              <div 
                class="grade-fill" 
                :style="{ width: grade.percentage + '%' }"
                :class="getGradeClass(grade.label)"
              ></div>
            </div>
            <div class="grade-stats">
              <span class="count">{{ grade.count }}人</span>
              <span class="percentage">{{ grade.percentage }}%</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 题目分析 -->
      <div class="question-analysis">
        <h3>题目分析</h3>
        <div class="questions-table">
          <table>
            <thead>
              <tr>
                <th>题号</th>
                <th>题型</th>
                <th>正确率</th>
                <th>平均得分</th>
                <th>难度等级</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              <tr 
                v-for="question in analyticsData.question_analysis" 
                :key="question.question_id"
                :class="getDifficultyClass(question.difficulty_level)"
              >
                <td>{{ question.question_order }}</td>
                <td>{{ getQuestionTypeLabel(question.question_type) }}</td>
                <td>
                  <div class="correct-rate">
                    <span class="rate-value">{{ question.correct_rate }}%</span>
                    <div class="rate-bar">
                      <div 
                        class="rate-fill" 
                        :style="{ width: question.correct_rate + '%' }"
                      ></div>
                    </div>
                  </div>
                </td>
                <td>{{ question.average_score }}/{{ question.max_score }}</td>
                <td>
                  <span :class="['difficulty-badge', question.difficulty_level]">
                    {{ getDifficultyLabel(question.difficulty_level) }}
                  </span>
                </td>
                <td>
                  <button 
                    @click="viewQuestionDetail(question)" 
                    class="btn btn-sm btn-outline"
                  >
                    查看详情
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- 知识点掌握度分析 -->
      <div class="knowledge-analysis">
        <h3>知识点掌握度分析</h3>
        <div class="knowledge-grid">
          <div 
            v-for="knowledge in analyticsData.knowledge_analysis" 
            :key="knowledge.knowledge_point"
            :class="['knowledge-card', knowledge.mastery_level]"
          >
            <div class="knowledge-name">{{ knowledge.knowledge_point }}</div>
            <div class="mastery-rate">{{ knowledge.mastery_rate }}%</div>
            <div class="mastery-progress">
              <div 
                class="mastery-fill" 
                :style="{ width: knowledge.mastery_rate + '%' }"
              ></div>
            </div>
            <div class="mastery-label">{{ getMasteryLabel(knowledge.mastery_level) }}</div>
            <div class="attempt-stats">
              {{ knowledge.correct_attempts }}/{{ knowledge.total_attempts }} 正确
            </div>
          </div>
        </div>
      </div>

      <!-- 学生表现分析 -->
      <div class="student-performance">
        <h3>学生表现分析</h3>
        <div class="performance-summary">
          <div class="summary-card">
            <h4>需要关注的学生 ({{ analyticsData.student_performance.struggling_students.length }}人)</h4>
            <div class="student-list">
              <div 
                v-for="student in analyticsData.student_performance.struggling_students" 
                :key="student.student_id"
                class="student-item struggling"
              >
                <span class="student-name">{{ student.name }}</span>
                <span class="student-class">{{ student.class_name }}</span>
                <span class="student-score">{{ student.score_percentage }}%</span>
              </div>
            </div>
          </div>

          <div class="summary-card">
            <h4>表现优秀的学生 ({{ analyticsData.student_performance.excellent_students.length }}人)</h4>
            <div class="student-list">
              <div 
                v-for="student in analyticsData.student_performance.excellent_students" 
                :key="student.student_id"
                class="student-item excellent"
              >
                <span class="student-name">{{ student.name }}</span>
                <span class="student-class">{{ student.class_name }}</span>
                <span class="student-score">{{ student.score_percentage }}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 教学建议 -->
      <div class="teaching-suggestions">
        <h3>教学建议</h3>
        <div class="suggestions-list">
          <div 
            v-for="suggestion in analyticsData.teaching_suggestions" 
            :key="suggestion.type"
            :class="['suggestion-card', `priority-${suggestion.priority}`]"
          >
            <div class="suggestion-header">
              <h4>{{ suggestion.title }}</h4>
              <span :class="['priority-badge', suggestion.priority]">
                {{ getPriorityLabel(suggestion.priority) }}
              </span>
            </div>
            <div class="suggestion-content">
              <p>{{ suggestion.content }}</p>
              <div v-if="suggestion.action_items && suggestion.action_items.length > 0" class="action-items">
                <h5>具体行动：</h5>
                <ul>
                  <li v-for="action in suggestion.action_items" :key="action">
                    {{ action }}
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 操作按钮 -->
      <div class="analytics-actions">
        <button @click="exportReport('pdf')" class="btn btn-primary">
          <i class="fas fa-file-pdf"></i> 导出PDF
        </button>
        <button @click="exportReport('excel')" class="btn btn-secondary">
          <i class="fas fa-file-excel"></i> 导出Excel
        </button>
        <button @click="printReport" class="btn btn-outline">
          <i class="fas fa-print"></i> 打印报告
        </button>
        <button @click="$emit('close')" class="btn btn-outline">
          <i class="fas fa-times"></i> 关闭
        </button>
      </div>
    </div>

    <!-- 错误状态 -->
    <div v-else-if="error" class="error-container">
      <div class="error-icon">⚠️</div>
      <h3>生成报告失败</h3>
      <p>{{ error }}</p>
      <button @click="loadAnalytics" class="btn btn-primary">重试</button>
    </div>

    <!-- 题目详情模态框 -->
    <div v-if="selectedQuestion" class="modal-overlay" @click="closeQuestionDetail">
      <div class="modal-content" @click.stop>
        <div class="modal-header">
          <h4>第{{ selectedQuestion.question_order }}题 详细分析</h4>
          <button @click="closeQuestionDetail" class="close-btn">&times;</button>
        </div>
        <div class="modal-body">
          <div class="question-detail">
            <div class="detail-item">
              <label>题目内容：</label>
              <div class="detail-value">{{ selectedQuestion.question_content }}</div>
            </div>
            <div class="detail-item">
              <label>题目类型：</label>
              <div class="detail-value">{{ getQuestionTypeLabel(selectedQuestion.question_type) }}</div>
            </div>
            <div class="detail-item">
              <label>正确率：</label>
              <div class="detail-value">{{ selectedQuestion.correct_rate }}%</div>
            </div>
            <div class="detail-item">
              <label>平均得分：</label>
              <div class="detail-value">{{ selectedQuestion.average_score }}/{{ selectedQuestion.max_score }}</div>
            </div>
            <div class="detail-item">
              <label>难度等级：</label>
              <div class="detail-value">
                <span :class="['difficulty-badge', selectedQuestion.difficulty_level]">
                  {{ getDifficultyLabel(selectedQuestion.difficulty_level) }}
                </span>
              </div>
            </div>
            <div class="detail-item">
              <label>答题统计：</label>
              <div class="detail-value">
                {{ selectedQuestion.correct_answers }}/{{ selectedQuestion.total_answers }} 正确
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'HomeworkAnalytics',
  props: {
    homeworkId: {
      type: [String, Number],
      required: true
    }
  },
  data() {
    return {
      loading: false,
      analyticsData: null,
      error: null,
      selectedQuestion: null
    }
  },
  mounted() {
    this.loadAnalytics()
  },
  methods: {
    async loadAnalytics() {
      this.loading = true
      this.error = null
      
      try {
        const response = await axios.get(`/api/analytics/homework/${this.homeworkId}`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        })
        
        if (response.data.success) {
          this.analyticsData = response.data.data
          this.$nextTick(() => {
            this.renderScoreChart()
          })
        } else {
          this.error = response.data.message || '生成分析报告失败'
        }
      } catch (error) {
        console.error('加载分析报告失败:', error)
        this.error = error.response?.data?.message || '网络错误，请稍后重试'
      } finally {
        this.loading = false
      }
    },

    renderScoreChart() {
      if (!this.analyticsData || !this.$refs.scoreChart) return
      
      const canvas = this.$refs.scoreChart
      const ctx = canvas.getContext('2d')
      const distribution = this.analyticsData.score_distribution.distribution
      
      // 简单的柱状图绘制
      const barWidth = canvas.width / distribution.length
      const maxHeight = canvas.height - 40
      const maxPercentage = Math.max(...distribution.map(d => d.percentage))
      
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      
      distribution.forEach((item, index) => {
        const barHeight = (item.percentage / maxPercentage) * maxHeight
        const x = index * barWidth
        const y = canvas.height - barHeight - 20
        
        // 绘制柱子
        ctx.fillStyle = this.getGradeColor(item.label)
        ctx.fillRect(x + 5, y, barWidth - 10, barHeight)
        
        // 绘制标签
        ctx.fillStyle = '#333'
        ctx.font = '12px Arial'
        ctx.textAlign = 'center'
        ctx.fillText(item.label, x + barWidth/2, canvas.height - 5)
        ctx.fillText(item.percentage + '%', x + barWidth/2, y - 5)
      })
    },

    getGradeColor(label) {
      const colors = {
        '优秀': '#28a745',
        '良好': '#17a2b8',
        '中等': '#ffc107',
        '及格': '#fd7e14',
        '不及格': '#dc3545'
      }
      return colors[label] || '#6c757d'
    },

    getGradeClass(label) {
      const classes = {
        '优秀': 'excellent',
        '良好': 'good',
        '中等': 'average',
        '及格': 'pass',
        '不及格': 'fail'
      }
      return classes[label] || 'default'
    },

    getDifficultyClass(level) {
      return `difficulty-${level}`
    },

    getDifficultyLabel(level) {
      const labels = {
        easy: '简单',
        medium: '中等',
        hard: '困难'
      }
      return labels[level] || level
    },

    getMasteryLabel(level) {
      const labels = {
        excellent: '优秀',
        good: '良好',
        needs_improvement: '需改进'
      }
      return labels[level] || level
    },

    getPriorityLabel(priority) {
      const labels = {
        high: '高优先级',
        medium: '中优先级',
        low: '低优先级'
      }
      return labels[priority] || priority
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

    formatDateTime(dateString) {
      if (!dateString) return ''
      const date = new Date(dateString)
      return date.toLocaleString('zh-CN')
    },

    viewQuestionDetail(question) {
      this.selectedQuestion = question
    },

    closeQuestionDetail() {
      this.selectedQuestion = null
    },

    async exportReport(format) {
      try {
        const response = await axios.post(`/api/analytics/homework/${this.homeworkId}/export`, {
          format: format
        }, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        })
        
        if (response.data.success) {
          alert(`报告导出成功！下载链接：${response.data.download_url}`)
        } else {
          alert('导出失败：' + response.data.message)
        }
      } catch (error) {
        console.error('导出失败:', error)
        alert('导出失败，请稍后重试')
      }
    },

    printReport() {
      window.print()
    }
  }
}
</script>

<style scoped>
.homework-analytics {
  max-width: 1200px;
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

.analytics-header {
  text-align: center;
  margin-bottom: 30px;
  padding-bottom: 20px;
  border-bottom: 2px solid #eee;
}

.analytics-header h2 {
  color: #333;
  margin-bottom: 15px;
}

.report-meta {
  display: flex;
  justify-content: center;
  gap: 20px;
  flex-wrap: wrap;
}

.report-meta span {
  padding: 4px 12px;
  background: #f8f9fa;
  border-radius: 20px;
  font-size: 14px;
  color: #666;
}

.statistics-overview, .score-distribution, .question-analysis, 
.knowledge-analysis, .student-performance, .teaching-suggestions {
  margin-bottom: 40px;
}

.statistics-overview h3, .score-distribution h3, .question-analysis h3,
.knowledge-analysis h3, .student-performance h3, .teaching-suggestions h3 {
  color: #333;
  margin-bottom: 20px;
  padding-bottom: 10px;
  border-bottom: 2px solid #007bff;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 20px;
}

.stat-card {
  text-align: center;
  padding: 20px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-radius: 8px;
}

.stat-value {
  font-size: 32px;
  font-weight: bold;
  margin-bottom: 8px;
}

.stat-label {
  font-size: 14px;
  opacity: 0.9;
}

.distribution-chart {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 30px;
  margin-bottom: 20px;
}

.chart-container {
  background: #f8f9fa;
  padding: 20px;
  border-radius: 8px;
}

.distribution-stats {
  padding: 20px;
  background: #f8f9fa;
  border-radius: 8px;
}

.stat-item {
  display: flex;
  justify-content: space-between;
  margin-bottom: 10px;
  padding: 5px 0;
}

.stat-item .label {
  color: #666;
}

.stat-item .value {
  font-weight: bold;
  color: #333;
}

.grade-distribution {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.grade-bar {
  display: grid;
  grid-template-columns: 80px 1fr 100px;
  align-items: center;
  gap: 15px;
}

.grade-label {
  font-weight: bold;
  color: #333;
}

.grade-progress {
  height: 20px;
  background: #e9ecef;
  border-radius: 10px;
  overflow: hidden;
}

.grade-fill {
  height: 100%;
  border-radius: 10px;
  transition: width 0.3s ease;
}

.grade-fill.excellent { background: #28a745; }
.grade-fill.good { background: #17a2b8; }
.grade-fill.average { background: #ffc107; }
.grade-fill.pass { background: #fd7e14; }
.grade-fill.fail { background: #dc3545; }

.grade-stats {
  text-align: right;
  font-size: 14px;
  color: #666;
}

.questions-table {
  overflow-x: auto;
}

.questions-table table {
  width: 100%;
  border-collapse: collapse;
  background: white;
}

.questions-table th,
.questions-table td {
  padding: 12px;
  text-align: left;
  border-bottom: 1px solid #ddd;
}

.questions-table th {
  background: #f8f9fa;
  font-weight: bold;
  color: #333;
}

.questions-table tr:hover {
  background: #f8f9fa;
}

.questions-table tr.difficulty-hard {
  background: #fff5f5;
}

.questions-table tr.difficulty-easy {
  background: #f8fff8;
}

.correct-rate {
  display: flex;
  align-items: center;
  gap: 10px;
}

.rate-value {
  font-weight: bold;
  min-width: 50px;
}

.rate-bar {
  flex: 1;
  height: 8px;
  background: #e9ecef;
  border-radius: 4px;
  overflow: hidden;
}

.rate-fill {
  height: 100%;
  background: linear-gradient(90deg, #dc3545 0%, #ffc107 50%, #28a745 100%);
  border-radius: 4px;
}

.difficulty-badge {
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: bold;
}

.difficulty-badge.easy {
  background: #d4edda;
  color: #155724;
}

.difficulty-badge.medium {
  background: #fff3cd;
  color: #856404;
}

.difficulty-badge.hard {
  background: #f8d7da;
  color: #721c24;
}

.knowledge-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 20px;
}

.knowledge-card {
  padding: 20px;
  border-radius: 8px;
  border: 1px solid #ddd;
  background: white;
}

.knowledge-card.excellent {
  border-left: 4px solid #28a745;
  background: #f8fff8;
}

.knowledge-card.good {
  border-left: 4px solid #17a2b8;
  background: #f0f9ff;
}

.knowledge-card.needs_improvement {
  border-left: 4px solid #dc3545;
  background: #fff5f5;
}

.knowledge-name {
  font-weight: bold;
  color: #333;
  margin-bottom: 10px;
}

.mastery-rate {
  font-size: 24px;
  font-weight: bold;
  color: #007bff;
  margin-bottom: 8px;
}

.mastery-progress {
  height: 8px;
  background: #e9ecef;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 8px;
}

.mastery-fill {
  height: 100%;
  background: linear-gradient(90deg, #dc3545 0%, #ffc107 50%, #28a745 100%);
  border-radius: 4px;
}

.mastery-label {
  font-size: 14px;
  color: #666;
  margin-bottom: 5px;
}

.attempt-stats {
  font-size: 12px;
  color: #999;
}

.performance-summary {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 30px;
}

.summary-card {
  padding: 20px;
  border-radius: 8px;
  background: #f8f9fa;
}

.summary-card h4 {
  color: #333;
  margin-bottom: 15px;
}

.student-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.student-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px;
  border-radius: 6px;
  background: white;
}

.student-item.struggling {
  border-left: 4px solid #dc3545;
}

.student-item.excellent {
  border-left: 4px solid #28a745;
}

.student-name {
  font-weight: bold;
  color: #333;
}

.student-class {
  color: #666;
  font-size: 14px;
}

.student-score {
  font-weight: bold;
  color: #007bff;
}

.suggestions-list {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.suggestion-card {
  padding: 20px;
  border-radius: 8px;
  border: 1px solid #ddd;
  background: white;
}

.suggestion-card.priority-high {
  border-left: 4px solid #dc3545;
  background: #fff5f5;
}

.suggestion-card.priority-medium {
  border-left: 4px solid #ffc107;
  background: #fffbf0;
}

.suggestion-card.priority-low {
  border-left: 4px solid #28a745;
  background: #f8fff8;
}

.suggestion-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.suggestion-header h4 {
  color: #333;
  margin: 0;
}

.priority-badge {
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: bold;
}

.priority-badge.high {
  background: #f8d7da;
  color: #721c24;
}

.priority-badge.medium {
  background: #fff3cd;
  color: #856404;
}

.priority-badge.low {
  background: #d4edda;
  color: #155724;
}

.suggestion-content p {
  color: #666;
  line-height: 1.6;
  margin-bottom: 15px;
}

.action-items h5 {
  color: #333;
  margin-bottom: 10px;
}

.action-items ul {
  margin: 0;
  padding-left: 20px;
}

.action-items li {
  color: #666;
  margin-bottom: 5px;
}

.analytics-actions {
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

.btn-sm {
  padding: 5px 10px;
  font-size: 12px;
}

.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0,0,0,0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background: white;
  border-radius: 8px;
  max-width: 600px;
  width: 90%;
  max-height: 80vh;
  overflow-y: auto;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px;
  border-bottom: 1px solid #ddd;
}

.modal-header h4 {
  margin: 0;
  color: #333;
}

.close-btn {
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
  color: #666;
}

.close-btn:hover {
  color: #333;
}

.modal-body {
  padding: 20px;
}

.question-detail {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.detail-item {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.detail-item label {
  font-weight: bold;
  color: #666;
  font-size: 14px;
}

.detail-value {
  color: #333;
  line-height: 1.5;
}

@media (max-width: 768px) {
  .stats-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .distribution-chart {
    grid-template-columns: 1fr;
  }
  
  .performance-summary {
    grid-template-columns: 1fr;
  }
  
  .knowledge-grid {
    grid-template-columns: 1fr;
  }
  
  .analytics-actions {
    flex-direction: column;
    align-items: center;
  }
  
  .questions-table {
    font-size: 14px;
  }
  
  .grade-bar {
    grid-template-columns: 60px 1fr 80px;
    gap: 10px;
  }
}
</style>
