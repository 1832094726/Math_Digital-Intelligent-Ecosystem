<template>
  <div class="learning-analytics-dashboard">
    <!-- 仪表板标题 -->
    <div class="dashboard-header">
      <h2 class="dashboard-title">
        <i class="icon-analytics"></i>
        学习分析仪表板
      </h2>
      <div class="dashboard-actions">
        <button @click="refreshData" class="refresh-btn" :disabled="loading">
          <i class="icon-refresh" :class="{ 'spinning': loading }"></i>
          刷新数据
        </button>
        <button @click="exportReport" class="export-btn">
          <i class="icon-download"></i>
          导出报告
        </button>
      </div>
    </div>

    <!-- 加载状态 -->
    <div v-if="loading" class="loading-container">
      <div class="loading-spinner"></div>
      <p>正在分析学习数据...</p>
    </div>

    <!-- 错误状态 -->
    <div v-else-if="error" class="error-container">
      <i class="icon-error"></i>
      <h3>数据加载失败</h3>
      <p>{{ error }}</p>
      <button @click="refreshData" class="retry-btn">重试</button>
    </div>

    <!-- 主要内容 -->
    <div v-else class="dashboard-content">
      <!-- 概览卡片 -->
      <div class="overview-cards">
        <div class="overview-card activity-card">
          <div class="card-icon">
            <i class="icon-activity"></i>
          </div>
          <div class="card-content">
            <h3>活动水平</h3>
            <div class="activity-level" :class="activityLevelClass">
              {{ activityLevelText }}
            </div>
            <p class="card-description">基于使用频率和一致性评估</p>
          </div>
        </div>

        <div class="overview-card learning-style-card">
          <div class="card-icon">
            <i class="icon-brain"></i>
          </div>
          <div class="card-content">
            <h3>学习风格</h3>
            <div class="learning-style">{{ learningStyleText }}</div>
            <p class="card-description">{{ learningStyleDescription }}</p>
          </div>
        </div>

        <div class="overview-card progress-card">
          <div class="card-icon">
            <i class="icon-progress"></i>
          </div>
          <div class="card-content">
            <h3>总体进度</h3>
            <div class="progress-score">{{ Math.round(overallProgress * 100) }}%</div>
            <div class="progress-bar">
              <div class="progress-fill" :style="{ width: overallProgress * 100 + '%' }"></div>
            </div>
          </div>
        </div>

        <div class="overview-card mastery-card">
          <div class="card-icon">
            <i class="icon-star"></i>
          </div>
          <div class="card-content">
            <h3>掌握符号</h3>
            <div class="mastery-count">{{ masteredSymbolsCount }}</div>
            <p class="card-description">掌握度 > 70% 的符号数量</p>
          </div>
        </div>
      </div>

      <!-- 详细分析 -->
      <div class="detailed-analysis">
        <!-- 学习模式分析 -->
        <div class="analysis-section">
          <h3 class="section-title">
            <i class="icon-pattern"></i>
            学习模式分析
          </h3>
          <div class="pattern-metrics">
            <div class="metric-item">
              <label>学习一致性</label>
              <div class="metric-bar">
                <div class="metric-fill" :style="{ width: learningConsistency * 100 + '%' }"></div>
              </div>
              <span class="metric-value">{{ Math.round(learningConsistency * 100) }}%</span>
            </div>
            <div class="metric-item">
              <label>符号多样性</label>
              <div class="metric-bar">
                <div class="metric-fill" :style="{ width: symbolDiversity * 100 + '%' }"></div>
              </div>
              <span class="metric-value">{{ Math.round(symbolDiversity * 100) }}%</span>
            </div>
            <div class="metric-item">
              <label>知识保持率</label>
              <div class="metric-bar">
                <div class="metric-fill" :style="{ width: retentionRate * 100 + '%' }"></div>
              </div>
              <span class="metric-value">{{ Math.round(retentionRate * 100) }}%</span>
            </div>
          </div>
        </div>

        <!-- 优势与改进建议 -->
        <div class="analysis-section">
          <h3 class="section-title">
            <i class="icon-insights"></i>
            学习洞察
          </h3>
          <div class="insights-content">
            <div class="strengths-section">
              <h4>学习优势</h4>
              <ul class="strengths-list">
                <li v-for="strength in strengths" :key="strength" class="strength-item">
                  <i class="icon-check"></i>
                  {{ strength }}
                </li>
              </ul>
              <div v-if="strengths.length === 0" class="no-data">
                <i class="icon-info"></i>
                <span>继续学习以发现您的优势</span>
              </div>
            </div>

            <div class="improvements-section">
              <h4>改进建议</h4>
              <ul class="improvements-list">
                <li v-for="improvement in areasForImprovement" :key="improvement" class="improvement-item">
                  <i class="icon-arrow-up"></i>
                  {{ improvement }}
                </li>
              </ul>
              <div v-if="areasForImprovement.length === 0" class="no-data">
                <i class="icon-thumbs-up"></i>
                <span>您的学习表现很好！</span>
              </div>
            </div>
          </div>
        </div>

        <!-- 学习建议 -->
        <div class="analysis-section">
          <h3 class="section-title">
            <i class="icon-lightbulb"></i>
            个性化学习建议
          </h3>
          <div class="recommendations-list">
            <div v-for="(recommendation, index) in learningRecommendations" 
                 :key="index" 
                 class="recommendation-item">
              <div class="recommendation-icon">
                <i class="icon-suggestion"></i>
              </div>
              <div class="recommendation-content">
                <p>{{ recommendation }}</p>
              </div>
            </div>
            <div v-if="learningRecommendations.length === 0" class="no-recommendations">
              <i class="icon-info"></i>
              <p>继续使用系统，我们将为您提供个性化建议</p>
            </div>
          </div>
        </div>

        <!-- 偏好类别 -->
        <div class="analysis-section" v-if="preferredCategories.length > 0">
          <h3 class="section-title">
            <i class="icon-categories"></i>
            偏好符号类别
          </h3>
          <div class="categories-list">
            <div v-for="category in preferredCategories" 
                 :key="category" 
                 class="category-tag">
              {{ category }}
            </div>
          </div>
        </div>
      </div>

      <!-- 学习趋势图表 -->
      <div class="charts-section" v-if="showCharts">
        <h3 class="section-title">
          <i class="icon-chart"></i>
          学习趋势
        </h3>
        <div class="charts-container">
          <div class="chart-placeholder">
            <i class="icon-chart-line"></i>
            <p>学习趋势图表</p>
            <small>（图表功能开发中）</small>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import RecommendationService from '@/services/RecommendationService';

export default {
  name: 'LearningAnalyticsDashboard',
  
  props: {
    studentModel: {
      type: Object,
      required: true
    },
    showCharts: {
      type: Boolean,
      default: true
    }
  },
  
  data() {
    return {
      loading: false,
      error: null,
      analyticsData: null,
      insightsData: null,
      recommendationService: null
    };
  },
  
  computed: {
    activityLevelText() {
      const level = this.analyticsData?.learning_pattern?.activity_level || 'unknown';
      const levelMap = {
        'very_low': '很低',
        'low': '较低',
        'medium': '中等',
        'high': '较高',
        'very_high': '很高',
        'unknown': '未知'
      };
      return levelMap[level] || '未知';
    },
    
    activityLevelClass() {
      const level = this.analyticsData?.learning_pattern?.activity_level || 'unknown';
      return `activity-${level.replace('_', '-')}`;
    },
    
    learningStyleText() {
      const style = this.analyticsData?.learning_pattern?.learning_style || 'unknown';
      const styleMap = {
        'explorer': '探索型',
        'specialist': '专精型',
        'focused': '专注型',
        'balanced': '平衡型',
        'unknown': '未知'
      };
      return styleMap[style] || '未知';
    },
    
    learningStyleDescription() {
      const style = this.analyticsData?.learning_pattern?.learning_style || 'unknown';
      const descMap = {
        'explorer': '喜欢尝试新符号和概念',
        'specialist': '专注于掌握少数核心符号',
        'focused': '有明确的学习偏好和目标',
        'balanced': '均衡使用各种符号',
        'unknown': '继续学习以识别您的风格'
      };
      return descMap[style] || '继续学习以识别您的风格';
    },
    
    overallProgress() {
      return this.insightsData?.progress_indicators?.overall_progress || 0;
    },
    
    learningConsistency() {
      return this.analyticsData?.learning_pattern?.learning_consistency || 0;
    },
    
    symbolDiversity() {
      return this.analyticsData?.learning_pattern?.symbol_diversity || 0;
    },
    
    retentionRate() {
      return this.analyticsData?.learning_pattern?.retention_rate || 0;
    },
    
    masteredSymbolsCount() {
      const masteryLevels = this.analyticsData?.learning_pattern?.mastery_levels || {};
      return Object.values(masteryLevels).filter(level => level > 0.7).length;
    },
    
    strengths() {
      return this.insightsData?.strengths || [];
    },
    
    areasForImprovement() {
      return this.insightsData?.areas_for_improvement || [];
    },
    
    learningRecommendations() {
      return this.insightsData?.learning_recommendations || [];
    },
    
    preferredCategories() {
      return this.analyticsData?.learning_pattern?.preferred_categories || [];
    }
  },
  
  mounted() {
    this.recommendationService = new RecommendationService();
    this.loadAnalyticsData();
  },
  
  methods: {
    async loadAnalyticsData() {
      this.loading = true;
      this.error = null;
      
      try {
        // 并行加载分析数据和洞察数据
        const [analyticsData, insightsData] = await Promise.all([
          this.recommendationService.getUserLearningAnalytics(this.studentModel),
          this.recommendationService.getLearningInsights(this.studentModel)
        ]);
        
        this.analyticsData = analyticsData;
        this.insightsData = insightsData;
        
        // 发出数据加载完成事件
        this.$emit('data-loaded', {
          analytics: analyticsData,
          insights: insightsData
        });
        
      } catch (error) {
        console.error('Failed to load analytics data:', error);
        this.error = '加载学习分析数据失败，请稍后重试';
      } finally {
        this.loading = false;
      }
    },
    
    async refreshData() {
      await this.loadAnalyticsData();
    },
    
    exportReport() {
      // 导出学习报告功能
      const reportData = {
        studentId: this.studentModel.studentId,
        timestamp: new Date().toISOString(),
        analytics: this.analyticsData,
        insights: this.insightsData
      };
      
      const dataStr = JSON.stringify(reportData, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });
      
      const link = document.createElement('a');
      link.href = URL.createObjectURL(dataBlob);
      link.download = `learning-report-${this.studentModel.studentId}-${new Date().toISOString().split('T')[0]}.json`;
      link.click();
      
      this.$emit('report-exported', reportData);
    }
  }
};
</script>

<style scoped>
.learning-analytics-dashboard {
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  overflow: hidden;
}

.dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 24px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.dashboard-title {
  margin: 0;
  font-size: 24px;
  font-weight: 600;
  display: flex;
  align-items: center;
}

.dashboard-title i {
  margin-right: 12px;
  font-size: 28px;
}

.dashboard-actions {
  display: flex;
  gap: 12px;
}

.refresh-btn, .export-btn {
  display: flex;
  align-items: center;
  padding: 8px 16px;
  background: rgba(255, 255, 255, 0.2);
  border: 1px solid rgba(255, 255, 255, 0.3);
  border-radius: 6px;
  color: white;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 14px;
}

.refresh-btn:hover, .export-btn:hover {
  background: rgba(255, 255, 255, 0.3);
}

.refresh-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.refresh-btn i, .export-btn i {
  margin-right: 6px;
}

.spinning {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.loading-container, .error-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 20px;
  text-align: center;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #667eea;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 16px;
}

.error-container i {
  font-size: 48px;
  color: #f56565;
  margin-bottom: 16px;
}

.retry-btn {
  margin-top: 16px;
  padding: 8px 16px;
  background: #667eea;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
}

.dashboard-content {
  padding: 24px;
}

.overview-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin-bottom: 32px;
}

.overview-card {
  display: flex;
  align-items: center;
  padding: 20px;
  background: #f8fafc;
  border-radius: 12px;
  border-left: 4px solid #667eea;
}

.card-icon {
  margin-right: 16px;
  font-size: 32px;
  color: #667eea;
}

.card-content h3 {
  margin: 0 0 8px 0;
  font-size: 14px;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.activity-level, .learning-style, .progress-score, .mastery-count {
  font-size: 24px;
  font-weight: 700;
  color: #1e293b;
  margin-bottom: 4px;
}

.card-description {
  margin: 0;
  font-size: 12px;
  color: #64748b;
}

.progress-bar {
  width: 100%;
  height: 6px;
  background: #e2e8f0;
  border-radius: 3px;
  overflow: hidden;
  margin-top: 8px;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #667eea, #764ba2);
  transition: width 0.3s ease;
}

.detailed-analysis {
  display: grid;
  gap: 24px;
}

.analysis-section {
  background: #f8fafc;
  border-radius: 12px;
  padding: 20px;
}

.section-title {
  display: flex;
  align-items: center;
  margin: 0 0 16px 0;
  font-size: 18px;
  color: #1e293b;
}

.section-title i {
  margin-right: 8px;
  color: #667eea;
}

.pattern-metrics {
  display: grid;
  gap: 16px;
}

.metric-item {
  display: flex;
  align-items: center;
  gap: 12px;
}

.metric-item label {
  min-width: 100px;
  font-size: 14px;
  color: #64748b;
}

.metric-bar {
  flex: 1;
  height: 8px;
  background: #e2e8f0;
  border-radius: 4px;
  overflow: hidden;
}

.metric-fill {
  height: 100%;
  background: linear-gradient(90deg, #667eea, #764ba2);
  transition: width 0.3s ease;
}

.metric-value {
  min-width: 40px;
  text-align: right;
  font-weight: 600;
  color: #1e293b;
}

.insights-content {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}

.strengths-section h4, .improvements-section h4 {
  margin: 0 0 12px 0;
  color: #1e293b;
}

.strengths-list, .improvements-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.strength-item, .improvement-item {
  display: flex;
  align-items: center;
  padding: 8px 0;
  border-bottom: 1px solid #e2e8f0;
}

.strength-item:last-child, .improvement-item:last-child {
  border-bottom: none;
}

.strength-item i {
  margin-right: 8px;
  color: #10b981;
}

.improvement-item i {
  margin-right: 8px;
  color: #f59e0b;
}

.no-data {
  display: flex;
  align-items: center;
  padding: 16px;
  background: #f1f5f9;
  border-radius: 8px;
  color: #64748b;
}

.no-data i {
  margin-right: 8px;
}

.recommendations-list {
  display: grid;
  gap: 12px;
}

.recommendation-item {
  display: flex;
  align-items: flex-start;
  padding: 12px;
  background: white;
  border-radius: 8px;
  border-left: 3px solid #667eea;
}

.recommendation-icon {
  margin-right: 12px;
  color: #667eea;
  font-size: 16px;
  margin-top: 2px;
}

.recommendation-content p {
  margin: 0;
  color: #1e293b;
  line-height: 1.5;
}

.no-recommendations {
  text-align: center;
  padding: 24px;
  color: #64748b;
}

.no-recommendations i {
  font-size: 24px;
  margin-bottom: 8px;
  display: block;
}

.categories-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.category-tag {
  padding: 6px 12px;
  background: #667eea;
  color: white;
  border-radius: 16px;
  font-size: 12px;
  font-weight: 500;
}

.charts-section {
  margin-top: 24px;
  background: #f8fafc;
  border-radius: 12px;
  padding: 20px;
}

.charts-container {
  margin-top: 16px;
}

.chart-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 200px;
  background: white;
  border-radius: 8px;
  border: 2px dashed #e2e8f0;
  color: #64748b;
}

.chart-placeholder i {
  font-size: 48px;
  margin-bottom: 12px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .dashboard-header {
    flex-direction: column;
    gap: 16px;
    text-align: center;
  }
  
  .overview-cards {
    grid-template-columns: 1fr;
  }
  
  .insights-content {
    grid-template-columns: 1fr;
  }
  
  .overview-card {
    flex-direction: column;
    text-align: center;
  }
  
  .card-icon {
    margin-right: 0;
    margin-bottom: 12px;
  }
}
</style>
