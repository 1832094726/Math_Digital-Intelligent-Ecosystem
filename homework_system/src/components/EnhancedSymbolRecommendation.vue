<template>
  <div class="enhanced-symbol-recommendation">
    <!-- 标题栏 -->
    <div class="recommendation-header">
      <h3 class="header-title">
        <i class="icon-magic"></i>
        智能符号推荐
      </h3>
      <div class="header-controls">
        <button 
          @click="toggleMode" 
          class="mode-toggle"
          :class="{ active: isAdaptiveMode }"
        >
          <i class="icon-brain"></i>
          {{ isAdaptiveMode ? '智能模式' : '基础模式' }}
        </button>
        <button @click="showAnalytics = !showAnalytics" class="analytics-toggle">
          <i class="icon-chart"></i>
          学习分析
        </button>
      </div>
    </div>

    <!-- 学习分析面板 -->
    <div v-if="showAnalytics" class="analytics-panel">
      <learning-analytics-dashboard
        :student-model="studentModel"
        :show-charts="false"
        @data-loaded="handleAnalyticsLoaded"
      />
    </div>

    <!-- 推荐模式指示器 -->
    <div class="mode-indicator" v-if="isAdaptiveMode">
      <div class="indicator-content">
        <i class="icon-sparkles"></i>
        <span>智能推荐已启用</span>
        <div class="learning-style-badge" v-if="learningStyle">
          {{ learningStyleText }}
        </div>
      </div>
    </div>

    <!-- 学习建议 -->
    <div v-if="learningSuggestions.length > 0" class="learning-suggestions">
      <h4 class="suggestions-title">
        <i class="icon-lightbulb"></i>
        个性化建议
      </h4>
      <div class="suggestions-list">
        <div 
          v-for="(suggestion, index) in learningSuggestions.slice(0, 3)" 
          :key="index"
          class="suggestion-item"
        >
          <i class="icon-arrow-right"></i>
          <span>{{ suggestion }}</span>
        </div>
      </div>
    </div>

    <!-- 符号推荐区域 -->
    <div class="recommendations-container">
      <!-- 加载状态 -->
      <div v-if="loading" class="loading-state">
        <div class="loading-spinner"></div>
        <p>正在生成智能推荐...</p>
      </div>

      <!-- 推荐符号列表 -->
      <div v-else-if="recommendations.length > 0" class="recommendations-grid">
        <div 
          v-for="recommendation in recommendations" 
          :key="recommendation.id"
          class="recommendation-card"
          :class="{ 
            'high-confidence': recommendation.score > 0.8,
            'adaptive': recommendation.adapted_score !== undefined
          }"
          @click="selectRecommendation(recommendation)"
        >
          <!-- 符号显示 -->
          <div class="symbol-display">
            <span class="symbol-text" v-html="recommendation.symbol || recommendation.name"></span>
            <div class="symbol-latex" v-if="recommendation.latex && recommendation.latex !== recommendation.symbol">
              {{ recommendation.latex }}
            </div>
          </div>

          <!-- 符号信息 -->
          <div class="symbol-info">
            <div class="symbol-description">{{ recommendation.description }}</div>
            <div class="symbol-category">{{ recommendation.category }}</div>
          </div>

          <!-- 推荐分数和来源 -->
          <div class="recommendation-meta">
            <div class="confidence-score" :class="getConfidenceClass(recommendation.score)">
              {{ Math.round((recommendation.adapted_score || recommendation.score) * 100) }}%
            </div>
            <div class="recommendation-source">
              <i :class="getSourceIcon(recommendation.source)"></i>
              {{ getSourceLabel(recommendation.source) }}
            </div>
          </div>

          <!-- 推荐解释 -->
          <div v-if="recommendation.explanation || recommendation.learning_adaptation" class="recommendation-explanation">
            <i class="icon-info"></i>
            <span>{{ recommendation.explanation || recommendation.learning_adaptation }}</span>
          </div>

          <!-- 掌握度指示器 -->
          <div v-if="recommendation.mastery_level !== undefined" class="mastery-indicator">
            <div class="mastery-bar">
              <div 
                class="mastery-fill" 
                :style="{ width: recommendation.mastery_level * 100 + '%' }"
              ></div>
            </div>
            <span class="mastery-text">掌握度 {{ Math.round(recommendation.mastery_level * 100) }}%</span>
          </div>
        </div>
      </div>

      <!-- 无推荐状态 -->
      <div v-else class="no-recommendations">
        <i class="icon-search"></i>
        <h4>暂无推荐</h4>
        <p>请输入更多内容或调整上下文信息</p>
        <button @click="refreshRecommendations" class="refresh-btn">
          <i class="icon-refresh"></i>
          刷新推荐
        </button>
      </div>
    </div>

    <!-- 快速操作栏 -->
    <div class="quick-actions">
      <button @click="showAllCategories" class="action-btn">
        <i class="icon-grid"></i>
        浏览分类
      </button>
      <button @click="showSearchDialog" class="action-btn">
        <i class="icon-search"></i>
        搜索符号
      </button>
      <button @click="showHistory" class="action-btn">
        <i class="icon-history"></i>
        使用历史
      </button>
    </div>

    <!-- 符号搜索对话框 -->
    <div v-if="showSearch" class="search-dialog-overlay" @click="closeSearchDialog">
      <div class="search-dialog" @click.stop>
        <div class="search-header">
          <h4>搜索符号</h4>
          <button @click="closeSearchDialog" class="close-btn">
            <i class="icon-close"></i>
          </button>
        </div>
        <div class="search-content">
          <input 
            v-model="searchQuery"
            @input="performSearch"
            placeholder="输入符号名称、描述或LaTeX命令..."
            class="search-input"
            ref="searchInput"
          >
          <div v-if="searchResults.length > 0" class="search-results">
            <div 
              v-for="result in searchResults" 
              :key="result.id"
              class="search-result-item"
              @click="selectSearchResult(result)"
            >
              <span class="result-symbol" v-html="result.symbol"></span>
              <div class="result-info">
                <div class="result-description">{{ result.description }}</div>
                <div class="result-match">{{ result.match_type }}</div>
              </div>
            </div>
          </div>
          <div v-else-if="searchQuery && !searchLoading" class="no-search-results">
            <i class="icon-search"></i>
            <p>未找到匹配的符号</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import RecommendationService from '@/services/RecommendationService';
import LearningAnalyticsDashboard from './LearningAnalyticsDashboard.vue';

export default {
  name: 'EnhancedSymbolRecommendation',
  
  components: {
    LearningAnalyticsDashboard
  },
  
  props: {
    studentModel: {
      type: Object,
      required: true
    },
    context: {
      type: Object,
      default: () => ({})
    },
    autoRefresh: {
      type: Boolean,
      default: true
    }
  },
  
  data() {
    return {
      loading: false,
      recommendations: [],
      isAdaptiveMode: true,
      showAnalytics: false,
      learningSuggestions: [],
      learningStyle: null,
      
      // 搜索相关
      showSearch: false,
      searchQuery: '',
      searchResults: [],
      searchLoading: false,
      searchTimeout: null,
      
      // 服务实例
      recommendationService: null
    };
  },
  
  computed: {
    learningStyleText() {
      const styleMap = {
        'explorer': '探索型',
        'specialist': '专精型', 
        'focused': '专注型',
        'balanced': '平衡型'
      };
      return styleMap[this.learningStyle] || this.learningStyle;
    }
  },
  
  watch: {
    context: {
      handler() {
        if (this.autoRefresh) {
          this.loadRecommendations();
        }
      },
      deep: true
    }
  },
  
  mounted() {
    this.recommendationService = new RecommendationService();
    this.loadRecommendations();
  },
  
  methods: {
    async loadRecommendations() {
      this.loading = true;
      
      try {
        if (this.isAdaptiveMode) {
          // 获取自适应推荐
          const adaptiveData = await this.recommendationService.getAdaptiveRecommendations(
            this.studentModel, 
            this.context
          );
          
          this.recommendations = adaptiveData.symbols || [];
          this.learningSuggestions = adaptiveData.learningSuggestions || [];
          this.learningStyle = adaptiveData.learningPattern?.learning_style;
        } else {
          // 获取基础推荐
          const basicRecommendations = await this.recommendationService.getSymbolRecommendations(
            this.studentModel,
            this.context
          );
          
          this.recommendations = basicRecommendations;
          this.learningSuggestions = [];
          this.learningStyle = null;
        }
        
        // 发出推荐更新事件
        this.$emit('recommendations-updated', this.recommendations);
        
      } catch (error) {
        console.error('Failed to load recommendations:', error);
        this.recommendations = [];
      } finally {
        this.loading = false;
      }
    },
    
    toggleMode() {
      this.isAdaptiveMode = !this.isAdaptiveMode;
      this.loadRecommendations();
    },
    
    selectRecommendation(recommendation) {
      // 记录符号使用
      this.recommendationService.recordEnhancedSymbolUsage(
        this.studentModel,
        recommendation,
        this.context
      );
      
      // 发出选择事件
      this.$emit('symbol-selected', recommendation);
    },
    
    refreshRecommendations() {
      this.loadRecommendations();
    },
    
    handleAnalyticsLoaded(data) {
      // 处理学习分析数据加载完成
      this.$emit('analytics-loaded', data);
    },
    
    // 搜索相关方法
    showSearchDialog() {
      this.showSearch = true;
      this.$nextTick(() => {
        this.$refs.searchInput?.focus();
      });
    },
    
    closeSearchDialog() {
      this.showSearch = false;
      this.searchQuery = '';
      this.searchResults = [];
    },
    
    async performSearch() {
      if (!this.searchQuery.trim()) {
        this.searchResults = [];
        return;
      }
      
      // 防抖搜索
      if (this.searchTimeout) {
        clearTimeout(this.searchTimeout);
      }
      
      this.searchTimeout = setTimeout(async () => {
        this.searchLoading = true;
        
        try {
          this.searchResults = await this.recommendationService.searchSymbols(this.searchQuery);
        } catch (error) {
          console.error('Search failed:', error);
          this.searchResults = [];
        } finally {
          this.searchLoading = false;
        }
      }, 300);
    },
    
    selectSearchResult(result) {
      this.selectRecommendation(result);
      this.closeSearchDialog();
    },
    
    // 工具方法
    getConfidenceClass(score) {
      if (score > 0.8) return 'high';
      if (score > 0.6) return 'medium';
      return 'low';
    },
    
    getSourceIcon(source) {
      const iconMap = {
        'basic': 'icon-star',
        'context': 'icon-target',
        'personalized': 'icon-user',
        'collaborative_filtering': 'icon-users',
        'knowledge_graph': 'icon-network'
      };
      return iconMap[source] || 'icon-star';
    },
    
    getSourceLabel(source) {
      const labelMap = {
        'basic': '基础',
        'context': '上下文',
        'personalized': '个性化',
        'collaborative_filtering': '协同过滤',
        'knowledge_graph': '知识图谱'
      };
      return labelMap[source] || source;
    },
    
    // 快速操作
    showAllCategories() {
      this.$emit('show-categories');
    },
    
    showHistory() {
      this.$emit('show-history');
    }
  }
};
</script>

<style scoped>
.enhanced-symbol-recommendation {
  background: white;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  overflow: hidden;
}

.recommendation-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.header-title {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  display: flex;
  align-items: center;
}

.header-title i {
  margin-right: 8px;
}

.header-controls {
  display: flex;
  gap: 8px;
}

.mode-toggle, .analytics-toggle {
  display: flex;
  align-items: center;
  padding: 6px 12px;
  background: rgba(255, 255, 255, 0.2);
  border: 1px solid rgba(255, 255, 255, 0.3);
  border-radius: 6px;
  color: white;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 12px;
}

.mode-toggle.active {
  background: rgba(255, 255, 255, 0.3);
}

.mode-toggle i, .analytics-toggle i {
  margin-right: 4px;
}

.analytics-panel {
  border-bottom: 1px solid #e2e8f0;
}

.mode-indicator {
  padding: 12px 20px;
  background: #f0f9ff;
  border-bottom: 1px solid #e0f2fe;
}

.indicator-content {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #0369a1;
  font-size: 14px;
}

.learning-style-badge {
  padding: 2px 8px;
  background: #0369a1;
  color: white;
  border-radius: 12px;
  font-size: 11px;
}

.learning-suggestions {
  padding: 16px 20px;
  background: #fefce8;
  border-bottom: 1px solid #fde047;
}

.suggestions-title {
  margin: 0 0 8px 0;
  font-size: 14px;
  color: #a16207;
  display: flex;
  align-items: center;
}

.suggestions-title i {
  margin-right: 6px;
}

.suggestions-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.suggestion-item {
  display: flex;
  align-items: center;
  font-size: 13px;
  color: #92400e;
}

.suggestion-item i {
  margin-right: 6px;
  font-size: 10px;
}

.recommendations-container {
  padding: 20px;
}

.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 40px 20px;
  text-align: center;
}

.loading-spinner {
  width: 32px;
  height: 32px;
  border: 3px solid #f3f3f3;
  border-top: 3px solid #667eea;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 12px;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.recommendations-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 16px;
}

.recommendation-card {
  padding: 16px;
  border: 2px solid #e2e8f0;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.3s ease;
  background: white;
}

.recommendation-card:hover {
  border-color: #667eea;
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
  transform: translateY(-2px);
}

.recommendation-card.high-confidence {
  border-color: #10b981;
  background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
}

.recommendation-card.adaptive {
  border-color: #8b5cf6;
  background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%);
}

.symbol-display {
  text-align: center;
  margin-bottom: 12px;
}

.symbol-text {
  font-size: 32px;
  font-weight: bold;
  color: #1e293b;
  display: block;
  margin-bottom: 4px;
}

.symbol-latex {
  font-size: 12px;
  color: #64748b;
  font-family: 'Consolas', 'Monaco', monospace;
}

.symbol-info {
  margin-bottom: 12px;
}

.symbol-description {
  font-size: 14px;
  color: #374151;
  margin-bottom: 4px;
}

.symbol-category {
  font-size: 12px;
  color: #6b7280;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.recommendation-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.confidence-score {
  font-weight: 600;
  font-size: 14px;
}

.confidence-score.high {
  color: #10b981;
}

.confidence-score.medium {
  color: #f59e0b;
}

.confidence-score.low {
  color: #ef4444;
}

.recommendation-source {
  display: flex;
  align-items: center;
  font-size: 11px;
  color: #6b7280;
}

.recommendation-source i {
  margin-right: 4px;
}

.recommendation-explanation {
  display: flex;
  align-items: flex-start;
  padding: 8px;
  background: #f8fafc;
  border-radius: 6px;
  font-size: 12px;
  color: #475569;
  margin-bottom: 8px;
}

.recommendation-explanation i {
  margin-right: 6px;
  margin-top: 1px;
  color: #667eea;
}

.mastery-indicator {
  margin-top: 8px;
}

.mastery-bar {
  width: 100%;
  height: 4px;
  background: #e2e8f0;
  border-radius: 2px;
  overflow: hidden;
  margin-bottom: 4px;
}

.mastery-fill {
  height: 100%;
  background: linear-gradient(90deg, #ef4444, #f59e0b, #10b981);
  transition: width 0.3s ease;
}

.mastery-text {
  font-size: 11px;
  color: #6b7280;
}

.no-recommendations {
  text-align: center;
  padding: 40px 20px;
  color: #6b7280;
}

.no-recommendations i {
  font-size: 48px;
  margin-bottom: 16px;
  color: #d1d5db;
}

.refresh-btn {
  margin-top: 16px;
  padding: 8px 16px;
  background: #667eea;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
}

.quick-actions {
  display: flex;
  justify-content: center;
  gap: 12px;
  padding: 16px 20px;
  background: #f8fafc;
  border-top: 1px solid #e2e8f0;
}

.action-btn {
  display: flex;
  align-items: center;
  padding: 8px 12px;
  background: white;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 12px;
  color: #374151;
}

.action-btn:hover {
  border-color: #667eea;
  color: #667eea;
}

.action-btn i {
  margin-right: 6px;
}

.search-dialog-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.search-dialog {
  background: white;
  border-radius: 12px;
  width: 90%;
  max-width: 500px;
  max-height: 80vh;
  overflow: hidden;
  box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
}

.search-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  background: #f8fafc;
  border-bottom: 1px solid #e2e8f0;
}

.search-header h4 {
  margin: 0;
  color: #1e293b;
}

.close-btn {
  background: none;
  border: none;
  cursor: pointer;
  color: #6b7280;
  font-size: 18px;
}

.search-content {
  padding: 20px;
}

.search-input {
  width: 100%;
  padding: 12px;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  font-size: 14px;
  margin-bottom: 16px;
}

.search-input:focus {
  outline: none;
  border-color: #667eea;
}

.search-results {
  max-height: 300px;
  overflow-y: auto;
}

.search-result-item {
  display: flex;
  align-items: center;
  padding: 12px;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.2s ease;
}

.search-result-item:hover {
  background: #f8fafc;
}

.result-symbol {
  font-size: 24px;
  margin-right: 12px;
  min-width: 40px;
  text-align: center;
}

.result-info {
  flex: 1;
}

.result-description {
  font-size: 14px;
  color: #374151;
  margin-bottom: 2px;
}

.result-match {
  font-size: 12px;
  color: #6b7280;
}

.no-search-results {
  text-align: center;
  padding: 40px 20px;
  color: #6b7280;
}

.no-search-results i {
  font-size: 32px;
  margin-bottom: 12px;
  color: #d1d5db;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .recommendation-header {
    flex-direction: column;
    gap: 12px;
  }
  
  .recommendations-grid {
    grid-template-columns: 1fr;
  }
  
  .quick-actions {
    flex-wrap: wrap;
  }
  
  .search-dialog {
    width: 95%;
    margin: 20px;
  }
}
</style>
