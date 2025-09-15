<template>
  <div class="symbol-recommendation">
    <!-- 推荐头部 -->
    <div class="recommendation-header">
      <h4 class="title">
        <i class="el-icon-magic-stick"></i>
        智能符号推荐
      </h4>
      <el-input 
        v-model="searchQuery" 
        placeholder="搜索符号..."
        size="small"
        prefix-icon="el-icon-search"
        @input="handleSearch"
        clearable
      />
    </div>
    
    <!-- 符号分类标签 -->
    <div class="symbol-categories">
      <el-tabs v-model="activeCategory" @tab-click="handleCategoryChange" size="small">
        <el-tab-pane label="推荐" name="recommended">
          <div class="symbol-grid" v-loading="loading">
            <div 
              v-for="symbol in displayedSymbols" 
              :key="symbol.id"
              class="symbol-item"
              :class="{ 'high-confidence': symbol.confidence > 0.8 }"
              @click="selectSymbol(symbol)"
            >
              <div class="symbol-content">
                <span class="symbol-text">{{ symbol.symbol_text }}</span>
                <span class="symbol-name">{{ symbol.symbol_name }}</span>
                <div class="symbol-meta">
                  <span class="confidence">{{ (symbol.confidence * 100).toFixed(0) }}%</span>
                  <span class="category">{{ getCategoryName(symbol.category) }}</span>
                </div>
              </div>
            </div>
          </div>
          
          <!-- 空状态 -->
          <div v-if="!loading && displayedSymbols.length === 0" class="empty-state">
            <i class="el-icon-info"></i>
            <p>暂无推荐符号</p>
            <p class="hint">请输入数学表达式获取智能推荐</p>
          </div>
        </el-tab-pane>
        
        <el-tab-pane label="常用" name="common">
          <div class="symbol-grid">
            <div 
              v-for="symbol in commonSymbols" 
              :key="symbol.id"
              class="symbol-item"
              @click="selectSymbol(symbol)"
            >
              <div class="symbol-content">
                <span class="symbol-text">{{ symbol.symbol_text }}</span>
                <span class="symbol-name">{{ symbol.symbol_name }}</span>
              </div>
            </div>
          </div>
        </el-tab-pane>
        
        <el-tab-pane label="历史" name="history">
          <div class="symbol-grid">
            <div 
              v-for="symbol in recentSymbols" 
              :key="symbol.id"
              class="symbol-item recent"
              @click="selectSymbol(symbol)"
            >
              <div class="symbol-content">
                <span class="symbol-text">{{ symbol.symbol_text }}</span>
                <span class="symbol-name">{{ symbol.symbol_name }}</span>
                <span class="usage-count">{{ symbol.usage_count }}次</span>
              </div>
            </div>
          </div>
        </el-tab-pane>
      </el-tabs>
    </div>
    
    <!-- 推荐统计 -->
    <div class="recommendation-stats" v-if="stats">
      <el-tooltip content="推荐准确率" placement="top">
        <div class="stat-item">
          <i class="el-icon-success"></i>
          <span>{{ stats.usage_rate.toFixed(1) }}%</span>
        </div>
      </el-tooltip>
      <el-tooltip content="今日推荐次数" placement="top">
        <div class="stat-item">
          <i class="el-icon-data-line"></i>
          <span>{{ stats.total_recommendations }}</span>
        </div>
      </el-tooltip>
    </div>
  </div>
</template>

<script>
import { mapGetters } from 'vuex'

export default {
  name: 'SymbolRecommendation',
  props: {
    context: {
      type: String,
      default: ''
    },
    questionId: {
      type: Number,
      default: null
    },
    visible: {
      type: Boolean,
      default: true
    }
  },
  data() {
    return {
      recommendedSymbols: [],
      commonSymbols: [
        { id: 'c1', symbol_text: '+', symbol_name: '加号', category: 'arithmetic' },
        { id: 'c2', symbol_text: '-', symbol_name: '减号', category: 'arithmetic' },
        { id: 'c3', symbol_text: '×', symbol_name: '乘号', category: 'arithmetic' },
        { id: 'c4', symbol_text: '÷', symbol_name: '除号', category: 'arithmetic' },
        { id: 'c5', symbol_text: '=', symbol_name: '等号', category: 'relation' },
        { id: 'c6', symbol_text: 'x', symbol_name: '未知数x', category: 'variable' },
        { id: 'c7', symbol_text: 'y', symbol_name: '未知数y', category: 'variable' },
        { id: 'c8', symbol_text: '√', symbol_name: '根号', category: 'root' }
      ],
      recentSymbols: [],
      activeCategory: 'recommended',
      searchQuery: '',
      loading: false,
      stats: null,
      categoryNames: {
        'arithmetic': '运算',
        'relation': '关系',
        'variable': '变量',
        'parameter': '参数',
        'function': '函数',
        'geometry': '几何',
        'fraction': '分数',
        'root': '根式'
      }
    }
  },
  computed: {
    ...mapGetters(['isAuthenticated', 'userToken']),
    
    displayedSymbols() {
      if (!this.searchQuery) {
        return this.recommendedSymbols
      }
      
      return this.recommendedSymbols.filter(symbol => 
        symbol.symbol_text.includes(this.searchQuery) ||
        symbol.symbol_name.includes(this.searchQuery)
      )
    }
  },
  watch: {
    context: {
      handler: 'fetchRecommendations',
      immediate: true
    },
    visible: {
      handler(newVal) {
        if (newVal) {
          this.fetchStats()
        }
      }
    }
  },
  mounted() {
    this.fetchStats()
    this.loadRecentSymbols()
  },
  methods: {
    async fetchRecommendations() {
      if (!this.context || !this.isAuthenticated) return
      
      this.loading = true
      try {
        const response = await this.$http.post('/api/recommend/symbols', {
          context: this.context,
          question_id: this.questionId,
          limit: 12
        }, {
          headers: {
            'Authorization': `Bearer ${this.userToken}`
          }
        })
        
        if (response.data.success) {
          this.recommendedSymbols = response.data.recommendations
        } else {
          this.$message.warning('获取符号推荐失败')
        }
      } catch (error) {
        console.error('获取符号推荐失败:', error)
        this.$message.error('推荐服务暂时不可用')
      } finally {
        this.loading = false
      }
    },
    
    async fetchStats() {
      if (!this.isAuthenticated) return
      
      try {
        const response = await this.$http.get('/api/recommend/stats', {
          headers: {
            'Authorization': `Bearer ${this.userToken}`
          }
        })
        
        if (response.data.success) {
          this.stats = response.data.stats
        }
      } catch (error) {
        console.error('获取推荐统计失败:', error)
      }
    },
    
    loadRecentSymbols() {
      // 从本地存储加载最近使用的符号
      const recent = localStorage.getItem('recent_symbols')
      if (recent) {
        this.recentSymbols = JSON.parse(recent)
      }
    },
    
    selectSymbol(symbol) {
      // 发射符号选择事件
      this.$emit('symbol-selected', symbol)
      
      // 记录使用统计
      this.recordSymbolUsage(symbol)
      
      // 更新最近使用
      this.updateRecentSymbols(symbol)
      
      // 显示选择反馈
      this.$message.success(`已选择符号: ${symbol.symbol_name}`)
    },
    
    async recordSymbolUsage(symbol) {
      if (!this.isAuthenticated) return
      
      try {
        await this.$http.post('/api/recommend/symbols/usage', {
          symbol_id: symbol.id,
          symbol: symbol.symbol_text,
          context: this.context
        }, {
          headers: {
            'Authorization': `Bearer ${this.userToken}`
          }
        })
      } catch (error) {
        console.error('记录符号使用失败:', error)
      }
    },
    
    updateRecentSymbols(symbol) {
      // 更新最近使用的符号
      let recent = [...this.recentSymbols]
      
      // 查找是否已存在
      const existingIndex = recent.findIndex(s => s.symbol_text === symbol.symbol_text)
      
      if (existingIndex >= 0) {
        // 增加使用次数并移到前面
        recent[existingIndex].usage_count = (recent[existingIndex].usage_count || 0) + 1
        const item = recent.splice(existingIndex, 1)[0]
        recent.unshift(item)
      } else {
        // 添加新符号
        recent.unshift({
          ...symbol,
          usage_count: 1
        })
      }
      
      // 限制数量
      recent = recent.slice(0, 10)
      
      this.recentSymbols = recent
      localStorage.setItem('recent_symbols', JSON.stringify(recent))
    },
    
    handleSearch() {
      // 搜索处理已在计算属性中实现
    },
    
    handleCategoryChange() {
      // 分类切换处理
      if (this.activeCategory === 'recommended' && this.recommendedSymbols.length === 0) {
        this.fetchRecommendations()
      }
    },
    
    getCategoryName(category) {
      return this.categoryNames[category] || category
    }
  }
}
</script>

<style scoped>
.symbol-recommendation {
  background: #fff;
  border-radius: 8px;
  padding: 16px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.recommendation-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.title {
  margin: 0;
  color: #303133;
  font-size: 16px;
  font-weight: 600;
}

.title i {
  color: #409eff;
  margin-right: 8px;
}

.symbol-categories {
  margin-bottom: 16px;
}

.symbol-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 12px;
  min-height: 200px;
}

.symbol-item {
  border: 1px solid #e4e7ed;
  border-radius: 6px;
  padding: 12px;
  cursor: pointer;
  transition: all 0.3s;
  background: #fff;
}

.symbol-item:hover {
  border-color: #409eff;
  box-shadow: 0 2px 8px rgba(64, 158, 255, 0.2);
  transform: translateY(-2px);
}

.symbol-item.high-confidence {
  border-color: #67c23a;
  background: linear-gradient(135deg, #f0f9ff 0%, #e6f7ff 100%);
}

.symbol-item.recent {
  border-color: #e6a23c;
}

.symbol-content {
  text-align: center;
}

.symbol-text {
  display: block;
  font-size: 24px;
  font-weight: bold;
  color: #303133;
  margin-bottom: 4px;
}

.symbol-name {
  display: block;
  font-size: 12px;
  color: #606266;
  margin-bottom: 8px;
}

.symbol-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 10px;
}

.confidence {
  background: #409eff;
  color: white;
  padding: 2px 6px;
  border-radius: 10px;
}

.category {
  color: #909399;
}

.usage-count {
  font-size: 10px;
  color: #e6a23c;
  font-weight: bold;
}

.empty-state {
  grid-column: 1 / -1;
  text-align: center;
  padding: 40px 20px;
  color: #909399;
}

.empty-state i {
  font-size: 48px;
  margin-bottom: 16px;
}

.empty-state p {
  margin: 8px 0;
}

.hint {
  font-size: 12px;
  color: #c0c4cc;
}

.recommendation-stats {
  display: flex;
  justify-content: center;
  gap: 20px;
  padding-top: 16px;
  border-top: 1px solid #f0f0f0;
}

.stat-item {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  color: #606266;
}

.stat-item i {
  color: #409eff;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .symbol-grid {
    grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
    gap: 8px;
  }
  
  .symbol-item {
    padding: 8px;
  }
  
  .symbol-text {
    font-size: 20px;
  }
  
  .recommendation-header {
    flex-direction: column;
    gap: 12px;
  }
}
</style>
