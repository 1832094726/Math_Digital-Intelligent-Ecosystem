<template>
  <div class="symbol-recommendation-container">
    <h3 class="recommendation-title">
      <i class="icon-recommendation"></i>
      推荐符号
      <button 
        class="refresh-btn" 
        @click="refreshRecommendations" 
        :disabled="loading"
        title="刷新推荐"
      >
        <i class="icon-refresh" :class="{ 'rotating': loading }"></i>
      </button>
    </h3>
    
    <div v-if="loading" class="loading-container">
      <div class="loading-spinner"></div>
      <span>加载推荐中...</span>
    </div>
    
    <div v-else-if="error" class="error-container">
      <i class="icon-error"></i>
      <span>{{ error }}</span>
      <button class="retry-btn" @click="refreshRecommendations">重试</button>
    </div>
    
    <div v-else>
      <!-- 符号推荐区域 -->
      <div class="symbol-section" v-if="recommendedSymbols.length > 0">
        <h4>常用符号</h4>
        <div class="symbol-grid">
          <div 
            v-for="symbol in recommendedSymbols" 
            :key="symbol.id"
            class="symbol-item"
            :title="symbol.description"
            @click="selectSymbol(symbol)"
          >
            <span v-html="symbol.name"></span>
          </div>
        </div>
      </div>
      
      <!-- 公式推荐区域 -->
      <div class="formula-section" v-if="recommendedFormulas.length > 0">
        <h4>相关公式</h4>
        <div class="formula-list">
          <div 
            v-for="formula in recommendedFormulas" 
            :key="formula.id"
            class="formula-item"
            @click="selectFormula(formula)"
          >
            <div class="formula-name">{{ formula.name }}</div>
            <div class="formula-latex" v-html="renderLatex(formula.latex)"></div>
          </div>
        </div>
      </div>
      
      <!-- 知识点推荐区域 -->
      <div class="knowledge-section" v-if="recommendedKnowledge.length > 0">
        <h4>相关知识点</h4>
        <div class="knowledge-list">
          <div 
            v-for="knowledge in recommendedKnowledge" 
            :key="knowledge.id"
            class="knowledge-item"
            @click="selectKnowledge(knowledge)"
          >
            <i class="icon-knowledge"></i>
            <div class="knowledge-content">
              <div class="knowledge-name">{{ knowledge.name }}</div>
              <div class="knowledge-description">{{ knowledge.description }}</div>
            </div>
          </div>
        </div>
      </div>
      
      <div v-if="!hasRecommendations" class="empty-recommendations">
        <i class="icon-empty"></i>
        <p>暂无推荐内容</p>
        <p class="empty-tip">继续输入内容，系统将为您提供推荐</p>
      </div>
    </div>
    
    <!-- 设备适配提示 -->
    <div class="device-adaptation" v-if="showDeviceAdaptation">
      <h4>设备适配</h4>
      <div class="device-options">
        <button 
          v-for="device in deviceOptions" 
          :key="device.id"
          class="device-option"
          :class="{ 'active': currentDevice === device.id }"
          @click="switchDevice(device.id)"
        >
          <i :class="`icon-${device.id}`"></i>
          <span>{{ device.name }}</span>
        </button>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'SymbolRecommendation',
  
  props: {
    recommendationService: {
      type: Object,
      required: true
    },
    studentModel: {
      type: Object,
      required: true
    },
    context: {
      type: Object,
      default: () => ({})
    },
    showDeviceAdaptation: {
      type: Boolean,
      default: false
    }
  },
  
  data() {
    return {
      loading: false,
      error: null,
      recommendedSymbols: [],
      recommendedFormulas: [],
      recommendedKnowledge: [],
      currentDevice: 'pc', // 默认设备类型
      deviceOptions: [
        { id: 'pc', name: '电脑' },
        { id: 'tablet', name: '平板' },
        { id: 'mobile', name: '手机' },
        { id: 'robot', name: '机器人' }
      ],
      // 用于渲染LaTeX的函数，实际应用中会替换为真实的渲染函数
      latexRenderer: null
    };
  },
  
  computed: {
    hasRecommendations() {
      return this.recommendedSymbols.length > 0 || 
             this.recommendedFormulas.length > 0 || 
             this.recommendedKnowledge.length > 0;
    }
  },
  
  watch: {
    context: {
      handler(newContext) {
        // 当上下文发生变化时，重新获取推荐
        this.getRecommendations();
      },
      deep: true
    }
  },
  
  mounted() {
    // 初始化LaTeX渲染器
    this.initLatexRenderer();
    
    // 获取设备偏好
    this.currentDevice = this.studentModel.features.basic.devicePreference || 'pc';
    
    // 获取初始推荐
    this.getRecommendations();
  },
  
  methods: {
    async getRecommendations() {
      if (this.loading) return;
      
      this.loading = true;
      this.error = null;
      
      try {
        // 构建上下文
        const enrichedContext = {
          ...this.context,
          device: this.currentDevice,
          timestamp: new Date().toISOString()
        };
        
        // 并行获取各种推荐
        const [symbols, formulas, knowledge] = await Promise.all([
          this.recommendationService.getSymbolRecommendations(this.studentModel, enrichedContext),
          this.recommendationService.getFormulaRecommendations(this.studentModel, enrichedContext),
          this.recommendationService.getKnowledgePointRecommendations(this.studentModel, enrichedContext)
        ]);
        
        // 更新推荐结果
        this.recommendedSymbols = symbols.slice(0, 12); // 限制显示数量
        this.recommendedFormulas = formulas.slice(0, 3);
        this.recommendedKnowledge = knowledge.slice(0, 3);
      } catch (error) {
        console.error('Failed to get recommendations:', error);
        this.error = '获取推荐失败，请重试';
      } finally {
        this.loading = false;
      }
    },
    
    refreshRecommendations() {
      this.getRecommendations();
    },
    
    selectSymbol(symbol) {
      this.$emit('select-symbol', symbol);
      
      // 记录学生行为
      this.recordSymbolSelection(symbol);
    },
    
    selectFormula(formula) {
      this.$emit('select-formula', formula);
      
      // 记录学生行为
      this.recordFormulaSelection(formula);
    },
    
    selectKnowledge(knowledge) {
      this.$emit('select-knowledge', knowledge);
      
      // 记录学生行为
      this.recordKnowledgeSelection(knowledge);
    },
    
    switchDevice(deviceId) {
      this.currentDevice = deviceId;
      
      // 更新学生模型中的设备偏好
      if (this.studentModel) {
        this.studentModel.update({
          basic: {
            devicePreference: deviceId
          }
        });
      }
      
      // 重新获取推荐
      this.getRecommendations();
      
      // 发出设备切换事件
      this.$emit('device-changed', deviceId);
    },
    
    initLatexRenderer() {
      // 实际应用中，这里会初始化真实的LaTeX渲染器，如MathJax或KaTeX
      // 现在使用一个简单的占位实现
      this.latexRenderer = (latex) => {
        return `<span class="latex-formula">${latex}</span>`;
      };
    },
    
    renderLatex(latex) {
      if (!this.latexRenderer) return latex;
      return this.latexRenderer(latex);
    },
    
    recordSymbolSelection(symbol) {
      // 更新学生模型中的符号使用模式
      if (!this.studentModel) return;
      
      const symbolId = symbol.id;
      const currentPatterns = this.studentModel.features.preferences.symbolUsagePatterns || {};
      
      // 增加使用次数
      const updatedPatterns = {
        ...currentPatterns,
        [symbolId]: (currentPatterns[symbolId] || 0) + 1
      };
      
      // 更新学生模型
      this.studentModel.update({
        preferences: {
          symbolUsagePatterns: updatedPatterns
        }
      });
    },
    
    recordFormulaSelection(formula) {
      // 更新学生模型中的公式偏好
      if (!this.studentModel) return;
      
      const formulaId = formula.id;
      const currentPreferences = this.studentModel.features.preferences.formulaPreferences || [];
      
      // 如果公式已存在于偏好中，将其移到最前面
      const existingIndex = currentPreferences.findIndex(id => id === formulaId);
      let updatedPreferences;
      
      if (existingIndex >= 0) {
        updatedPreferences = [
          formulaId,
          ...currentPreferences.slice(0, existingIndex),
          ...currentPreferences.slice(existingIndex + 1)
        ];
      } else {
        // 否则添加到最前面，并保持列表不超过10个
        updatedPreferences = [formulaId, ...currentPreferences].slice(0, 10);
      }
      
      // 更新学生模型
      this.studentModel.update({
        preferences: {
          formulaPreferences: updatedPreferences
        }
      });
    },
    
    recordKnowledgeSelection(knowledge) {
      // 更新学生模型中的知识点兴趣
      if (!this.studentModel) return;
      
      const knowledgeId = knowledge.id;
      
      // 记录学习行为
      this.studentModel.recordBehavior('engagement_action', {
        engagementScore: 80, // 查看知识点是一个高参与度的行为
        duration: 1, // 假设查看时间为1分钟
        type: 'knowledge_exploration',
        knowledgeId
      });
    }
  }
};
</script>

<style scoped>
.symbol-recommendation-container {
  background-color: var(--background-light);
  border-radius: 8px;
  padding: 15px;
  margin-bottom: 20px;
}

.recommendation-title {
  display: flex;
  align-items: center;
  margin-top: 0;
  margin-bottom: 15px;
  font-size: 18px;
  color: var(--text-color);
}

.recommendation-title i {
  margin-right: 8px;
  color: var(--primary-color);
}

.refresh-btn {
  margin-left: auto;
  background: none;
  border: none;
  color: var(--light-text);
  cursor: pointer;
  padding: 5px;
}

.refresh-btn:hover {
  color: var(--primary-color);
}

.refresh-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.rotating {
  animation: rotate 1s linear infinite;
}

@keyframes rotate {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px;
  color: var(--light-text);
}

.loading-spinner {
  border: 3px solid var(--background-dark);
  border-top: 3px solid var(--primary-color);
  border-radius: 50%;
  width: 24px;
  height: 24px;
  animation: spin 1s linear infinite;
  margin-bottom: 10px;
}

.error-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px;
  color: #dc3545;
}

.error-container i {
  margin-bottom: 10px;
  font-size: 24px;
}

.retry-btn {
  margin-top: 10px;
  padding: 5px 10px;
  background-color: var(--background-dark);
  border: none;
  border-radius: 4px;
  color: var(--text-color);
  cursor: pointer;
}

.retry-btn:hover {
  background-color: var(--border-color);
}

h4 {
  margin-top: 15px;
  margin-bottom: 10px;
  font-size: 16px;
  color: var(--text-color);
}

/* 符号网格 */
.symbol-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(40px, 1fr));
  gap: 8px;
}

.symbol-item {
  background-color: white;
  border: 1px solid var(--border-color);
  border-radius: 4px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: var(--transition);
  font-size: 18px;
}

.symbol-item:hover {
  background-color: var(--primary-color);
  color: white;
  border-color: var(--primary-color);
}

/* 公式列表 */
.formula-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.formula-item {
  background-color: white;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  padding: 10px;
  cursor: pointer;
  transition: var(--transition);
}

.formula-item:hover {
  background-color: var(--background-light);
  border-color: var(--primary-color);
}

.formula-name {
  font-weight: 500;
  margin-bottom: 5px;
  color: var(--text-color);
}

.formula-latex {
  font-size: 16px;
  color: var(--text-color);
  overflow-x: auto;
  padding: 5px 0;
}

/* 知识点列表 */
.knowledge-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.knowledge-item {
  background-color: white;
  border-radius: 6px;
  padding: 10px;
  display: flex;
  align-items: flex-start;
  cursor: pointer;
  transition: var(--transition);
  border: 1px solid var(--border-color);
}

.knowledge-item:hover {
  background-color: var(--background-light);
  border-color: var(--primary-color);
}

.knowledge-item i {
  margin-right: 10px;
  color: var(--primary-color);
  margin-top: 2px;
}

.knowledge-content {
  flex: 1;
}

.knowledge-name {
  font-weight: 500;
  margin-bottom: 5px;
  color: var(--text-color);
}

.knowledge-description {
  font-size: 14px;
  color: var(--light-text);
}

/* 空状态 */
.empty-recommendations {
  text-align: center;
  padding: 20px;
  color: var(--light-text);
}

.empty-recommendations i {
  font-size: 32px;
  margin-bottom: 10px;
  opacity: 0.5;
}

.empty-tip {
  font-size: 14px;
  margin-top: 5px;
}

/* 设备适配 */
.device-adaptation {
  margin-top: 20px;
  padding-top: 15px;
  border-top: 1px solid var(--border-color);
}

.device-options {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.device-option {
  background-color: white;
  border: 1px solid var(--border-color);
  border-radius: 4px;
  padding: 8px 12px;
  display: flex;
  align-items: center;
  cursor: pointer;
  transition: var(--transition);
  font-size: 14px;
}

.device-option i {
  margin-right: 5px;
}

.device-option:hover {
  background-color: var(--background-light);
}

.device-option.active {
  background-color: var(--primary-color);
  color: white;
  border-color: var(--primary-color);
}

/* 响应式调整 */
@media (max-width: 768px) {
  .symbol-grid {
    grid-template-columns: repeat(auto-fill, minmax(35px, 1fr));
  }
  
  .device-options {
    justify-content: space-between;
  }
  
  .device-option {
    flex: 1;
    justify-content: center;
  }
}
</style> 