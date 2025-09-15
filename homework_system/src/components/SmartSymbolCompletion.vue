<template>
  <div class="smart-symbol-completion">
    <!-- 输入区域 -->
    <div class="input-container">
      <div class="input-wrapper" :class="{ 'has-completions': showCompletions }">
        <textarea
          ref="inputArea"
          v-model="inputContent"
          @input="handleInput"
          @keydown="handleKeydown"
          @click="updateCursorPosition"
          @blur="handleBlur"
          @focus="handleFocus"
          :placeholder="placeholder"
          class="symbol-input"
          spellcheck="false"
        ></textarea>
        
        <!-- 光标位置指示器 -->
        <div class="cursor-indicator" v-if="showCursorInfo">
          <span>行: {{ cursorPosition.row + 1 }}, 列: {{ cursorPosition.column + 1 }}</span>
        </div>
      </div>
      
      <!-- 补全建议下拉框 -->
      <div 
        class="completion-dropdown" 
        v-if="showCompletions && completions.length > 0"
        :style="dropdownStyle"
      >
        <div class="completion-header">
          <i class="icon-lightbulb"></i>
          <span>符号建议</span>
          <span class="completion-count">({{ completions.length }})</span>
        </div>
        
        <div class="completion-list">
          <div
            v-for="(completion, index) in completions"
            :key="completion.id"
            class="completion-item"
            :class="{ 
              'selected': selectedIndex === index,
              'high-score': completion.score > 0.8,
              'medium-score': completion.score > 0.5 && completion.score <= 0.8
            }"
            @click="selectCompletion(completion)"
            @mouseenter="selectedIndex = index"
          >
            <div class="completion-symbol">
              <span class="symbol-display" v-html="completion.symbol"></span>
              <span class="latex-display" v-if="completion.latex !== completion.symbol">
                {{ completion.latex }}
              </span>
            </div>
            
            <div class="completion-info">
              <div class="completion-description">{{ completion.description }}</div>
              <div class="completion-meta">
                <span class="score-indicator" :class="getScoreClass(completion.score)">
                  {{ Math.round(completion.score * 100) }}%
                </span>
                <span class="completion-source">{{ getSourceLabel(completion.sources) }}</span>
              </div>
            </div>
            
            <div class="completion-shortcut">
              <kbd v-if="index < 9">{{ index + 1 }}</kbd>
              <kbd v-else-if="index === selectedIndex">Enter</kbd>
            </div>
          </div>
        </div>
        
        <div class="completion-footer">
          <span class="help-text">
            <kbd>↑↓</kbd> 选择 <kbd>Enter</kbd> 确认 <kbd>Esc</kbd> 取消
          </span>
        </div>
      </div>
    </div>
    
    <!-- 实时预览区域 -->
    <div class="preview-container" v-if="showPreview">
      <div class="preview-header">
        <i class="icon-eye"></i>
        <span>实时预览</span>
        <button @click="togglePreview" class="preview-toggle">
          <i class="icon-minimize"></i>
        </button>
      </div>
      <div class="preview-content" ref="previewContent">
        <!-- 这里会渲染LaTeX内容 -->
      </div>
    </div>
    
    <!-- 符号面板 -->
    <div class="symbol-panel" v-if="showSymbolPanel">
      <div class="panel-header">
        <i class="icon-grid"></i>
        <span>常用符号</span>
        <button @click="toggleSymbolPanel" class="panel-toggle">
          <i class="icon-close"></i>
        </button>
      </div>
      
      <div class="symbol-categories">
        <button
          v-for="category in symbolCategories"
          :key="category.id"
          class="category-btn"
          :class="{ active: activeCategory === category.id }"
          @click="switchCategory(category.id)"
        >
          <i :class="category.icon"></i>
          <span>{{ category.name }}</span>
        </button>
      </div>
      
      <div class="symbol-grid">
        <button
          v-for="symbol in currentCategorySymbols"
          :key="symbol.id"
          class="symbol-btn"
          :title="symbol.description"
          @click="insertSymbol(symbol)"
        >
          <span v-html="symbol.symbol"></span>
        </button>
      </div>
    </div>
    
    <!-- 工具栏 -->
    <div class="toolbar">
      <button @click="toggleSymbolPanel" class="tool-btn" :class="{ active: showSymbolPanel }">
        <i class="icon-grid"></i>
        <span>符号面板</span>
      </button>
      
      <button @click="togglePreview" class="tool-btn" :class="{ active: showPreview }">
        <i class="icon-eye"></i>
        <span>预览</span>
      </button>
      
      <button @click="clearInput" class="tool-btn">
        <i class="icon-trash"></i>
        <span>清空</span>
      </button>
      
      <button @click="toggleCursorInfo" class="tool-btn" :class="{ active: showCursorInfo }">
        <i class="icon-info"></i>
        <span>光标信息</span>
      </button>
    </div>
  </div>
</template>

<script>
import RecommendationService from '@/services/RecommendationService';

export default {
  name: 'SmartSymbolCompletion',
  
  props: {
    value: {
      type: String,
      default: ''
    },
    placeholder: {
      type: String,
      default: '输入数学符号或LaTeX命令...'
    },
    studentModel: {
      type: Object,
      required: true
    },
    context: {
      type: Object,
      default: () => ({})
    },
    showPreview: {
      type: Boolean,
      default: true
    },
    showSymbolPanel: {
      type: Boolean,
      default: false
    },
    autoComplete: {
      type: Boolean,
      default: true
    },
    debounceDelay: {
      type: Number,
      default: 300
    }
  },
  
  data() {
    return {
      inputContent: this.value,
      completions: [],
      showCompletions: false,
      selectedIndex: 0,
      cursorPosition: { row: 0, column: 0 },
      showCursorInfo: false,
      dropdownStyle: {},
      
      // 符号面板相关
      activeCategory: 'basic',
      symbolCategories: [
        { id: 'basic', name: '基础', icon: 'icon-plus' },
        { id: 'greek', name: '希腊', icon: 'icon-alpha' },
        { id: 'calculus', name: '微积分', icon: 'icon-integral' },
        { id: 'geometry', name: '几何', icon: 'icon-triangle' },
        { id: 'algebra', name: '代数', icon: 'icon-function' }
      ],
      
      // 服务实例
      recommendationService: null,
      
      // 防抖定时器
      debounceTimer: null,
      
      // 输入历史
      inputHistory: [],
      historyIndex: -1
    };
  },
  
  computed: {
    currentCategorySymbols() {
      // 这里应该根据activeCategory返回对应的符号列表
      // 暂时返回一些示例符号
      const symbolSets = {
        basic: [
          { id: 'plus', symbol: '+', description: '加号' },
          { id: 'minus', symbol: '−', description: '减号' },
          { id: 'times', symbol: '×', description: '乘号' },
          { id: 'div', symbol: '÷', description: '除号' }
        ],
        greek: [
          { id: 'alpha', symbol: 'α', description: '希腊字母alpha' },
          { id: 'beta', symbol: 'β', description: '希腊字母beta' },
          { id: 'gamma', symbol: 'γ', description: '希腊字母gamma' },
          { id: 'pi', symbol: 'π', description: '圆周率' }
        ],
        calculus: [
          { id: 'integral', symbol: '∫', description: '积分符号' },
          { id: 'sum', symbol: '∑', description: '求和符号' },
          { id: 'partial', symbol: '∂', description: '偏导数' },
          { id: 'infinity', symbol: '∞', description: '无穷大' }
        ],
        geometry: [
          { id: 'angle', symbol: '∠', description: '角度' },
          { id: 'triangle', symbol: '△', description: '三角形' },
          { id: 'perpendicular', symbol: '⊥', description: '垂直' },
          { id: 'parallel', symbol: '∥', description: '平行' }
        ],
        algebra: [
          { id: 'pm', symbol: '±', description: '正负号' },
          { id: 'neq', symbol: '≠', description: '不等于' },
          { id: 'leq', symbol: '≤', description: '小于等于' },
          { id: 'geq', symbol: '≥', description: '大于等于' }
        ]
      };
      
      return symbolSets[this.activeCategory] || [];
    }
  },
  
  watch: {
    value(newValue) {
      if (newValue !== this.inputContent) {
        this.inputContent = newValue;
      }
    },
    
    inputContent(newValue) {
      this.$emit('input', newValue);
    }
  },
  
  mounted() {
    this.recommendationService = new RecommendationService();
    this.updateCursorPosition();
  },
  
  methods: {
    handleInput() {
      this.updateCursorPosition();

      if (this.autoComplete) {
        // 清除之前的防抖定时器
        if (this.debounceTimer) {
          clearTimeout(this.debounceTimer);
        }

        // 设置新的防抖定时器
        this.debounceTimer = setTimeout(() => {
          this.getCompletions();
        }, this.debounceDelay);
      }

      // 更新预览
      if (this.showPreview) {
        this.updatePreview();
      }

      // 发出输入事件
      this.$emit('input-change', {
        content: this.inputContent,
        cursorPosition: this.cursorPosition
      });
    },

    handleKeydown(event) {
      // 处理补全下拉框的键盘导航
      if (this.showCompletions && this.completions.length > 0) {
        switch (event.key) {
          case 'ArrowDown':
            event.preventDefault();
            this.selectedIndex = Math.min(this.selectedIndex + 1, this.completions.length - 1);
            break;
          case 'ArrowUp':
            event.preventDefault();
            this.selectedIndex = Math.max(this.selectedIndex - 1, 0);
            break;
          case 'Enter':
            event.preventDefault();
            if (this.completions[this.selectedIndex]) {
              this.selectCompletion(this.completions[this.selectedIndex]);
            }
            break;
          case 'Escape':
            event.preventDefault();
            this.hideCompletions();
            break;
          case 'Tab':
            event.preventDefault();
            if (this.completions[this.selectedIndex]) {
              this.selectCompletion(this.completions[this.selectedIndex]);
            }
            break;
          default:
            // 数字键快速选择
            if (event.key >= '1' && event.key <= '9') {
              const index = parseInt(event.key) - 1;
              if (index < this.completions.length) {
                event.preventDefault();
                this.selectCompletion(this.completions[index]);
              }
            }
        }
      }

      // 处理其他快捷键
      if (event.ctrlKey || event.metaKey) {
        switch (event.key) {
          case 'z':
            // 撤销功能可以在这里实现
            break;
          case ' ':
            // Ctrl+Space 触发补全
            event.preventDefault();
            this.getCompletions();
            break;
        }
      }

      // 更新光标位置
      this.$nextTick(() => {
        this.updateCursorPosition();
      });
    },

    handleBlur() {
      // 延迟隐藏补全框，以便点击补全项时能正常工作
      setTimeout(() => {
        this.hideCompletions();
      }, 200);
    },

    handleFocus() {
      if (this.autoComplete && this.inputContent) {
        this.getCompletions();
      }
    },

    async getCompletions() {
      if (!this.recommendationService || !this.studentModel) {
        return;
      }

      const context = {
        ...this.context,
        currentInput: this.inputContent,
        cursorPosition: this.getCursorPosition()
      };

      try {
        const completions = await this.recommendationService.getSymbolCompletions(
          this.studentModel,
          context
        );

        this.completions = completions;
        this.selectedIndex = 0;
        this.showCompletions = completions.length > 0;

        if (this.showCompletions) {
          this.updateDropdownPosition();
        }
      } catch (error) {
        console.error('Failed to get completions:', error);
        this.hideCompletions();
      }
    },

    selectCompletion(completion) {
      const textarea = this.$refs.inputArea;
      if (!textarea || !completion) return;

      const cursorPos = this.getCursorPosition();
      const replaceLength = completion.replaceLength || 0;

      // 计算替换位置
      const startPos = cursorPos - replaceLength;
      const endPos = cursorPos;

      // 替换文本
      const newContent =
        this.inputContent.substring(0, startPos) +
        completion.insertText +
        this.inputContent.substring(endPos);

      this.inputContent = newContent;

      // 设置新的光标位置
      const newCursorPos = startPos + completion.insertText.length;
      this.$nextTick(() => {
        textarea.focus();
        textarea.setSelectionRange(newCursorPos, newCursorPos);
        this.updateCursorPosition();
      });

      // 隐藏补全框
      this.hideCompletions();

      // 记录符号使用
      if (this.recommendationService && this.studentModel) {
        this.recommendationService.recordSymbolUsage(
          this.studentModel,
          completion.symbol,
          { ...this.context, currentInput: this.inputContent }
        );
      }

      // 发出选择事件
      this.$emit('completion-selected', completion);
    },

    hideCompletions() {
      this.showCompletions = false;
      this.completions = [];
      this.selectedIndex = 0;
    },

    updateCursorPosition() {
      const textarea = this.$refs.inputArea;
      if (!textarea) return;

      const cursorPos = textarea.selectionStart;
      const textBeforeCursor = this.inputContent.substring(0, cursorPos);
      const lines = textBeforeCursor.split('\n');

      this.cursorPosition = {
        row: lines.length - 1,
        column: lines[lines.length - 1].length,
        absolute: cursorPos
      };
    },

    getCursorPosition() {
      const textarea = this.$refs.inputArea;
      return textarea ? textarea.selectionStart : 0;
    },

    updateDropdownPosition() {
      // 计算下拉框位置，确保不会超出视窗
      this.$nextTick(() => {
        const inputRect = this.$refs.inputArea?.getBoundingClientRect();
        if (inputRect) {
          const viewportHeight = window.innerHeight;
          const spaceBelow = viewportHeight - inputRect.bottom;

          if (spaceBelow < 300) {
            // 空间不足，显示在输入框上方
            this.dropdownStyle = {
              bottom: '100%',
              top: 'auto'
            };
          } else {
            // 显示在输入框下方
            this.dropdownStyle = {
              top: '100%',
              bottom: 'auto'
            };
          }
        }
      });
    },

    updatePreview() {
      // 这里应该集成LaTeX渲染器（如MathJax或KaTeX）
      // 暂时使用简单的文本显示
      if (this.$refs.previewContent) {
        this.$refs.previewContent.innerHTML = `<pre>${this.inputContent}</pre>`;
      }
    },

    togglePreview() {
      this.$emit('update:showPreview', !this.showPreview);
    },

    toggleSymbolPanel() {
      this.$emit('update:showSymbolPanel', !this.showSymbolPanel);
    },

    toggleCursorInfo() {
      this.showCursorInfo = !this.showCursorInfo;
    },

    clearInput() {
      if (this.inputContent && !confirm('确定要清空输入内容吗？')) {
        return;
      }

      this.inputContent = '';
      this.hideCompletions();

      this.$nextTick(() => {
        this.$refs.inputArea?.focus();
      });

      this.$emit('input-cleared');
    },

    switchCategory(categoryId) {
      this.activeCategory = categoryId;
    },

    insertSymbol(symbol) {
      const textarea = this.$refs.inputArea;
      if (!textarea) return;

      const cursorPos = this.getCursorPosition();
      const insertText = symbol.latex || symbol.symbol;

      // 插入符号
      const newContent =
        this.inputContent.substring(0, cursorPos) +
        insertText +
        this.inputContent.substring(cursorPos);

      this.inputContent = newContent;

      // 设置新的光标位置
      const newCursorPos = cursorPos + insertText.length;
      this.$nextTick(() => {
        textarea.focus();
        textarea.setSelectionRange(newCursorPos, newCursorPos);
        this.updateCursorPosition();
      });

      // 记录符号使用
      if (this.recommendationService && this.studentModel) {
        this.recommendationService.recordSymbolUsage(
          this.studentModel,
          symbol.symbol,
          { ...this.context, currentInput: this.inputContent }
        );
      }

      // 发出符号插入事件
      this.$emit('symbol-inserted', symbol);
    },

    getScoreClass(score) {
      if (score > 0.8) return 'high-score';
      if (score > 0.5) return 'medium-score';
      return 'low-score';
    },

    getSourceLabel(sources) {
      if (!sources || sources.length === 0) return '';

      const sourceLabels = {
        'api': 'API',
        'local_context': '上下文',
        'context_analysis': '智能分析',
        'usage_history': '使用历史'
      };

      return sources.map(source => sourceLabels[source] || source).join(', ');
    }
  }
};
</script>

<style scoped>
.smart-symbol-completion {
  position: relative;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.input-container {
  position: relative;
}

.input-wrapper {
  position: relative;
  border: 2px solid #e1e5e9;
  border-radius: 6px;
  transition: border-color 0.3s ease;
}

.input-wrapper:focus-within {
  border-color: #409eff;
}

.input-wrapper.has-completions {
  border-bottom-left-radius: 0;
  border-bottom-right-radius: 0;
}

.symbol-input {
  width: 100%;
  min-height: 120px;
  padding: 12px;
  border: none;
  outline: none;
  font-family: 'Consolas', 'Monaco', monospace;
  font-size: 14px;
  line-height: 1.5;
  resize: vertical;
  background: transparent;
}

.cursor-indicator {
  position: absolute;
  bottom: 8px;
  right: 12px;
  font-size: 12px;
  color: #909399;
  background: rgba(255, 255, 255, 0.9);
  padding: 2px 6px;
  border-radius: 3px;
}

.completion-dropdown {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  background: white;
  border: 2px solid #409eff;
  border-top: none;
  border-radius: 0 0 6px 6px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  z-index: 1000;
  max-height: 300px;
  overflow: hidden;
}

.completion-header {
  display: flex;
  align-items: center;
  padding: 8px 12px;
  background: #f5f7fa;
  border-bottom: 1px solid #e4e7ed;
  font-size: 12px;
  color: #606266;
}

.completion-header i {
  margin-right: 6px;
  color: #409eff;
}

.completion-count {
  margin-left: auto;
  color: #909399;
}

.completion-list {
  max-height: 240px;
  overflow-y: auto;
}

.completion-item {
  display: flex;
  align-items: center;
  padding: 8px 12px;
  cursor: pointer;
  border-bottom: 1px solid #f0f0f0;
  transition: background-color 0.2s ease;
}

.completion-item:hover,
.completion-item.selected {
  background-color: #f5f7fa;
}

.completion-item.selected {
  background-color: #ecf5ff;
  border-left: 3px solid #409eff;
}

.completion-symbol {
  display: flex;
  flex-direction: column;
  align-items: center;
  min-width: 60px;
  margin-right: 12px;
}

.symbol-display {
  font-size: 18px;
  font-weight: bold;
  color: #303133;
}

.latex-display {
  font-size: 10px;
  color: #909399;
  font-family: 'Consolas', 'Monaco', monospace;
  margin-top: 2px;
}

.completion-info {
  flex: 1;
  min-width: 0;
}

.completion-description {
  font-size: 13px;
  color: #606266;
  margin-bottom: 2px;
}

.completion-meta {
  display: flex;
  align-items: center;
  font-size: 11px;
  color: #909399;
}

.score-indicator {
  padding: 1px 4px;
  border-radius: 2px;
  font-weight: bold;
  margin-right: 8px;
}

.score-indicator.high-score {
  background: #f0f9ff;
  color: #409eff;
}

.score-indicator.medium-score {
  background: #fdf6ec;
  color: #e6a23c;
}

.score-indicator.low-score {
  background: #fef0f0;
  color: #f56c6c;
}

.completion-source {
  font-style: italic;
}

.completion-shortcut {
  display: flex;
  align-items: center;
  margin-left: 8px;
}

.completion-shortcut kbd {
  background: #f5f7fa;
  border: 1px solid #dcdfe6;
  border-radius: 3px;
  padding: 2px 6px;
  font-size: 10px;
  color: #606266;
}

.completion-footer {
  padding: 6px 12px;
  background: #fafafa;
  border-top: 1px solid #e4e7ed;
  text-align: center;
}

.help-text {
  font-size: 11px;
  color: #909399;
}

.help-text kbd {
  background: #f5f7fa;
  border: 1px solid #dcdfe6;
  border-radius: 2px;
  padding: 1px 3px;
  margin: 0 2px;
  font-size: 10px;
}

.preview-container {
  margin-top: 12px;
  border: 1px solid #e1e5e9;
  border-radius: 6px;
  background: #fafbfc;
}

.preview-header {
  display: flex;
  align-items: center;
  padding: 8px 12px;
  background: #f5f7fa;
  border-bottom: 1px solid #e4e7ed;
  border-radius: 6px 6px 0 0;
}

.preview-header i {
  margin-right: 6px;
  color: #409eff;
}

.preview-toggle {
  margin-left: auto;
  background: none;
  border: none;
  color: #909399;
  cursor: pointer;
  padding: 4px;
  border-radius: 3px;
}

.preview-toggle:hover {
  background: #e4e7ed;
}

.preview-content {
  padding: 12px;
  min-height: 60px;
  font-family: 'Times New Roman', serif;
  font-size: 16px;
}

.symbol-panel {
  margin-top: 12px;
  border: 1px solid #e1e5e9;
  border-radius: 6px;
  background: white;
}

.panel-header {
  display: flex;
  align-items: center;
  padding: 8px 12px;
  background: #f5f7fa;
  border-bottom: 1px solid #e4e7ed;
  border-radius: 6px 6px 0 0;
}

.panel-header i {
  margin-right: 6px;
  color: #409eff;
}

.panel-toggle {
  margin-left: auto;
  background: none;
  border: none;
  color: #909399;
  cursor: pointer;
  padding: 4px;
  border-radius: 3px;
}

.panel-toggle:hover {
  background: #e4e7ed;
}

.symbol-categories {
  display: flex;
  padding: 8px;
  gap: 4px;
  border-bottom: 1px solid #e4e7ed;
}

.category-btn {
  display: flex;
  align-items: center;
  padding: 6px 12px;
  background: #f5f7fa;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s ease;
  font-size: 12px;
  color: #606266;
}

.category-btn:hover {
  background: #ecf5ff;
  border-color: #b3d8ff;
}

.category-btn.active {
  background: #409eff;
  border-color: #409eff;
  color: white;
}

.category-btn i {
  margin-right: 4px;
}

.symbol-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(40px, 1fr));
  gap: 4px;
  padding: 12px;
}

.symbol-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  background: #f5f7fa;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s ease;
  font-size: 16px;
  color: #303133;
}

.symbol-btn:hover {
  background: #409eff;
  border-color: #409eff;
  color: white;
  transform: translateY(-1px);
  box-shadow: 0 2px 4px rgba(64, 158, 255, 0.3);
}

.toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: #f5f7fa;
  border-top: 1px solid #e4e7ed;
  border-radius: 0 0 8px 8px;
}

.tool-btn {
  display: flex;
  align-items: center;
  padding: 6px 12px;
  background: white;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s ease;
  font-size: 12px;
  color: #606266;
}

.tool-btn:hover {
  background: #ecf5ff;
  border-color: #b3d8ff;
  color: #409eff;
}

.tool-btn.active {
  background: #409eff;
  border-color: #409eff;
  color: white;
}

.tool-btn i {
  margin-right: 4px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .smart-symbol-completion {
    margin: 0;
    border-radius: 0;
  }

  .symbol-grid {
    grid-template-columns: repeat(auto-fill, minmax(35px, 1fr));
  }

  .symbol-btn {
    width: 35px;
    height: 35px;
    font-size: 14px;
  }

  .toolbar {
    flex-wrap: wrap;
    gap: 4px;
  }

  .tool-btn {
    padding: 4px 8px;
    font-size: 11px;
  }

  .completion-dropdown {
    max-height: 200px;
  }
}

/* 深色主题支持 */
@media (prefers-color-scheme: dark) {
  .smart-symbol-completion {
    background: #2d3748;
    color: #e2e8f0;
  }

  .input-wrapper {
    border-color: #4a5568;
    background: #2d3748;
  }

  .symbol-input {
    background: #2d3748;
    color: #e2e8f0;
  }

  .completion-dropdown {
    background: #2d3748;
    border-color: #4a5568;
  }

  .completion-header {
    background: #1a202c;
    border-color: #4a5568;
  }

  .completion-item:hover,
  .completion-item.selected {
    background: #4a5568;
  }
}
</style>
