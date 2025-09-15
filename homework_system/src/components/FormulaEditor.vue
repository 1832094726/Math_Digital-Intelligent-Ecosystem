<template>
  <div class="formula-editor-container">
    <div class="editor-header">
      <h3 class="editor-title">数学公式编辑器</h3>
      <div class="editor-controls">
        <button 
          class="control-btn" 
          @click="toggleSymbolPanel"
          :class="{ 'active': showSymbolPanel }"
          title="符号面板"
        >
          <i class="icon-symbols"></i>
        </button>
        <button 
          class="control-btn" 
          @click="togglePreview"
          :class="{ 'active': showPreview }"
          title="预览"
        >
          <i class="icon-preview"></i>
        </button>
        <button 
          class="control-btn" 
          @click="clearEditor"
          title="清空"
        >
          <i class="icon-clear"></i>
        </button>
      </div>
    </div>
    
    <div class="editor-content">
      <div class="editor-main">
        <div 
          class="editor-area" 
          :class="{ 'with-preview': showPreview }"
        >
          <textarea
            ref="editorTextarea"
            v-model="latexContent"
            @input="handleInput"
            @keydown="handleKeydown"
            @click="updateCursorPosition"
            placeholder="在此输入LaTeX公式..."
            spellcheck="false"
          ></textarea>
          
          <div class="editor-cursor-info" v-if="showCursorInfo">
            <span>行: {{ cursorPosition.row + 1 }}, 列: {{ cursorPosition.column + 1 }}</span>
          </div>
        </div>
        
        <div class="preview-area" v-if="showPreview">
          <div class="preview-header">预览</div>
          <div class="preview-content" ref="previewContent"></div>
        </div>
      </div>
      
      <div class="symbol-panel" v-if="showSymbolPanel">
        <!-- 使用新的智能符号补全组件 -->
        <smart-symbol-completion
          v-model="latexContent"
          :student-model="studentModel"
          :context="editorContext"
          :show-preview="false"
          :show-symbol-panel="true"
          @input-change="handleSmartInput"
          @completion-selected="handleCompletionSelected"
          @symbol-inserted="handleSymbolInserted"
        ></smart-symbol-completion>

        <!-- 保留原有的符号推荐组件作为备选 -->
        <symbol-recommendation
          v-if="showLegacyPanel"
          :recommendation-service="recommendationService"
          :student-model="studentModel"
          :context="editorContext"
          :show-device-adaptation="true"
          @select-symbol="insertSymbol"
          @select-formula="insertFormula"
          @select-knowledge="showKnowledgeInfo"
          @device-changed="handleDeviceChange"
        ></symbol-recommendation>
      </div>
    </div>
    
    <div class="editor-footer">
      <div class="editor-status">
        <span v-if="isSaved" class="status-saved">
          <i class="icon-saved"></i> 已保存
        </span>
        <span v-else class="status-unsaved">
          <i class="icon-unsaved"></i> 未保存
        </span>
      </div>
      
      <div class="editor-actions">
        <button 
          class="action-btn secondary" 
          @click="cancelEdit"
        >
          取消
        </button>
        <button 
          class="action-btn primary" 
          @click="saveFormula"
          :disabled="!latexContent.trim()"
        >
          保存
        </button>
      </div>
    </div>
    
    <!-- 知识点信息弹窗 -->
    <div class="knowledge-modal" v-if="showKnowledgeModal">
      <div class="knowledge-modal-content">
        <div class="knowledge-modal-header">
          <h3>{{ selectedKnowledge.name }}</h3>
          <button class="close-btn" @click="closeKnowledgeModal">
            <i class="icon-close"></i>
          </button>
        </div>
        <div class="knowledge-modal-body">
          <p class="knowledge-description">{{ selectedKnowledge.description }}</p>
          <div class="knowledge-examples" v-if="selectedKnowledge.examples && selectedKnowledge.examples.length">
            <h4>示例:</h4>
            <div 
              v-for="(example, index) in selectedKnowledge.examples" 
              :key="index"
              class="knowledge-example"
            >
              <div class="example-latex" v-html="renderLatex(example.latex)"></div>
              <p class="example-description">{{ example.description }}</p>
              <button 
                class="use-example-btn" 
                @click="useExample(example.latex)"
              >
                使用此示例
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import SymbolRecommendation from './SymbolRecommendation.vue';
import SmartSymbolCompletion from './SmartSymbolCompletion.vue';

export default {
  name: 'FormulaEditor',

  components: {
    SymbolRecommendation,
    SmartSymbolCompletion
  },
  
  props: {
    initialValue: {
      type: String,
      default: ''
    },
    recommendationService: {
      type: Object,
      required: true
    },
    studentModel: {
      type: Object,
      required: true
    }
  },
  
  data() {
    return {
      latexContent: this.initialValue,
      showSymbolPanel: true,
      showPreview: true,
      showCursorInfo: false,
      isSaved: true,
      cursorPosition: {
        row: 0,
        column: 0
      },
      editorContext: {
        beforeCursor: '',
        afterCursor: '',
        currentLine: '',
        recentSymbols: []
      },
      // 知识点弹窗
      showKnowledgeModal: false,
      selectedKnowledge: {
        name: '',
        description: '',
        examples: []
      },
      // 自动保存计时器
      autoSaveTimer: null,
      // 用于渲染LaTeX的对象
      latexRenderer: null,
      // 编辑历史
      history: [],
      historyIndex: -1,
      // 最大历史记录数
      maxHistorySize: 50,
      // 智能补全相关
      showLegacyPanel: false,
      smartCompletionEnabled: true
    };
  },
  
  watch: {
    latexContent: {
      handler(newValue) {
        if (newValue !== this.initialValue) {
          this.isSaved = false;
        }
        
        if (this.showPreview) {
          this.updatePreview();
        }
      }
    }
  },
  
  mounted() {
    // 初始化LaTeX渲染器
    this.initLatexRenderer();
    
    // 初始化预览
    if (this.showPreview && this.latexContent) {
      this.updatePreview();
    }
    
    // 设置自动保存
    this.setupAutoSave();
    
    // 添加初始状态到历史记录
    this.addToHistory(this.latexContent);
    
    // 设置初始光标位置
    this.$nextTick(() => {
      if (this.$refs.editorTextarea) {
        this.$refs.editorTextarea.focus();
        this.updateCursorPosition();
      }
    });
  },
  
  beforeUnmount() {
    // 清除自动保存计时器
    if (this.autoSaveTimer) {
      clearInterval(this.autoSaveTimer);
    }
  },
  
  methods: {
    handleInput() {
      // 更新上下文
      this.updateEditorContext();
      
      // 添加到历史记录
      this.addToHistory(this.latexContent);
    },
    
    handleKeydown(event) {
      // 处理撤销/重做
      if ((event.ctrlKey || event.metaKey) && event.key === 'z') {
        if (event.shiftKey) {
          // Ctrl+Shift+Z 或 Cmd+Shift+Z: 重做
          event.preventDefault();
          this.redo();
        } else {
          // Ctrl+Z 或 Cmd+Z: 撤销
          event.preventDefault();
          this.undo();
        }
      }
      
      // 处理Tab键，插入空格而不是切换焦点
      if (event.key === 'Tab') {
        event.preventDefault();
        this.insertTextAtCursor('  '); // 插入两个空格
      }
      
      // 更新光标位置
      this.$nextTick(() => {
        this.updateCursorPosition();
      });
    },
    
    updateCursorPosition() {
      const textarea = this.$refs.editorTextarea;
      if (!textarea) return;
      
      const cursorPos = textarea.selectionStart;
      const textBeforeCursor = this.latexContent.substring(0, cursorPos);
      const lines = textBeforeCursor.split('\n');
      
      this.cursorPosition = {
        row: lines.length - 1,
        column: lines[lines.length - 1].length
      };
      
      // 更新编辑器上下文
      this.updateEditorContext();
    },
    
    updateEditorContext() {
      const textarea = this.$refs.editorTextarea;
      if (!textarea) return;
      
      const cursorPos = textarea.selectionStart;
      const beforeCursor = this.latexContent.substring(0, cursorPos);
      const afterCursor = this.latexContent.substring(cursorPos);
      
      // 获取当前行
      const lines = this.latexContent.split('\n');
      const currentLineIndex = beforeCursor.split('\n').length - 1;
      const currentLine = lines[currentLineIndex] || '';
      
      // 提取最近使用的符号（简单实现，实际应用中可能需要更复杂的解析）
      const symbolRegex = /\\[a-zA-Z]+|[+\-*/=<>{}[\]()^_]/g;
      const allSymbols = beforeCursor.match(symbolRegex) || [];
      const recentSymbols = allSymbols.slice(-5); // 最近5个符号
      
      this.editorContext = {
        beforeCursor,
        afterCursor,
        currentLine,
        recentSymbols,
        cursorPosition: this.cursorPosition
      };
    },
    
    toggleSymbolPanel() {
      this.showSymbolPanel = !this.showSymbolPanel;
    },
    
    togglePreview() {
      this.showPreview = !this.showPreview;
      
      if (this.showPreview) {
        this.$nextTick(() => {
          this.updatePreview();
        });
      }
    },
    
    clearEditor() {
      if (this.latexContent && !confirm('确定要清空编辑器内容吗？')) {
        return;
      }
      
      this.latexContent = '';
      this.updateEditorContext();
      this.addToHistory('');
      
      // 聚焦编辑器
      this.$nextTick(() => {
        if (this.$refs.editorTextarea) {
          this.$refs.editorTextarea.focus();
        }
      });
    },
    
    insertSymbol(symbol) {
      // 在光标位置插入符号
      this.insertTextAtCursor(symbol.latex || symbol.name);
      
      // 记录学生行为（已在SymbolRecommendation组件中实现）
    },
    
    insertFormula(formula) {
      // 在光标位置插入公式
      this.insertTextAtCursor(formula.latex);
      
      // 记录学生行为（已在SymbolRecommendation组件中实现）
    },
    
    showKnowledgeInfo(knowledge) {
      this.selectedKnowledge = knowledge;
      this.showKnowledgeModal = true;
      
      // 记录学生行为（已在SymbolRecommendation组件中实现）
    },
    
    closeKnowledgeModal() {
      this.showKnowledgeModal = false;
    },
    
    useExample(latexExample) {
      // 将示例插入到编辑器中
      this.latexContent = latexExample;
      this.updateEditorContext();
      this.addToHistory(latexExample);
      this.closeKnowledgeModal();
      
      // 记录学生行为
      this.recordExampleUsage(latexExample);
    },
    
    handleDeviceChange(deviceId) {
      // 设备变更时可能需要调整编辑器布局
      this.$emit('device-changed', deviceId);
    },
    
    insertTextAtCursor(text) {
      const textarea = this.$refs.editorTextarea;
      if (!textarea) return;
      
      const startPos = textarea.selectionStart;
      const endPos = textarea.selectionEnd;
      
      // 插入文本
      this.latexContent = 
        this.latexContent.substring(0, startPos) + 
        text + 
        this.latexContent.substring(endPos);
      
      // 更新光标位置
      this.$nextTick(() => {
        textarea.focus();
        const newCursorPos = startPos + text.length;
        textarea.setSelectionRange(newCursorPos, newCursorPos);
        this.updateCursorPosition();
      });
      
      // 添加到历史记录
      this.addToHistory(this.latexContent);
    },
    
    updatePreview() {
      if (!this.$refs.previewContent || !this.latexRenderer) return;
      
      try {
        // 使用LaTeX渲染器渲染公式
        const rendered = this.renderLatex(this.latexContent);
        this.$refs.previewContent.innerHTML = rendered;
      } catch (error) {
        console.error('LaTeX渲染错误:', error);
        this.$refs.previewContent.innerHTML = `<div class="preview-error">渲染错误: ${error.message}</div>`;
      }
    },
    
    initLatexRenderer() {
      // 实际应用中，这里会初始化真实的LaTeX渲染器，如MathJax或KaTeX
      // 现在使用一个简单的占位实现
      this.latexRenderer = (latex) => {
        return `<div class="latex-preview">${latex}</div>`;
      };
    },
    
    renderLatex(latex) {
      if (!this.latexRenderer) return latex;
      return this.latexRenderer(latex);
    },
    
    setupAutoSave() {
      // 每30秒自动保存一次
      this.autoSaveTimer = setInterval(() => {
        if (!this.isSaved && this.latexContent.trim()) {
          this.autoSave();
        }
      }, 30000);
    },
    
    async autoSave() {
      try {
        // 实际应用中，这里会调用API保存公式
        // 现在只是模拟保存
        await new Promise(resolve => setTimeout(resolve, 300));
        this.isSaved = true;
        
        // 记录自动保存事件
        this.recordEditorAction('auto_save');
      } catch (error) {
        console.error('自动保存失败:', error);
      }
    },
    
    async saveFormula() {
      if (!this.latexContent.trim()) return;
      
      try {
        // 保存前先验证LaTeX语法
        if (!this.validateLatex(this.latexContent)) {
          if (!confirm('LaTeX语法可能有误，是否继续保存？')) {
            return;
          }
        }
        
        // 实际应用中，这里会调用API保存公式
        // 现在只是模拟保存
        await new Promise(resolve => setTimeout(resolve, 500));
        this.isSaved = true;
        
        // 发出保存事件
        this.$emit('save', {
          content: this.latexContent,
          timestamp: new Date().toISOString()
        });
        
        // 记录保存事件
        this.recordEditorAction('save');
      } catch (error) {
        console.error('保存失败:', error);
        alert('保存失败，请重试');
      }
    },
    
    cancelEdit() {
      if (!this.isSaved && this.latexContent !== this.initialValue) {
        if (!confirm('有未保存的更改，确定要取消吗？')) {
          return;
        }
      }
      
      // 发出取消事件
      this.$emit('cancel');
      
      // 记录取消事件
      this.recordEditorAction('cancel');
    },
    
    validateLatex(latex) {
      // 简单的LaTeX语法验证
      // 实际应用中，这里应该有更复杂的验证逻辑
      
      // 检查括号是否匹配
      const brackets = {
        '{': '}',
        '[': ']',
        '(': ')'
      };
      
      const stack = [];
      
      for (let i = 0; i < latex.length; i++) {
        const char = latex[i];
        
        if ('{[('.includes(char)) {
          stack.push(char);
        } else if ('}])'.includes(char)) {
          const lastBracket = stack.pop();
          if (brackets[lastBracket] !== char) {
            return false;
          }
        }
      }
      
      return stack.length === 0;
    },
    
    addToHistory(content) {
      // 如果当前不是最新状态，则删除当前之后的历史
      if (this.historyIndex < this.history.length - 1) {
        this.history = this.history.slice(0, this.historyIndex + 1);
      }
      
      // 如果内容与最后一个历史记录相同，则不添加
      if (this.history.length > 0 && this.history[this.history.length - 1] === content) {
        return;
      }
      
      // 添加新的历史记录
      this.history.push(content);
      
      // 如果历史记录超过最大数量，则删除最早的记录
      if (this.history.length > this.maxHistorySize) {
        this.history.shift();
      }
      
      // 更新历史索引
      this.historyIndex = this.history.length - 1;
    },
    
    undo() {
      if (this.historyIndex <= 0) return;
      
      this.historyIndex--;
      this.latexContent = this.history[this.historyIndex];
      this.updateEditorContext();
      
      // 记录撤销事件
      this.recordEditorAction('undo');
    },
    
    redo() {
      if (this.historyIndex >= this.history.length - 1) return;
      
      this.historyIndex++;
      this.latexContent = this.history[this.historyIndex];
      this.updateEditorContext();
      
      // 记录重做事件
      this.recordEditorAction('redo');
    },
    
    recordEditorAction(actionType, details = {}) {
      // 记录编辑器操作到学生模型
      if (!this.studentModel) return;
      
      this.studentModel.recordBehavior('editor_action', {
        actionType,
        contentLength: this.latexContent.length,
        timestamp: new Date().toISOString(),
        ...details
      });
    },
    
    recordExampleUsage(example) {
      // 记录使用示例的行为
      if (!this.studentModel) return;

      this.studentModel.recordBehavior('example_usage', {
        exampleLength: example.length,
        timestamp: new Date().toISOString()
      });
    },

    // 智能符号补全相关方法
    handleSmartInput(inputData) {
      // 处理智能输入变化
      this.latexContent = inputData.content;
      this.cursorPosition = inputData.cursorPosition;
      this.updateEditorContext();

      // 记录智能输入行为
      this.recordEditorAction('smart_input', {
        inputLength: inputData.content.length,
        cursorPosition: inputData.cursorPosition
      });
    },

    handleCompletionSelected(completion) {
      // 处理补全选择
      console.log('Completion selected:', completion);

      // 记录补全使用
      this.recordEditorAction('completion_used', {
        completionSymbol: completion.symbol,
        completionScore: completion.score,
        completionSource: completion.sources?.join(',') || 'unknown'
      });

      // 发出补全选择事件
      this.$emit('completion-selected', completion);
    },

    handleSymbolInserted(symbol) {
      // 处理符号插入
      console.log('Symbol inserted:', symbol);

      // 记录符号插入
      this.recordEditorAction('symbol_inserted', {
        symbolName: symbol.symbol,
        symbolCategory: symbol.category || 'unknown'
      });

      // 发出符号插入事件
      this.$emit('symbol-inserted', symbol);
    },

    toggleLegacyPanel() {
      // 切换传统符号面板
      this.showLegacyPanel = !this.showLegacyPanel;
    },

    toggleSmartCompletion() {
      // 切换智能补全功能
      this.smartCompletionEnabled = !this.smartCompletionEnabled;

      // 记录功能切换
      this.recordEditorAction('feature_toggle', {
        feature: 'smart_completion',
        enabled: this.smartCompletionEnabled
      });
    }
  }
};
</script>

<style scoped>
.formula-editor-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 400px;
  background-color: var(--background-light);
  border-radius: 8px;
  overflow: hidden;
}

.editor-header {
  display: flex;
  align-items: center;
  padding: 10px 15px;
  background-color: var(--primary-color);
  color: white;
}

.editor-title {
  margin: 0;
  font-size: 18px;
  font-weight: 500;
}

.editor-controls {
  margin-left: auto;
  display: flex;
  gap: 5px;
}

.control-btn {
  background: none;
  border: none;
  color: white;
  width: 32px;
  height: 32px;
  border-radius: 4px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
}

.control-btn:hover {
  background-color: rgba(255, 255, 255, 0.2);
}

.control-btn.active {
  background-color: rgba(255, 255, 255, 0.3);
}

.editor-content {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.editor-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.editor-area {
  flex: 1;
  position: relative;
  overflow: hidden;
}

.editor-area.with-preview {
  height: 50%;
}

.editor-area textarea {
  width: 100%;
  height: 100%;
  padding: 15px;
  border: none;
  resize: none;
  font-family: 'Courier New', monospace;
  font-size: 16px;
  line-height: 1.5;
  color: var(--text-color);
  background-color: white;
}

.editor-area textarea:focus {
  outline: none;
}

.editor-cursor-info {
  position: absolute;
  bottom: 5px;
  right: 10px;
  font-size: 12px;
  color: var(--light-text);
  background-color: rgba(255, 255, 255, 0.8);
  padding: 2px 5px;
  border-radius: 3px;
}

.preview-area {
  height: 50%;
  border-top: 1px solid var(--border-color);
  background-color: white;
  overflow: auto;
}

.preview-header {
  padding: 5px 15px;
  background-color: var(--background-dark);
  font-size: 14px;
  font-weight: 500;
  color: var(--text-color);
}

.preview-content {
  padding: 15px;
  min-height: 100px;
}

.preview-error {
  color: #dc3545;
  padding: 10px;
  background-color: #f8d7da;
  border-radius: 4px;
}

.symbol-panel {
  width: 300px;
  border-left: 1px solid var(--border-color);
  overflow-y: auto;
}

.editor-footer {
  display: flex;
  align-items: center;
  padding: 10px 15px;
  background-color: var(--background-dark);
  border-top: 1px solid var(--border-color);
}

.editor-status {
  font-size: 14px;
}

.status-saved {
  color: #28a745;
}

.status-unsaved {
  color: #dc3545;
}

.editor-actions {
  margin-left: auto;
  display: flex;
  gap: 10px;
}

.action-btn {
  padding: 8px 15px;
  border: none;
  border-radius: 4px;
  font-size: 14px;
  cursor: pointer;
  transition: var(--transition);
}

.action-btn.primary {
  background-color: var(--primary-color);
  color: white;
}

.action-btn.primary:hover {
  background-color: var(--primary-dark);
}

.action-btn.primary:disabled {
  background-color: var(--primary-color);
  opacity: 0.6;
  cursor: not-allowed;
}

.action-btn.secondary {
  background-color: var(--background-dark);
  color: var(--text-color);
}

.action-btn.secondary:hover {
  background-color: var(--border-color);
}

/* 知识点弹窗 */
.knowledge-modal {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.knowledge-modal-content {
  background-color: white;
  border-radius: 8px;
  width: 80%;
  max-width: 600px;
  max-height: 80vh;
  overflow-y: auto;
}

.knowledge-modal-header {
  display: flex;
  align-items: center;
  padding: 15px;
  border-bottom: 1px solid var(--border-color);
}

.knowledge-modal-header h3 {
  margin: 0;
  font-size: 18px;
  color: var(--text-color);
}

.close-btn {
  margin-left: auto;
  background: none;
  border: none;
  font-size: 20px;
  color: var(--light-text);
  cursor: pointer;
}

.close-btn:hover {
  color: var(--text-color);
}

.knowledge-modal-body {
  padding: 15px;
}

.knowledge-description {
  margin-top: 0;
  margin-bottom: 20px;
  line-height: 1.6;
  color: var(--text-color);
}

.knowledge-examples h4 {
  margin-top: 0;
  margin-bottom: 10px;
  font-size: 16px;
  color: var(--text-color);
}

.knowledge-example {
  padding: 15px;
  margin-bottom: 15px;
  background-color: var(--background-light);
  border-radius: 6px;
}

.example-latex {
  margin-bottom: 10px;
  padding: 10px;
  background-color: white;
  border-radius: 4px;
  border: 1px solid var(--border-color);
}

.example-description {
  margin: 10px 0;
  font-size: 14px;
  color: var(--light-text);
}

.use-example-btn {
  background-color: var(--primary-color);
  color: white;
  border: none;
  border-radius: 4px;
  padding: 6px 12px;
  font-size: 14px;
  cursor: pointer;
  transition: var(--transition);
}

.use-example-btn:hover {
  background-color: var(--primary-dark);
}

/* 响应式调整 */
@media (max-width: 768px) {
  .editor-content {
    flex-direction: column;
  }
  
  .symbol-panel {
    width: 100%;
    height: 300px;
    border-left: none;
    border-top: 1px solid var(--border-color);
  }
  
  .editor-area.with-preview,
  .preview-area {
    height: 300px;
  }
}
</style> 