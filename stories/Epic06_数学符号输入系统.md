# Epic 06: 数学符号输入系统

## 史诗概述

### 史诗标题
数学符号输入系统

### 史诗描述
构建智能化的数学符号输入系统，集成动态键盘、符号推荐、公式编辑器等功能，支持多种输入方式（键盘、触摸、手写），为学生提供便捷、准确的数学表达式输入体验。

### 业务价值
- 提高学生数学表达式输入效率
- 降低符号输入的学习成本和错误率
- 支持复杂数学公式的可视化编辑
- 提供智能化的符号推荐和自动补全
- 适配多种设备和输入方式

### 验收标准
- [ ] 支持常用数学符号的快速输入
- [ ] 提供智能符号推荐和上下文感知
- [ ] 实现可视化的公式编辑功能
- [ ] 支持多设备适配和响应式设计
- [ ] 符号输入准确率达到95%以上

## 用户故事

### Story 6.1: 动态数学键盘

**作为** 学生  
**我希望** 有一个专门的数学符号键盘  
**以便** 快速输入各种数学符号

#### 验收标准
- [ ] 提供分类的数学符号键盘（运算、关系、函数、几何等）
- [ ] 支持符号键盘的动态切换和自定义
- [ ] 根据当前输入内容智能调整键盘布局
- [ ] 支持符号的快速搜索和定位
- [ ] 提供符号的使用说明和示例
- [ ] 支持常用符号的快捷访问
- [ ] 适配不同屏幕尺寸和设备类型

#### 技术任务
- [ ] 设计数学符号键盘UI组件 `MathKeyboard.vue`
- [ ] 实现符号分类和布局算法
- [ ] 创建符号搜索和筛选功能
- [ ] 实现键盘布局的响应式设计
- [ ] 添加符号使用统计和优化
- [ ] 创建键盘配置和个性化设置
- [ ] 实现键盘主题和样式定制

#### 数据库变更
- [ ] 完善math_symbols表的键盘位置字段
- [ ] 创建用户键盘配置表
- [ ] 添加符号使用频率统计表
- [ ] 优化符号查询和分类索引

### Story 6.2: 智能符号推荐集成

**作为** 学生  
**我希望** 在输入时获得智能的符号推荐  
**以便** 减少查找符号的时间

#### 验收标准
- [ ] 基于当前输入内容推荐相关符号
- [ ] 根据用户历史使用习惯个性化推荐
- [ ] 支持上下文感知的智能推荐
- [ ] 提供推荐符号的快速插入功能
- [ ] 显示推荐理由和使用建议
- [ ] 支持推荐结果的学习和优化
- [ ] 实现推荐的实时更新

#### 技术任务
- [ ] 集成符号推荐API到键盘组件
- [ ] 实现推荐结果的实时显示
- [ ] 创建推荐符号的快速插入机制
- [ ] 实现推荐学习和反馈收集
- [ ] 添加推荐结果的缓存机制
- [ ] 创建推荐效果监控和优化
- [ ] 实现离线推荐功能

#### 数据库变更
- [ ] 利用现有推荐系统数据表
- [ ] 添加键盘推荐使用统计
- [ ] 创建推荐效果评估表
- [ ] 优化推荐查询性能

### Story 6.3: 公式可视化编辑器

**作为** 学生  
**我希望** 能够可视化地编辑复杂的数学公式  
**以便** 准确输入复杂的数学表达式

#### 验收标准
- [ ] 提供所见即所得的公式编辑界面
- [ ] 支持分数、根式、指数、积分等复杂结构
- [ ] 实现公式的拖拽和结构调整
- [ ] 支持公式模板的快速应用
- [ ] 提供公式语法检查和错误提示
- [ ] 支持公式的实时预览和渲染
- [ ] 实现公式的导入导出功能

#### 技术任务
- [ ] 集成或开发公式编辑器组件
- [ ] 实现MathML和LaTeX的相互转换
- [ ] 创建公式模板库和管理系统
- [ ] 实现公式语法验证器
- [ ] 添加公式渲染引擎（MathJax/KaTeX）
- [ ] 创建公式编辑工具栏
- [ ] 实现公式的撤销重做功能

#### 数据库变更
- [ ] 创建公式模板存储表
- [ ] 添加用户公式历史表
- [ ] 创建公式语法规则表
- [ ] 优化公式数据存储格式

### Story 6.4: 手写识别输入

**作为** 学生  
**我希望** 能够通过手写输入数学符号和公式  
**以便** 更自然地表达数学内容

#### 验收标准
- [ ] 支持手写数学符号的识别
- [ ] 识别准确率达到90%以上
- [ ] 支持连续手写和分段识别
- [ ] 提供识别结果的修正功能
- [ ] 支持手写公式的结构化识别
- [ ] 实现手写轨迹的平滑处理
- [ ] 适配不同的手写习惯和风格

#### 技术任务
- [ ] 集成手写识别引擎
- [ ] 实现手写输入界面组件 `HandwritingInput.vue`
- [ ] 创建手写轨迹收集和处理
- [ ] 实现识别结果的后处理和优化
- [ ] 添加识别模型的训练和更新
- [ ] 创建手写样本收集和标注
- [ ] 实现识别准确率监控

#### 数据库变更
- [ ] 创建手写样本数据表
- [ ] 添加识别结果统计表
- [ ] 创建用户手写习惯表
- [ ] 优化识别数据存储

### Story 6.5: 多设备适配优化

**作为** 学生  
**我希望** 在不同设备上都能方便地输入数学符号  
**以便** 随时随地进行数学学习

#### 验收标准
- [ ] 适配桌面端、平板端、手机端不同屏幕尺寸
- [ ] 支持触摸和鼠标两种交互方式
- [ ] 优化移动端的触摸体验和响应速度
- [ ] 实现设备间的输入配置同步
- [ ] 支持横竖屏切换的布局适配
- [ ] 优化不同设备的键盘布局
- [ ] 提供设备特定的输入优化

#### 技术任务
- [ ] 实现响应式布局设计
- [ ] 优化触摸事件处理和手势识别
- [ ] 创建设备检测和适配逻辑
- [ ] 实现配置云端同步功能
- [ ] 优化移动端性能和渲染
- [ ] 创建设备特定的UI组件
- [ ] 实现输入方式的智能切换

#### 数据库变更
- [ ] 添加设备配置同步表
- [ ] 创建设备使用统计表
- [ ] 优化移动端数据查询
- [ ] 添加设备适配日志表

### Story 6.6: 符号输入辅助功能

**作为** 学生  
**我希望** 有辅助功能帮助我更好地使用符号输入系统  
**以便** 提高输入效率和准确性

#### 验收标准
- [ ] 提供符号输入的快捷键支持
- [ ] 实现符号的自动补全和联想
- [ ] 支持符号输入的语音提示
- [ ] 提供符号使用的帮助文档和教程
- [ ] 实现输入历史的记录和回溯
- [ ] 支持符号收藏和个人符号库
- [ ] 提供输入错误的智能纠错

#### 技术任务
- [ ] 实现快捷键绑定和管理系统
- [ ] 创建符号自动补全算法
- [ ] 集成语音合成和提示功能
- [ ] 创建帮助文档和教程系统
- [ ] 实现输入历史记录和管理
- [ ] 创建个人符号库功能
- [ ] 实现智能纠错和建议系统

#### 数据库变更
- [ ] 创建用户快捷键配置表
- [ ] 添加输入历史记录表
- [ ] 创建个人符号收藏表
- [ ] 添加帮助文档访问统计

### Story 6.7: 符号输入性能优化

**作为** 系统  
**我需要** 优化符号输入系统的性能  
**以便** 提供流畅的用户体验

#### 验收标准
- [ ] 符号键盘响应时间控制在50ms以内
- [ ] 支持1000+符号的快速加载和渲染
- [ ] 实现符号数据的懒加载和分页
- [ ] 优化符号搜索的响应速度
- [ ] 减少内存占用和CPU消耗
- [ ] 实现符号缓存和预加载机制
- [ ] 优化网络请求和数据传输

#### 技术任务
- [ ] 实现符号数据的虚拟滚动
- [ ] 优化符号渲染和DOM操作
- [ ] 创建符号预加载和缓存策略
- [ ] 实现搜索算法优化
- [ ] 添加性能监控和分析
- [ ] 优化网络请求合并和压缩
- [ ] 实现WebWorker后台处理

#### 数据库变更
- [ ] 优化符号查询索引
- [ ] 实现符号数据分页查询
- [ ] 添加性能监控数据表
- [ ] 优化符号数据存储格式

### Story 6.8: 符号输入数据分析

**作为** 产品经理  
**我希望** 了解符号输入系统的使用情况  
**以便** 优化产品功能和用户体验

#### 验收标准
- [ ] 统计符号使用频率和分布
- [ ] 分析用户输入行为和习惯
- [ ] 监控输入错误和纠错情况
- [ ] 评估推荐系统的效果
- [ ] 分析不同设备的使用差异
- [ ] 生成符号输入分析报告
- [ ] 提供产品优化建议

#### 技术任务
- [ ] 实现符号使用数据收集
- [ ] 创建输入行为分析算法
- [ ] 实现错误统计和分析
- [ ] 创建推荐效果评估机制
- [ ] 实现设备使用对比分析
- [ ] 创建分析报告生成器
- [ ] 添加数据可视化展示

#### 数据库变更
- [ ] 完善符号使用统计表
- [ ] 创建输入行为分析表
- [ ] 添加错误统计和分类表
- [ ] 创建分析报告数据表

## 技术实现要点

### 数学键盘组件
```vue
<!-- 数学符号键盘组件 -->
<template>
  <div class="math-keyboard" :class="{ 'mobile': isMobile }">
    <!-- 键盘工具栏 -->
    <div class="keyboard-toolbar">
      <div class="category-tabs">
        <div 
          v-for="category in symbolCategories"
          :key="category.name"
          class="tab-item"
          :class="{ active: activeCategory === category.name }"
          @click="switchCategory(category.name)"
        >
          <i :class="category.icon"></i>
          <span>{{ category.label }}</span>
        </div>
      </div>
      
      <div class="keyboard-actions">
        <el-input 
          v-model="searchQuery"
          placeholder="搜索符号..."
          size="small"
          @input="handleSearch"
        />
        <el-button size="small" @click="toggleRecommendation">
          <i class="el-icon-magic-stick"></i>
          推荐
        </el-button>
      </div>
    </div>
    
    <!-- 推荐符号区域 -->
    <div v-if="showRecommendation" class="recommendation-section">
      <h4>推荐符号</h4>
      <div class="symbol-grid">
        <div 
          v-for="symbol in recommendedSymbols"
          :key="symbol.id"
          class="symbol-item recommended"
          @click="insertSymbol(symbol)"
        >
          <span class="symbol-text">{{ symbol.symbol_text }}</span>
          <span class="symbol-name">{{ symbol.symbol_name }}</span>
          <span class="confidence">{{ Math.round(symbol.confidence * 100) }}%</span>
        </div>
      </div>
    </div>
    
    <!-- 符号键盘主体 -->
    <div class="keyboard-body">
      <div class="symbol-grid">
        <div 
          v-for="symbol in filteredSymbols"
          :key="symbol.id"
          class="symbol-item"
          :class="{ 
            'frequent': symbol.is_common,
            'recent': recentSymbols.includes(symbol.id)
          }"
          @click="insertSymbol(symbol)"
          @contextmenu.prevent="showSymbolMenu(symbol, $event)"
        >
          <span class="symbol-text">{{ symbol.symbol_text }}</span>
          <span class="symbol-name">{{ symbol.symbol_name }}</span>
          <span v-if="symbol.latex_code" class="latex-code">{{ symbol.latex_code }}</span>
        </div>
      </div>
    </div>
    
    <!-- 符号详情弹窗 -->
    <el-dialog 
      title="符号详情" 
      :visible.sync="showSymbolDetail"
      width="400px"
    >
      <div v-if="selectedSymbol" class="symbol-detail">
        <div class="symbol-preview">
          <span class="large-symbol">{{ selectedSymbol.symbol_text }}</span>
          <h3>{{ selectedSymbol.symbol_name }}</h3>
        </div>
        
        <div class="symbol-info">
          <p><strong>分类：</strong>{{ selectedSymbol.category }}</p>
          <p><strong>LaTeX：</strong><code>{{ selectedSymbol.latex_code }}</code></p>
          <p><strong>描述：</strong>{{ selectedSymbol.description }}</p>
        </div>
        
        <div v-if="selectedSymbol.usage_examples" class="usage-examples">
          <h4>使用示例：</h4>
          <div 
            v-for="example in selectedSymbol.usage_examples"
            :key="example"
            class="example-item"
          >
            {{ example }}
          </div>
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<script>
export default {
  name: 'MathKeyboard',
  props: {
    context: String,
    targetInput: Object
  },
  data() {
    return {
      activeCategory: 'common',
      searchQuery: '',
      showRecommendation: true,
      showSymbolDetail: false,
      selectedSymbol: null,
      symbols: [],
      recommendedSymbols: [],
      recentSymbols: [],
      symbolCategories: [
        { name: 'common', label: '常用', icon: 'el-icon-star-on' },
        { name: 'operators', label: '运算', icon: 'el-icon-plus' },
        { name: 'relations', label: '关系', icon: 'el-icon-connection' },
        { name: 'functions', label: '函数', icon: 'el-icon-data-line' },
        { name: 'geometry', label: '几何', icon: 'el-icon-pie-chart' },
        { name: 'greek', label: '希腊', icon: 'el-icon-document' }
      ]
    };
  },
  computed: {
    isMobile() {
      return this.$store.getters.isMobile;
    },
    filteredSymbols() {
      let symbols = this.symbols.filter(s => s.category === this.activeCategory);
      
      if (this.searchQuery) {
        const query = this.searchQuery.toLowerCase();
        symbols = symbols.filter(s => 
          s.symbol_name.toLowerCase().includes(query) ||
          s.symbol_text.includes(query) ||
          (s.latex_code && s.latex_code.toLowerCase().includes(query))
        );
      }
      
      return symbols.sort((a, b) => {
        // 常用符号排在前面
        if (a.is_common && !b.is_common) return -1;
        if (!a.is_common && b.is_common) return 1;
        
        // 最近使用的排在前面
        const aRecent = this.recentSymbols.includes(a.id);
        const bRecent = this.recentSymbols.includes(b.id);
        if (aRecent && !bRecent) return -1;
        if (!aRecent && bRecent) return 1;
        
        // 按频率排序
        return b.frequency_score - a.frequency_score;
      });
    }
  },
  async mounted() {
    await this.loadSymbols();
    await this.loadRecommendations();
    this.loadRecentSymbols();
  },
  watch: {
    context: {
      handler: 'loadRecommendations',
      immediate: true
    }
  },
  methods: {
    async loadSymbols() {
      try {
        const response = await this.$api.get('/symbols/list');
        this.symbols = response.data;
      } catch (error) {
        console.error('加载符号失败:', error);
      }
    },
    
    async loadRecommendations() {
      if (!this.context) return;
      
      try {
        const response = await this.$api.post('/recommend/symbols', {
          context: this.context,
          limit: 8
        });
        this.recommendedSymbols = response.data.recommendations;
      } catch (error) {
        console.error('加载推荐符号失败:', error);
      }
    },
    
    loadRecentSymbols() {
      const recent = localStorage.getItem('recent_symbols');
      this.recentSymbols = recent ? JSON.parse(recent) : [];
    },
    
    insertSymbol(symbol) {
      // 插入符号到目标输入框
      this.$emit('symbol-inserted', symbol);
      
      // 记录使用历史
      this.recordSymbolUsage(symbol);
      
      // 更新最近使用
      this.updateRecentSymbols(symbol.id);
    },
    
    async recordSymbolUsage(symbol) {
      try {
        await this.$api.post('/symbols/usage', {
          symbol_id: symbol.id,
          context: this.context
        });
      } catch (error) {
        console.error('记录符号使用失败:', error);
      }
    },
    
    updateRecentSymbols(symbolId) {
      let recent = [...this.recentSymbols];
      recent = recent.filter(id => id !== symbolId);
      recent.unshift(symbolId);
      recent = recent.slice(0, 10); // 只保留最近10个
      
      this.recentSymbols = recent;
      localStorage.setItem('recent_symbols', JSON.stringify(recent));
    },
    
    switchCategory(category) {
      this.activeCategory = category;
      this.searchQuery = '';
    },
    
    handleSearch() {
      // 搜索功能已通过computed属性实现
    },
    
    toggleRecommendation() {
      this.showRecommendation = !this.showRecommendation;
    },
    
    showSymbolMenu(symbol, event) {
      // 显示符号右键菜单（收藏、详情等）
      this.selectedSymbol = symbol;
      this.showSymbolDetail = true;
    }
  }
};
</script>

<style lang="scss" scoped>
.math-keyboard {
  border: 1px solid #dcdfe6;
  border-radius: 8px;
  background: #fff;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  
  &.mobile {
    .symbol-item {
      min-height: 44px; // 移动端触摸友好尺寸
    }
  }
  
  .keyboard-toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    border-bottom: 1px solid #e4e7ed;
    
    .category-tabs {
      display: flex;
      gap: 8px;
      
      .tab-item {
        display: flex;
        align-items: center;
        gap: 4px;
        padding: 6px 12px;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.3s;
        
        &:hover {
          background: #f5f7fa;
        }
        
        &.active {
          background: #409eff;
          color: white;
        }
      }
    }
  }
  
  .recommendation-section {
    padding: 12px 16px;
    background: #f8f9fa;
    border-bottom: 1px solid #e4e7ed;
    
    .symbol-item.recommended {
      background: linear-gradient(45deg, #409eff, #67c23a);
      color: white;
      
      .confidence {
        font-size: 10px;
        opacity: 0.8;
      }
    }
  }
  
  .symbol-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
    gap: 8px;
    padding: 16px;
    
    .symbol-item {
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 12px 8px;
      border: 1px solid #e4e7ed;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.3s;
      min-height: 60px;
      
      &:hover {
        border-color: #409eff;
        box-shadow: 0 2px 8px rgba(64, 158, 255, 0.2);
      }
      
      &.frequent {
        border-color: #67c23a;
        background: #f0f9ff;
      }
      
      &.recent {
        border-color: #e6a23c;
        background: #fdf6ec;
      }
      
      .symbol-text {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 4px;
      }
      
      .symbol-name {
        font-size: 10px;
        color: #606266;
        text-align: center;
      }
      
      .latex-code {
        font-size: 8px;
        color: #909399;
        font-family: 'Courier New', monospace;
      }
    }
  }
}
</style>
```

### 公式编辑器集成
```javascript
// 公式编辑器服务
class FormulaEditorService {
  constructor() {
    this.editor = null;
    this.mathField = null;
  }
  
  initialize(container, options = {}) {
    // 初始化MathQuill编辑器
    this.mathField = MQ.MathField(container, {
      spaceBehavesLikeTab: true,
      leftRightIntoCmdGoes: 'up',
      restrictMismatchedBrackets: true,
      sumStartsWithNEquals: true,
      supSubsRequireOperand: true,
      charsThatBreakOutOfSupSub: '+-=<>',
      autoSubscriptNumerals: true,
      autoCommands: 'pi theta sqrt sum prod alpha beta gamma delta epsilon',
      autoOperatorNames: 'sin cos tan log ln exp lim max min',
      ...options
    });
    
    return this.mathField;
  }
  
  insertSymbol(symbol) {
    if (!this.mathField) return;
    
    if (symbol.latex_code) {
      this.mathField.cmd(symbol.latex_code);
    } else {
      this.mathField.write(symbol.symbol_text);
    }
    
    this.mathField.focus();
  }
  
  insertTemplate(template) {
    if (!this.mathField) return;
    
    this.mathField.cmd(template.latex_code);
    this.mathField.focus();
  }
  
  getLatex() {
    return this.mathField ? this.mathField.latex() : '';
  }
  
  setLatex(latex) {
    if (this.mathField) {
      this.mathField.latex(latex);
    }
  }
  
  clear() {
    if (this.mathField) {
      this.mathField.latex('');
    }
  }
  
  validate() {
    const latex = this.getLatex();
    
    // 基础语法检查
    const errors = [];
    
    // 检查括号匹配
    if (!this.checkBracketMatching(latex)) {
      errors.push('括号不匹配');
    }
    
    // 检查必需参数
    if (!this.checkRequiredParameters(latex)) {
      errors.push('缺少必需参数');
    }
    
    return {
      isValid: errors.length === 0,
      errors
    };
  }
  
  checkBracketMatching(latex) {
    const brackets = { '(': ')', '[': ']', '{': '}' };
    const stack = [];
    
    for (let char of latex) {
      if (brackets[char]) {
        stack.push(brackets[char]);
      } else if (Object.values(brackets).includes(char)) {
        if (stack.pop() !== char) {
          return false;
        }
      }
    }
    
    return stack.length === 0;
  }
  
  checkRequiredParameters(latex) {
    // 检查需要参数的命令
    const commandsNeedingParams = ['\\frac', '\\sqrt', '\\sum', '\\int'];
    
    for (let cmd of commandsNeedingParams) {
      const regex = new RegExp(`\\${cmd}\\s*(?!\\{|\\[)`, 'g');
      if (regex.test(latex)) {
        return false;
      }
    }
    
    return true;
  }
}
```

### 手写识别集成
```javascript
// 手写识别服务
class HandwritingRecognitionService {
  constructor() {
    this.canvas = null;
    this.ctx = null;
    this.isDrawing = false;
    this.strokes = [];
    this.currentStroke = [];
  }
  
  initialize(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    
    // 设置绘制样式
    this.ctx.strokeStyle = '#333';
    this.ctx.lineWidth = 3;
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';
    
    this.bindEvents();
  }
  
  bindEvents() {
    // 鼠标事件
    this.canvas.addEventListener('mousedown', this.startDrawing.bind(this));
    this.canvas.addEventListener('mousemove', this.draw.bind(this));
    this.canvas.addEventListener('mouseup', this.stopDrawing.bind(this));
    
    // 触摸事件
    this.canvas.addEventListener('touchstart', this.handleTouch.bind(this));
    this.canvas.addEventListener('touchmove', this.handleTouch.bind(this));
    this.canvas.addEventListener('touchend', this.handleTouch.bind(this));
  }
  
  startDrawing(e) {
    this.isDrawing = true;
    const point = this.getPoint(e);
    this.currentStroke = [point];
    this.ctx.beginPath();
    this.ctx.moveTo(point.x, point.y);
  }
  
  draw(e) {
    if (!this.isDrawing) return;
    
    const point = this.getPoint(e);
    this.currentStroke.push(point);
    
    this.ctx.lineTo(point.x, point.y);
    this.ctx.stroke();
  }
  
  stopDrawing() {
    if (!this.isDrawing) return;
    
    this.isDrawing = false;
    this.strokes.push([...this.currentStroke]);
    this.currentStroke = [];
    
    // 延迟识别，避免频繁调用
    clearTimeout(this.recognitionTimeout);
    this.recognitionTimeout = setTimeout(() => {
      this.recognizeStrokes();
    }, 500);
  }
  
  getPoint(e) {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: (e.clientX || e.touches[0].clientX) - rect.left,
      y: (e.clientY || e.touches[0].clientY) - rect.top,
      timestamp: Date.now()
    };
  }
  
  handleTouch(e) {
    e.preventDefault();
    
    switch (e.type) {
      case 'touchstart':
        this.startDrawing(e);
        break;
      case 'touchmove':
        this.draw(e);
        break;
      case 'touchend':
        this.stopDrawing();
        break;
    }
  }
  
  async recognizeStrokes() {
    if (this.strokes.length === 0) return;
    
    try {
      const response = await fetch('/api/handwriting/recognize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          strokes: this.strokes,
          canvas_size: {
            width: this.canvas.width,
            height: this.canvas.height
          }
        })
      });
      
      const result = await response.json();
      
      if (result.success) {
        this.onRecognitionResult(result.data);
      }
    } catch (error) {
      console.error('手写识别失败:', error);
    }
  }
  
  onRecognitionResult(results) {
    // 显示识别结果供用户选择
    this.showRecognitionResults(results);
  }
  
  showRecognitionResults(results) {
    // 创建结果选择界面
    const resultsContainer = document.createElement('div');
    resultsContainer.className = 'recognition-results';
    
    results.forEach((result, index) => {
      const item = document.createElement('div');
      item.className = 'result-item';
      item.innerHTML = `
        <span class="symbol">${result.symbol}</span>
        <span class="confidence">${Math.round(result.confidence * 100)}%</span>
      `;
      
      item.addEventListener('click', () => {
        this.selectRecognitionResult(result);
        resultsContainer.remove();
      });
      
      resultsContainer.appendChild(item);
    });
    
    document.body.appendChild(resultsContainer);
  }
  
  selectRecognitionResult(result) {
    // 触发符号选择事件
    this.onSymbolSelected(result);
    this.clear();
  }
  
  clear() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.strokes = [];
    this.currentStroke = [];
  }
  
  onSymbolSelected(symbol) {
    // 由外部设置回调函数
  }
}
```

## 测试策略

### 功能测试
- [ ] 符号输入准确性测试
- [ ] 推荐系统效果测试
- [ ] 手写识别准确率测试
- [ ] 公式编辑器功能测试

### 兼容性测试
- [ ] 多浏览器兼容性测试
- [ ] 不同设备适配测试
- [ ] 触摸和鼠标交互测试

### 性能测试
- [ ] 符号加载和渲染性能测试
- [ ] 手写识别响应时间测试
- [ ] 内存使用和优化测试

## 部署考虑

### 前端优化
- 实现符号资源的CDN加速
- 优化符号图片和字体加载
- 实现组件的懒加载

### 服务端优化
- 部署手写识别服务
- 配置符号推荐API
- 实现符号数据缓存

### 移动端优化
- 优化触摸体验和响应
- 减少网络请求和数据传输
- 实现离线符号库

## 风险和依赖

### 技术风险
- 手写识别准确率
- 跨平台兼容性问题
- 性能优化挑战

### 业务风险
- 用户学习成本
- 符号标准化问题
- 设备适配复杂性

### 依赖关系
- 依赖数学符号数据库
- 依赖手写识别引擎
- 依赖推荐算法服务


