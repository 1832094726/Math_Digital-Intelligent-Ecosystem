# 符号推荐功能演示文档

## 功能概述

在答题界面（http://localhost:8080/homework）中集成了智能符号推荐功能，为学生提供上下文相关的数学符号推荐，提升答题效率和准确性。

## 功能特点

### 1. 智能触发
- **自动激活**：当学生点击答案输入框时，自动显示符号推荐面板
- **上下文感知**：根据题目内容智能推荐相关符号
- **实时响应**：快速加载推荐内容，提供流畅的用户体验

### 2. IEEE学术风格设计
- **专业外观**：采用IEEE会议论文的配色方案和字体
- **学术色彩**：深蓝色渐变头部，Times New Roman字体
- **简洁布局**：清晰的分类结构，易于浏览和选择

### 3. 多层次符号分类

#### 基础运算符号
```
+ - × ÷ = ≠ > < ≥ ≤ ² ³ √ π
```

#### 几何符号
```
∠ △ □ ○ ∥ ⊥ ∽ ≅
```

#### 智能推荐符号
- 根据题目内容动态推荐
- 带有星标标识的高优先级符号
- 基于用户历史使用频率的个性化推荐

## 使用流程

### 步骤1：进入答题界面
1. 访问 http://localhost:8080/homework
2. 选择一个作业
3. 展开任意题目

### 步骤2：激活符号推荐
1. 点击答案输入框
2. 符号推荐面板自动弹出
3. 显示加载状态，然后展示推荐符号

### 步骤3：选择和插入符号
1. 浏览不同类别的符号
2. 点击所需符号
3. 符号自动插入到答案中
4. 显示插入成功提示

### 步骤4：关闭面板
1. 点击面板右上角的关闭按钮
2. 或点击其他区域自动关闭

## 技术实现

### 前端组件结构
```vue
<template>
  <div class="answer-input-container">
    <el-input 
      @focus="onAnswerFocus(question)"
      class="answer-input"
    />
    
    <div v-if="showSymbolPanel" class="symbol-recommendation-panel">
      <div class="panel-header">
        <h4><i class="el-icon-magic-stick"></i> 推荐符号</h4>
        <el-button @click="closeSymbolPanel" class="close-btn" />
      </div>
      
      <div class="symbol-content">
        <!-- 基础运算 -->
        <div class="symbol-category">
          <h5>基础运算</h5>
          <div class="symbol-grid">
            <button @click="insertSymbol(symbol.symbol, question.id)">
              {{ symbol.symbol }}
            </button>
          </div>
        </div>
        
        <!-- 智能推荐 -->
        <div class="symbol-category">
          <h5>智能推荐</h5>
          <div class="symbol-grid">
            <button class="symbol-btn recommended">
              {{ symbol.symbol }}
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
```

### 核心方法

#### 1. 符号推荐触发
```javascript
async onAnswerFocus(question) {
  this.currentQuestion = question;
  this.showSymbolPanel = true;
  await this.loadRecommendedSymbols(question);
}
```

#### 2. 智能推荐算法
```javascript
async loadRecommendedSymbols(question) {
  try {
    const response = await this.$http.post('/api/recommend/symbols', {
      user_id: this.user?.id || 1,
      question_text: question.content,
      current_topic: question.knowledge_points?.[0] || '',
      difficulty_level: this.currentHomework?.difficulty || 'medium'
    });
    
    this.recommendedSymbols = response.data.symbols;
  } catch (error) {
    // 使用默认推荐符号
    this.recommendedSymbols = this.getDefaultRecommendedSymbols(question);
  }
}
```

#### 3. 符号插入
```javascript
insertSymbol(symbol, questionId) {
  const currentAnswer = this.answers[questionId] || '';
  this.answers[questionId] = currentAnswer + symbol;
  this.saveProgress();
  this.$message.success(`已插入符号: ${symbol}`);
}
```

### CSS样式特点

#### IEEE学术风格
```css
.symbol-recommendation-panel {
  font-family: 'Times New Roman', serif;
  border: 2px solid #409EFF;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.panel-header {
  background: linear-gradient(135deg, #2C3E50 0%, #34495E 100%);
  color: white;
}

.symbol-category h5 {
  font-family: 'Times New Roman', serif;
  color: #2C3E50;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
```

#### 交互效果
```css
.symbol-btn:hover {
  border-color: #409EFF;
  background: #ECF5FF;
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(64, 158, 255, 0.2);
}

.symbol-btn.recommended {
  background: linear-gradient(135deg, #67C23A 0%, #85CE61 100%);
  color: white;
}

.symbol-btn.recommended::before {
  content: '★';
  position: absolute;
  top: -2px;
  right: -2px;
}
```

## 响应式设计

### 桌面端
- 面板宽度自适应输入框
- 符号按钮40x40px
- 网格布局自动填充

### 移动端
```css
@media (max-width: 768px) {
  .symbol-recommendation-panel {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 90%;
    max-width: 400px;
  }
  
  .symbol-btn {
    width: 35px;
    height: 35px;
    font-size: 14px;
  }
}
```

## API集成

### 后端接口
```
POST /api/recommend/symbols
Content-Type: application/json

{
  "user_id": 1,
  "question_text": "解方程 x² - 5x + 6 = 0",
  "current_topic": "一元二次方程",
  "difficulty_level": "medium"
}
```

### 响应格式
```json
{
  "symbols": [
    {
      "id": 1,
      "symbol": "x",
      "description": "未知数x",
      "category": "algebra",
      "relevance": 0.95
    },
    {
      "id": 2,
      "symbol": "²",
      "description": "平方",
      "category": "power",
      "relevance": 0.90
    }
  ]
}
```

## 用户体验优化

### 1. 性能优化
- 防抖处理避免频繁API调用
- 本地缓存常用符号
- 懒加载推荐内容

### 2. 交互优化
- 悬停效果提供即时反馈
- 插入成功提示确认操作
- 键盘快捷键支持

### 3. 可访问性
- 符号描述提供工具提示
- 键盘导航支持
- 高对比度模式兼容

## 扩展功能

### 1. 公式推荐
- 常用数学公式模板
- 基于题目类型的公式建议
- 公式编辑器集成

### 2. 历史记录
- 用户常用符号统计
- 个性化推荐优化
- 使用习惯分析

### 3. 多语言支持
- 符号描述多语言
- 界面文本国际化
- 数学符号标准化

## 总结

符号推荐功能成功集成到答题界面中，提供了：
- 智能化的符号推荐
- IEEE学术风格的专业界面
- 流畅的用户交互体验
- 完整的响应式设计
- 可扩展的架构设计

该功能显著提升了学生在数学答题过程中的效率和准确性，为智能教育系统提供了重要的辅助工具。
