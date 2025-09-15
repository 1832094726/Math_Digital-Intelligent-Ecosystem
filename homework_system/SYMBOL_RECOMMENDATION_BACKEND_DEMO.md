# 符号推荐后端服务演示文档

## 功能概述

为每个示例题目提供智能符号推荐，确保所有题目都有相应的符号推荐，提升学生答题体验。

## 后端服务架构

### 1. 符号推荐服务 (symbolRecommendationService.js)

#### 核心功能
- **预定义推荐**：为特定题目ID提供精确的符号推荐
- **内容分析推荐**：基于题目文本内容智能推荐符号
- **使用统计跟踪**：记录符号使用情况，优化推荐算法

#### API接口

##### getSymbolRecommendations(params)
```javascript
// 参数
{
  user_id: 1,
  question_id: 'hw1_q1',
  question_text: '解方程 x² - 5x + 6 = 0',
  current_topic: '一元二次方程',
  difficulty_level: 'medium'
}

// 返回
{
  data: {
    symbols: [
      { id: 'sym1', symbol: 'x', description: '未知数x', category: 'variable', relevance: 0.95 },
      { id: 'sym2', symbol: '²', description: '平方', category: 'operator', relevance: 0.90 }
    ],
    total: 2,
    question_id: 'hw1_q1',
    timestamp: '2023-12-07T10:30:00.000Z'
  }
}
```

## 题目符号推荐数据库

### 作业1题目推荐

#### hw1_q1 - 一元二次方程
```
题目：解方程 x² - 5x + 6 = 0
推荐符号：
- x (未知数x) - 相关性: 95%
- ² (平方) - 相关性: 90%
- = (等号) - 相关性: 85%
- ± (正负号) - 相关性: 80%
- √ (根号) - 相关性: 75%
```

#### hw1_q2 - 梯形面积
```
题目：计算梯形面积，上底5cm，下底8cm，高4cm
推荐符号：
- S (面积S) - 相关性: 95%
- = (等号) - 相关性: 90%
- ( (左括号) - 相关性: 85%
- ) (右括号) - 相关性: 85%
- + (加号) - 相关性: 80%
- × (乘号) - 相关性: 75%
- ÷ (除号) - 相关性: 70%
- 2 (数字2) - 相关性: 65%
```

#### hw1_q3 - 圆的面积
```
题目：计算半径为3cm的圆的面积
推荐符号：
- π (圆周率) - 相关性: 95%
- r (半径r) - 相关性: 90%
- ² (平方) - 相关性: 85%
- × (乘号) - 相关性: 80%
- = (等号) - 相关性: 75%
- S (面积S) - 相关性: 70%
```

### 作业2题目推荐

#### hw2_q1 - 分数运算
```
题目：计算 2/3 + 1/4
推荐符号：
- / (分数线) - 相关性: 95%
- + (加号) - 相关性: 90%
- - (减号) - 相关性: 85%
- = (等号) - 相关性: 80%
- ( (左括号) - 相关性: 75%
- ) (右括号) - 相关性: 75%
```

#### hw2_q2 - 百分比计算
```
题目：某商品原价100元，打8折后的价格是多少？
推荐符号：
- % (百分号) - 相关性: 95%
- × (乘号) - 相关性: 90%
- = (等号) - 相关性: 85%
- + (加号) - 相关性: 80%
- - (减号) - 相关性: 75%
```

#### hw2_q3 - 三角形角度
```
题目：三角形内角和为180°，已知两角分别为60°和80°，求第三角
推荐符号：
- ∠ (角) - 相关性: 95%
- △ (三角形) - 相关性: 90%
- ° (度) - 相关性: 85%
- = (等号) - 相关性: 80%
- + (加号) - 相关性: 75%
- 180 (180度) - 相关性: 70%
```

## 智能推荐算法

### 1. 预定义推荐优先
```javascript
if (question_id && questionSymbolDatabase[question_id]) {
  symbols = questionSymbolDatabase[question_id].symbols;
}
```

### 2. 内容分析推荐
```javascript
const contentBasedRecommendation = (questionText) => {
  const content = questionText.toLowerCase();
  
  // 方程相关
  if (content.includes('方程') || content.includes('解')) {
    return ['x', 'y', '=', '²', '√'];
  }
  
  // 面积相关
  if (content.includes('面积') || content.includes('周长')) {
    return ['S', '²', '×', 'π', 'r'];
  }
  
  // 更多规则...
}
```

### 3. 相关性排序
```javascript
symbols.sort((a, b) => b.relevance - a.relevance);
symbols = symbols.slice(0, 8); // 限制最多8个
```

## 前端集成

### 1. 服务调用
```javascript
// 加载推荐符号
async loadRecommendedSymbols(question) {
  const symbolService = await import('../services/symbolRecommendationService');
  
  const response = await symbolService.getSymbolRecommendations({
    user_id: this.user?.id || 1,
    question_id: question.id,
    question_text: question.content
  });
  
  this.recommendedSymbols = response.data.symbols;
}
```

### 2. 使用统计
```javascript
// 插入符号时记录使用
async insertSymbol(symbol, questionId) {
  // 插入符号到答案
  this.answers[questionId] += symbol;
  
  // 更新使用统计
  await symbolService.updateSymbolUsage({
    user_id: this.user?.id,
    question_id: questionId,
    symbol: symbol
  });
}
```

## 界面展示效果

### 符号分类显示
```
💡 智能推荐
[x] [²] [=] [±] [√]

➕ 基础运算  
[+] [-] [×] [÷] [=] [≠] [>] [<]

📐 几何符号
[∠] [△] [□] [○] [∥] [⊥] [∽] [≅]
```

### 推荐优先级
- **高相关性符号**：显示在智能推荐区域顶部
- **中等相关性符号**：显示在智能推荐区域底部
- **通用符号**：显示在基础运算和几何符号区域

## 性能优化

### 1. 缓存机制
- 相同题目的推荐结果缓存5分钟
- 用户会话期间保持推荐结果

### 2. 批量加载
```javascript
// 批量获取作业中所有题目的推荐
const batchResponse = await getBatchSymbolRecommendations(questions);
```

### 3. 异步加载
- 符号推荐异步加载，不阻塞界面显示
- 加载失败时降级到本地推荐算法

## 数据统计与分析

### 使用频率统计
```javascript
{
  user_id: 1,
  question_id: 'hw1_q1',
  symbol: 'x',
  usage_count: 15,
  last_used: '2023-12-07T10:30:00.000Z'
}
```

### 推荐效果评估
- **点击率**：推荐符号的使用比例
- **准确性**：用户实际使用的符号与推荐的匹配度
- **覆盖率**：有推荐符号的题目比例

## 扩展功能

### 1. 个性化推荐
- 基于用户历史使用习惯调整推荐
- 学习用户偏好的符号表示方式

### 2. 上下文感知
- 考虑前后题目的关联性
- 基于当前答案内容动态调整推荐

### 3. 多语言支持
- 支持不同地区的数学符号习惯
- 提供符号的多种表示方式

## 总结

符号推荐后端服务为每个示例题目提供了精确的符号推荐，通过预定义数据库和智能算法相结合的方式，确保所有题目都有相关的符号推荐，显著提升了学生的答题体验和效率。
