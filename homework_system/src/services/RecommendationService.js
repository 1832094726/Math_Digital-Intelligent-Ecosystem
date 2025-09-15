/**
 * 推荐服务类
 * 实现基于协同过滤、知识图谱和深度学习的推荐功能
 */

import axios from 'axios';

class RecommendationService {
  /**
   * 构造函数
   * @param {Object} options - 配置选项
   */
  constructor(options = {}) {
    this.options = {
      // 协同过滤配置
      collaborativeFiltering: {
        enabled: true,
        similarityThreshold: 0.6,
        maxRecommendations: 10,
        weightFactor: 0.4
      },
      
      // 知识图谱配置
      knowledgeGraph: {
        enabled: true,
        maxDepth: 2,
        relationThreshold: 0.5,
        weightFactor: 0.3
      },
      
      // 深度学习配置
      deepLearning: {
        enabled: true,
        modelEndpoint: '/api/recommendation/dl-model',
        confidenceThreshold: 0.7,
        weightFactor: 0.3
      },
      
      // 缓存配置
      cache: {
        enabled: true,
        ttl: 30 * 60 * 1000, // 30分钟
        maxSize: 1000
      },
      
      ...options
    };
    
    // 初始化缓存
    this.cache = new Map();
    
    // 最近的推荐结果
    this.lastRecommendations = {
      symbols: [],
      formulas: [],
      knowledgePoints: [],
      exercises: []
    };
  }
  
  /**
   * 获取符号推荐
   * @param {Object} studentModel - 学生模型
   * @param {Object} context - 上下文信息
   * @returns {Promise<Array>} - 推荐的符号列表
   */
  async getSymbolRecommendations(studentModel, context) {
    const cacheKey = `symbols_${studentModel.studentId}_${JSON.stringify(context)}`;
    
    // 检查缓存
    if (this.options.cache.enabled && this.cache.has(cacheKey)) {
      const cachedData = this.cache.get(cacheKey);
      if (Date.now() - cachedData.timestamp < this.options.cache.ttl) {
        return cachedData.data;
      }
    }
    
    try {
      // 调用后端API获取符号推荐
      const response = await axios.post('/api/recommend/symbols', {
        user_id: studentModel.studentId,
        question_text: context.questionText || '',
        current_topic: context.currentTopic || '',
        difficulty_level: context.difficultyLevel || 'medium'
      });
      
      let symbols = [];
      if (response.data && response.data.symbols) {
        symbols = response.data.symbols.map(symbol => ({
          id: symbol.id || `sym${Math.random().toString(36).substr(2, 9)}`,
          name: symbol.symbol || symbol.name,
          description: symbol.description || '',
          category: symbol.category || 'general',
          score: symbol.relevance || 0.8,
          sources: ['api']
        }));
      }
      
      // 更新最近的推荐结果
      this.lastRecommendations.symbols = symbols;
      
      // 缓存结果
      if (this.options.cache.enabled) {
        this.cache.set(cacheKey, {
          timestamp: Date.now(),
          data: symbols
        });
        
        // 清理过大的缓存
        if (this.cache.size > this.options.cache.maxSize) {
          const oldestKey = [...this.cache.keys()][0];
          this.cache.delete(oldestKey);
        }
      }
      
      return symbols;
    } catch (error) {
      console.error('Error getting symbol recommendations:', error);
      // 如果API调用失败，回退到模拟数据
      return this._getMockSymbols();
    }
  }
  
  /**
   * 获取公式推荐
   * @param {Object} studentModel - 学生模型
   * @param {Object} context - 上下文信息
   * @returns {Promise<Array>} - 推荐的公式列表
   */
  async getFormulaRecommendations(studentModel, context) {
    const cacheKey = `formulas_${studentModel.studentId}_${JSON.stringify(context)}`;
    
    // 检查缓存
    if (this.options.cache.enabled && this.cache.has(cacheKey)) {
      const cachedData = this.cache.get(cacheKey);
      if (Date.now() - cachedData.timestamp < this.options.cache.ttl) {
        return cachedData.data;
      }
    }
    
    try {
      // 调用后端API获取公式推荐
      // 目前后端没有专门的公式推荐API，可以通过知识点API获取相关公式
      const response = await axios.post('/api/recommend/knowledge', {
        user_id: studentModel.studentId,
        question_text: context.questionText || '',
        current_topic: context.currentTopic || '',
        difficulty_level: context.difficultyLevel || 'medium'
      });
      
      let formulas = [];
      if (response.data && response.data.knowledge_points) {
        // 从知识点中提取公式
        formulas = response.data.knowledge_points.flatMap(kp => {
          const kpFormulas = [];
          
          // 从key_points中提取可能的公式
          if (kp.key_points) {
            kp.key_points.forEach((point, index) => {
              if (point.includes('=') || point.includes('∫') || point.includes('∑')) {
                kpFormulas.push({
                  id: `form_${kp.id}_${index}`,
                  name: `${kp.name}公式`,
                  latex: point,
                  description: kp.description || '',
                  score: 0.9 - (index * 0.05),
                  sources: ['knowledge_graph']
                });
              }
            });
          }
          
          return kpFormulas;
        });
      }
      
      // 如果没有找到公式，使用模拟数据
      if (formulas.length === 0) {
        formulas = this._getMockFormulas();
      }
      
      // 更新最近的推荐结果
      this.lastRecommendations.formulas = formulas;
      
      // 缓存结果
      if (this.options.cache.enabled) {
        this.cache.set(cacheKey, {
          timestamp: Date.now(),
          data: formulas
        });
      }
      
      return formulas;
    } catch (error) {
      console.error('Error getting formula recommendations:', error);
      // 如果API调用失败，回退到模拟数据
      return this._getMockFormulas();
    }
  }
  
  /**
   * 获取知识点推荐
   * @param {Object} studentModel - 学生模型
   * @param {Object} context - 上下文信息
   * @returns {Promise<Array>} - 推荐的知识点列表
   */
  async getKnowledgePointRecommendations(studentModel, context) {
    const cacheKey = `knowledge_${studentModel.studentId}_${JSON.stringify(context)}`;
    
    // 检查缓存
    if (this.options.cache.enabled && this.cache.has(cacheKey)) {
      const cachedData = this.cache.get(cacheKey);
      if (Date.now() - cachedData.timestamp < this.options.cache.ttl) {
        return cachedData.data;
      }
    }
    
    try {
      // 调用后端API获取知识点推荐
      const response = await axios.post('/api/recommend/knowledge', {
        user_id: studentModel.studentId,
        question_text: context.questionText || '',
        current_topic: context.currentTopic || '',
        difficulty_level: context.difficultyLevel || 'medium'
      });
      
      let knowledgePoints = [];
      if (response.data && response.data.knowledge_points) {
        knowledgePoints = response.data.knowledge_points.map(kp => ({
          id: kp.id,
          name: kp.name,
          description: kp.description || '',
          category: kp.category || 'general',
          difficulty: kp.difficulty || 0.5,
          prerequisites: kp.prerequisites || [],
          relatedConcepts: kp.related_concepts || [],
          score: 0.9,
          sources: ['knowledge_graph']
        }));
      }
      
      // 如果题目ID存在，也尝试通过知识点API获取
      if (context.questionId) {
        try {
          const questionResponse = await axios.get('/api/knowledge/question', {
            params: { questionId: context.questionId }
          });
          
          if (questionResponse.data && questionResponse.data.knowledge_points) {
            const questionKnowledgePoints = questionResponse.data.knowledge_points.map(kp => ({
              id: kp.id,
              name: kp.name,
              description: kp.description || '',
              category: kp.category || 'general',
              difficulty: kp.difficulty || 0.5,
              prerequisites: kp.prerequisites || [],
              relatedConcepts: kp.related_concepts || [],
              score: 0.95, // 题目直接相关的知识点给予更高的分数
              sources: ['question_specific']
            }));
            
            // 合并两个知识点列表，去重
            const allKnowledgePoints = [...questionKnowledgePoints];
            knowledgePoints.forEach(kp => {
              if (!allKnowledgePoints.some(existing => existing.id === kp.id)) {
                allKnowledgePoints.push(kp);
              }
            });
            
            knowledgePoints = allKnowledgePoints;
          }
        } catch (error) {
          console.error('Error getting question-specific knowledge points:', error);
        }
      }
      
      // 如果没有找到知识点，使用模拟数据
      if (knowledgePoints.length === 0) {
        knowledgePoints = this._getMockKnowledgePoints();
      }
      
      // 更新最近的推荐结果
      this.lastRecommendations.knowledgePoints = knowledgePoints;
      
      // 缓存结果
      if (this.options.cache.enabled) {
        this.cache.set(cacheKey, {
          timestamp: Date.now(),
          data: knowledgePoints
        });
      }
      
      return knowledgePoints;
    } catch (error) {
      console.error('Error getting knowledge point recommendations:', error);
      // 如果API调用失败，回退到模拟数据
      return this._getMockKnowledgePoints();
    }
  }
  
  /**
   * 获取练习题推荐
   * @param {Object} studentModel - 学生模型
   * @param {Object} context - 上下文信息
   * @returns {Promise<Array>} - 推荐的练习题列表
   */
  async getExerciseRecommendations(studentModel, context) {
    const cacheKey = `exercises_${studentModel.studentId}_${JSON.stringify(context)}`;
    
    // 检查缓存
    if (this.options.cache.enabled && this.cache.has(cacheKey)) {
      const cachedData = this.cache.get(cacheKey);
      if (Date.now() - cachedData.timestamp < this.options.cache.ttl) {
        return cachedData.data;
      }
    }
    
    try {
      // 首先获取知识点
      let knowledgePoints = [];
      
      // 如果上下文中有知识点，直接使用
      if (context.knowledgePoints && context.knowledgePoints.length > 0) {
        knowledgePoints = context.knowledgePoints.map(kp => kp.name || kp);
      } 
      // 否则，如果有题目ID，尝试获取题目相关知识点
      else if (context.questionId) {
        try {
          const questionResponse = await axios.get('/api/knowledge/question', {
            params: { questionId: context.questionId }
          });
          
          if (questionResponse.data && questionResponse.data.knowledge_points) {
            knowledgePoints = questionResponse.data.knowledge_points.map(kp => kp.name);
          }
        } catch (error) {
          console.error('Error getting question-specific knowledge points:', error);
        }
      }
      
      // 调用后端API获取练习题推荐
      const response = await axios.post('/api/recommend/exercises', {
        user_id: studentModel.studentId,
        question_text: context.questionText || '',
        knowledge_points: knowledgePoints,
        difficulty_level: context.difficultyLevel || 'medium'
      });
      
      let exercises = [];
      if (response.data && response.data.exercises) {
        exercises = response.data.exercises.map(ex => ({
          id: ex.id,
          title: ex.title || ex.content.substring(0, 30) + '...',
          content: ex.content,
          difficulty: ex.difficulty || 'medium',
          type: ex.type || 'practice',
          knowledgePoints: ex.knowledge_points || [],
          score: ex.relevance || 0.8,
          homeworkId: ex.homework_id,
          sources: ['api']
        }));
      }
      
      // 如果没有找到练习题，使用模拟数据
      if (exercises.length === 0) {
        exercises = this._getMockExercises();
      }
      
      // 更新最近的推荐结果
      this.lastRecommendations.exercises = exercises;
      
      // 缓存结果
      if (this.options.cache.enabled) {
        this.cache.set(cacheKey, {
          timestamp: Date.now(),
          data: exercises
        });
      }
      
      return exercises;
    } catch (error) {
      console.error('Error getting exercise recommendations:', error);
      // 如果API调用失败，回退到模拟数据
      return this._getMockExercises();
    }
  }
  
  /**
   * 获取模拟符号数据
   * @returns {Array} - 模拟的符号列表
   * @private
   */
  _getMockSymbols() {
    return [
      { id: 'sym1', name: '∫', description: '积分符号', score: 0.95, sources: ['mock'] },
      { id: 'sym2', name: '∑', description: '求和符号', score: 0.87, sources: ['mock'] },
      { id: 'sym3', name: '∂', description: '偏导数符号', score: 0.82, sources: ['mock'] },
      { id: 'sym4', name: 'dx', description: '微分符号', score: 0.85, sources: ['mock'] },
      { id: 'sym5', name: '=', description: '等号', score: 0.79, sources: ['mock'] }
    ];
  }
  
  /**
   * 获取模拟公式数据
   * @returns {Array} - 模拟的公式列表
   * @private
   */
  _getMockFormulas() {
    return [
      { id: 'form1', name: '二次方程', latex: 'ax^2 + bx + c = 0', score: 0.92, sources: ['mock'] },
      { id: 'form2', name: '勾股定理', latex: 'a^2 + b^2 = c^2', score: 0.85, sources: ['mock'] },
      { id: 'form3', name: '一元一次方程', latex: 'ax + b = 0', score: 0.78, sources: ['mock'] },
      { id: 'form4', name: '面积公式', latex: 'S = πr^2', score: 0.83, sources: ['mock'] },
      { id: 'form5', name: '三角函数', latex: '\\sin^2(x) + \\cos^2(x) = 1', score: 0.77, sources: ['mock'] }
    ];
  }
  
  /**
   * 获取模拟知识点数据
   * @returns {Array} - 模拟的知识点列表
   * @private
   */
  _getMockKnowledgePoints() {
    return [
      { 
        id: 'kp1', 
        name: '二次函数', 
        description: '二次函数的性质和应用', 
        category: '代数',
        difficulty: 0.6,
        prerequisites: ['一元一次方程', '函数基础'],
        relatedConcepts: ['抛物线', '顶点', '对称轴'],
        score: 0.93, 
        sources: ['mock'] 
      },
      { 
        id: 'kp2', 
        name: '三角函数', 
        description: '三角函数的定义和性质', 
        category: '三角学',
        difficulty: 0.7,
        prerequisites: ['角度', '直角三角形'],
        relatedConcepts: ['周期', '振幅', '相位'],
        score: 0.86, 
        sources: ['mock'] 
      },
      { 
        id: 'kp3', 
        name: '概率论', 
        description: '概率的基本概念和计算', 
        category: '统计',
        difficulty: 0.65,
        prerequisites: ['集合', '计数原理'],
        relatedConcepts: ['随机变量', '期望值', '方差'],
        score: 0.79, 
        sources: ['mock'] 
      }
    ];
  }
  
  /**
   * 获取模拟练习题数据
   * @returns {Array} - 模拟的练习题列表
   * @private
   */
  _getMockExercises() {
    return [
      { 
        id: 'ex1', 
        title: '二次方程应用题', 
        content: '解方程：x² - 5x + 6 = 0',
        difficulty: 'medium', 
        type: 'calculation',
        knowledgePoints: ['二次方程', '因式分解'],
        score: 0.94, 
        sources: ['mock'] 
      },
      { 
        id: 'ex2', 
        title: '三角函数计算题', 
        content: '计算：sin(30°) + cos(60°)',
        difficulty: 'easy', 
        type: 'calculation',
        knowledgePoints: ['三角函数', '特殊角'],
        score: 0.87, 
        sources: ['mock'] 
      },
      { 
        id: 'ex3', 
        title: '概率计算题', 
        content: '一个袋子中有3个红球和2个白球，随机抽取2个球，求抽到2个红球的概率。',
        difficulty: 'medium', 
        type: 'application',
        knowledgePoints: ['概率论', '组合计数'],
        score: 0.80, 
        sources: ['mock'] 
      }
    ];
  }
  
  /**
   * 获取智能符号补全建议
   * @param {Object} studentModel - 学生模型
   * @param {Object} context - 上下文信息
   * @returns {Promise<Array>} 补全建议列表
   */
  async getSymbolCompletions(studentModel, context) {
    const currentInput = context.currentInput || '';
    const cursorPosition = context.cursorPosition || 0;

    // 获取光标前的文本
    const textBeforeCursor = currentInput.substring(0, cursorPosition);
    const lastWord = this.extractLastIncompleteSymbol(textBeforeCursor);

    if (!lastWord || lastWord.length < 1) {
      return [];
    }

    try {
      // 调用后端API获取补全建议
      const response = await axios.post('/api/symbols/complete', {
        user_id: studentModel.studentId,
        partial_input: lastWord,
        context: textBeforeCursor,
        question_text: context.questionText || '',
        max_suggestions: 10
      });

      if (response.data && response.data.completions) {
        return response.data.completions.map(completion => ({
          id: completion.id || `comp${Math.random().toString(36).substr(2, 9)}`,
          symbol: completion.symbol,
          latex: completion.latex || completion.symbol,
          description: completion.description || '',
          score: completion.score || 0.5,
          insertText: completion.insertText || completion.symbol,
          replaceLength: lastWord.length
        }));
      }
    } catch (error) {
      console.error('Failed to get symbol completions:', error);
    }

    // 使用本地补全逻辑作为后备
    return this.getLocalSymbolCompletions(lastWord, studentModel, context);
  }

  /**
   * 提取最后一个不完整的符号
   * @param {String} text - 输入文本
   * @returns {String} 最后一个不完整的符号
   */
  extractLastIncompleteSymbol(text) {
    // 匹配LaTeX命令或符号名称
    const latexMatch = text.match(/\\([a-zA-Z]*)$/);
    if (latexMatch) {
      return latexMatch[0]; // 包含反斜杠
    }

    // 匹配普通符号名称
    const symbolMatch = text.match(/([a-zA-Z]+)$/);
    if (symbolMatch) {
      return symbolMatch[1];
    }

    return '';
  }

  /**
   * 本地符号补全逻辑
   * @param {String} partialInput - 部分输入
   * @param {Object} studentModel - 学生模型
   * @param {Object} context - 上下文信息
   * @returns {Array} 补全建议列表
   */
  getLocalSymbolCompletions(partialInput, studentModel, context) {
    const commonSymbols = [
      { symbol: 'α', latex: '\\alpha', description: '希腊字母alpha' },
      { symbol: 'β', latex: '\\beta', description: '希腊字母beta' },
      { symbol: 'γ', latex: '\\gamma', description: '希腊字母gamma' },
      { symbol: 'δ', latex: '\\delta', description: '希腊字母delta' },
      { symbol: 'π', latex: '\\pi', description: '圆周率' },
      { symbol: '∑', latex: '\\sum', description: '求和符号' },
      { symbol: '∫', latex: '\\int', description: '积分符号' },
      { symbol: '√', latex: '\\sqrt', description: '平方根' },
      { symbol: '∞', latex: '\\infty', description: '无穷大' },
      { symbol: '≤', latex: '\\leq', description: '小于等于' },
      { symbol: '≥', latex: '\\geq', description: '大于等于' },
      { symbol: '≠', latex: '\\neq', description: '不等于' },
      { symbol: '±', latex: '\\pm', description: '正负号' }
    ];

    const input = partialInput.toLowerCase().replace('\\', '');

    return commonSymbols
      .filter(sym => {
        const latexName = sym.latex.replace('\\', '').toLowerCase();
        const symbolName = sym.description.toLowerCase();
        return latexName.includes(input) || symbolName.includes(input);
      })
      .map((sym, index) => ({
        id: `local_comp_${index}`,
        symbol: sym.symbol,
        latex: sym.latex,
        description: sym.description,
        score: 0.8 - (index * 0.05),
        insertText: partialInput.startsWith('\\') ? sym.latex : sym.symbol,
        replaceLength: partialInput.length
      }))
      .slice(0, 8);
  }

  /**
   * 获取上下文感知的符号推荐
   * @param {Object} studentModel - 学生模型
   * @param {Object} context - 上下文信息
   * @returns {Promise<Array>} 推荐的符号列表
   */
  async getContextAwareSymbolRecommendations(studentModel, context) {
    const currentInput = context.currentInput || '';
    const questionText = context.questionText || '';

    // 分析当前输入的数学内容类型
    const mathContext = this.analyzeMathContext(currentInput, questionText);

    try {
      const response = await axios.post('/api/recommend/symbols/context', {
        user_id: studentModel.studentId,
        current_input: currentInput,
        question_text: questionText,
        math_context: mathContext,
        cursor_position: context.cursorPosition || 0,
        usage_patterns: studentModel.features?.preferences?.symbolUsagePatterns || {}
      });

      if (response.data && response.data.symbols) {
        return response.data.symbols.map(symbol => ({
          id: symbol.id || `ctx${Math.random().toString(36).substr(2, 9)}`,
          name: symbol.symbol || symbol.name,
          latex: symbol.latex || symbol.symbol,
          description: symbol.description || '',
          category: symbol.category || 'contextual',
          score: symbol.relevance || 0.7,
          contextReason: symbol.reason || '基于上下文推荐',
          sources: ['context_analysis']
        }));
      }
    } catch (error) {
      console.error('Failed to get context-aware recommendations:', error);
    }

    // 使用本地上下文分析作为后备
    return this.getLocalContextAwareRecommendations(mathContext, studentModel);
  }

  /**
   * 分析数学上下文
   * @param {String} currentInput - 当前输入
   * @param {String} questionText - 题目文本
   * @returns {Object} 数学上下文信息
   */
  analyzeMathContext(currentInput, questionText) {
    const context = {
      hasEquations: /=/.test(currentInput) || /方程/.test(questionText),
      hasGeometry: /三角形|圆|角度|面积|周长/.test(questionText),
      hasCalculus: /导数|积分|极限/.test(questionText) || /\\int|\\sum|\\lim/.test(currentInput),
      hasAlgebra: /代数|多项式|因式分解/.test(questionText),
      hasStatistics: /概率|统计|平均/.test(questionText),
      hasFractions: /\\//.test(currentInput) || /分数/.test(questionText),
      hasRoots: /\\sqrt/.test(currentInput) || /根号|平方根/.test(questionText),
      hasGreekLetters: /\\[a-zA-Z]+/.test(currentInput),
      complexity: this.calculateComplexity(currentInput, questionText)
    };

    return context;
  }

  /**
   * 计算数学复杂度
   * @param {String} currentInput - 当前输入
   * @param {String} questionText - 题目文本
   * @returns {String} 复杂度等级
   */
  calculateComplexity(currentInput, questionText) {
    let score = 0;

    // 基于符号复杂度
    if (/\\int|\\sum|\\prod/.test(currentInput)) score += 3;
    if (/\\frac|\\sqrt/.test(currentInput)) score += 2;
    if (/[α-ωΑ-Ω]/.test(currentInput)) score += 1;

    // 基于题目复杂度
    if (/微积分|导数|积分/.test(questionText)) score += 3;
    if (/三角函数|对数/.test(questionText)) score += 2;
    if (/方程|不等式/.test(questionText)) score += 1;

    if (score >= 5) return 'high';
    if (score >= 3) return 'medium';
    return 'low';
  }

  /**
   * 获取最近的推荐结果
   * @param {String} type - 推荐类型 (symbols, formulas, knowledgePoints, exercises)
   * @returns {Array} - 最近的推荐结果
   */
  getLastRecommendations(type) {
    return this.lastRecommendations[type] || [];
  }

  /**
   * 本地上下文感知推荐
   * @param {Object} mathContext - 数学上下文
   * @param {Object} studentModel - 学生模型
   * @returns {Array} 推荐的符号列表
   */
  getLocalContextAwareRecommendations(mathContext, studentModel) {
    const recommendations = [];

    // 基于几何上下文的推荐
    if (mathContext.hasGeometry) {
      recommendations.push(
        { symbol: '∠', latex: '\\angle', description: '角度符号', category: 'geometry', score: 0.9 },
        { symbol: '△', latex: '\\triangle', description: '三角形', category: 'geometry', score: 0.85 },
        { symbol: '⊥', latex: '\\perp', description: '垂直', category: 'geometry', score: 0.8 },
        { symbol: '∥', latex: '\\parallel', description: '平行', category: 'geometry', score: 0.8 },
        { symbol: '○', latex: '\\circ', description: '圆', category: 'geometry', score: 0.75 }
      );
    }

    // 基于微积分上下文的推荐
    if (mathContext.hasCalculus) {
      recommendations.push(
        { symbol: '∫', latex: '\\int', description: '积分符号', category: 'calculus', score: 0.95 },
        { symbol: '∑', latex: '\\sum', description: '求和符号', category: 'calculus', score: 0.9 },
        { symbol: '∏', latex: '\\prod', description: '连乘符号', category: 'calculus', score: 0.85 },
        { symbol: '∂', latex: '\\partial', description: '偏导数', category: 'calculus', score: 0.8 },
        { symbol: '∇', latex: '\\nabla', description: '梯度算子', category: 'calculus', score: 0.75 },
        { symbol: '∞', latex: '\\infty', description: '无穷大', category: 'calculus', score: 0.7 }
      );
    }

    // 基于代数上下文的推荐
    if (mathContext.hasAlgebra || mathContext.hasEquations) {
      recommendations.push(
        { symbol: '±', latex: '\\pm', description: '正负号', category: 'algebra', score: 0.85 },
        { symbol: '≠', latex: '\\neq', description: '不等于', category: 'algebra', score: 0.8 },
        { symbol: '≤', latex: '\\leq', description: '小于等于', category: 'algebra', score: 0.8 },
        { symbol: '≥', latex: '\\geq', description: '大于等于', category: 'algebra', score: 0.8 },
        { symbol: '≈', latex: '\\approx', description: '约等于', category: 'algebra', score: 0.75 }
      );
    }

    // 基于分数上下文的推荐
    if (mathContext.hasFractions) {
      recommendations.push(
        { symbol: '/', latex: '\\frac{}{}', description: '分数', category: 'fraction', score: 0.9 }
      );
    }

    // 基于根号上下文的推荐
    if (mathContext.hasRoots) {
      recommendations.push(
        { symbol: '√', latex: '\\sqrt{}', description: '平方根', category: 'root', score: 0.9 },
        { symbol: '∛', latex: '\\sqrt[3]{}', description: '立方根', category: 'root', score: 0.8 }
      );
    }

    // 基于希腊字母上下文的推荐
    if (mathContext.hasGreekLetters || mathContext.complexity !== 'low') {
      recommendations.push(
        { symbol: 'α', latex: '\\alpha', description: '希腊字母alpha', category: 'greek', score: 0.8 },
        { symbol: 'β', latex: '\\beta', description: '希腊字母beta', category: 'greek', score: 0.75 },
        { symbol: 'γ', latex: '\\gamma', description: '希腊字母gamma', category: 'greek', score: 0.75 },
        { symbol: 'δ', latex: '\\delta', description: '希腊字母delta', category: 'greek', score: 0.75 },
        { symbol: 'π', latex: '\\pi', description: '圆周率', category: 'greek', score: 0.85 },
        { symbol: 'θ', latex: '\\theta', description: '希腊字母theta', category: 'greek', score: 0.8 }
      );
    }

    // 根据学生使用历史调整推荐
    const usagePatterns = studentModel.features?.preferences?.symbolUsagePatterns || {};
    recommendations.forEach(rec => {
      const usageCount = usagePatterns[rec.symbol] || 0;
      if (usageCount > 0) {
        rec.score += Math.min(0.1, usageCount * 0.01); // 使用频率加成
        rec.contextReason = `基于使用历史推荐（使用${usageCount}次）`;
      } else {
        rec.contextReason = '基于上下文推荐';
      }
    });

    // 按分数排序并返回
    return recommendations
      .sort((a, b) => b.score - a.score)
      .slice(0, 10)
      .map((rec, index) => ({
        id: `local_ctx_${index}`,
        name: rec.symbol,
        latex: rec.latex,
        description: rec.description,
        category: rec.category,
        score: rec.score,
        contextReason: rec.contextReason,
        sources: ['local_context']
      }));
  }

  /**
   * 记录符号使用行为
   * @param {Object} studentModel - 学生模型
   * @param {String} symbol - 使用的符号
   * @param {Object} context - 使用上下文
   */
  recordSymbolUsage(studentModel, symbol, context) {
    if (!studentModel || !symbol) return;

    // 更新符号使用模式
    const currentPatterns = studentModel.features?.preferences?.symbolUsagePatterns || {};
    const updatedPatterns = {
      ...currentPatterns,
      [symbol]: (currentPatterns[symbol] || 0) + 1
    };

    // 记录使用时间
    const usageHistory = studentModel.features?.preferences?.symbolUsageHistory || [];
    usageHistory.push({
      symbol,
      timestamp: new Date().toISOString(),
      context: {
        questionText: context.questionText || '',
        currentTopic: context.currentTopic || '',
        mathContext: this.analyzeMathContext(context.currentInput || '', context.questionText || '')
      }
    });

    // 保持历史记录在合理范围内
    if (usageHistory.length > 1000) {
      usageHistory.splice(0, usageHistory.length - 1000);
    }

    // 更新学生模型
    studentModel.update({
      preferences: {
        symbolUsagePatterns: updatedPatterns,
        symbolUsageHistory: usageHistory
      }
    });
  }

  /**
   * 获取个性化符号推荐
   * @param {Object} studentModel - 学生模型
   * @param {Object} context - 上下文信息
   * @returns {Promise<Array>} 个性化推荐列表
   */
  async getPersonalizedSymbolRecommendations(studentModel, context) {
    // 获取基础推荐
    const [basicRecommendations, contextRecommendations] = await Promise.all([
      this.getSymbolRecommendations(studentModel, context),
      this.getContextAwareSymbolRecommendations(studentModel, context)
    ]);

    // 合并推荐结果
    const allRecommendations = [...basicRecommendations, ...contextRecommendations];

    // 去重
    const uniqueRecommendations = [];
    const seenSymbols = new Set();

    for (const rec of allRecommendations) {
      const key = rec.name || rec.symbol;
      if (!seenSymbols.has(key)) {
        seenSymbols.add(key);
        uniqueRecommendations.push(rec);
      }
    }

    // 基于学生模型进行个性化调整
    const personalizedRecommendations = this.personalizeRecommendations(
      uniqueRecommendations,
      studentModel,
      context
    );

    return personalizedRecommendations.slice(0, 12);
  }

  /**
   * 个性化推荐调整
   * @param {Array} recommendations - 原始推荐列表
   * @param {Object} studentModel - 学生模型
   * @param {Object} context - 上下文信息
   * @returns {Array} 调整后的推荐列表
   */
  personalizeRecommendations(recommendations, studentModel, context) {
    const usagePatterns = studentModel.features?.preferences?.symbolUsagePatterns || {};
    const difficultyLevel = studentModel.features?.academic?.currentLevel || 'medium';

    return recommendations.map(rec => {
      let adjustedScore = rec.score || 0.5;

      // 基于使用频率调整
      const usageCount = usagePatterns[rec.name || rec.symbol] || 0;
      if (usageCount > 0) {
        adjustedScore += Math.min(0.2, usageCount * 0.02);
      }

      // 基于难度等级调整
      if (difficultyLevel === 'beginner' && rec.category === 'calculus') {
        adjustedScore -= 0.3; // 降低高级符号的推荐权重
      } else if (difficultyLevel === 'advanced' && rec.category === 'basic') {
        adjustedScore -= 0.1; // 降低基础符号的推荐权重
      }

      // 基于最近使用时间调整
      const recentUsage = this.getRecentSymbolUsage(studentModel, rec.name || rec.symbol);
      if (recentUsage && recentUsage.withinHour) {
        adjustedScore += 0.15;
      } else if (recentUsage && recentUsage.withinDay) {
        adjustedScore += 0.05;
      }

      return {
        ...rec,
        score: Math.min(1.0, Math.max(0.0, adjustedScore)),
        personalizedReason: this.generatePersonalizationReason(rec, usageCount, difficultyLevel)
      };
    }).sort((a, b) => b.score - a.score);
  }

  /**
   * 获取最近符号使用情况
   * @param {Object} studentModel - 学生模型
   * @param {String} symbol - 符号
   * @returns {Object} 最近使用情况
   */
  getRecentSymbolUsage(studentModel, symbol) {
    const usageHistory = studentModel.features?.preferences?.symbolUsageHistory || [];
    const recentUsages = usageHistory.filter(usage => usage.symbol === symbol);

    if (recentUsages.length === 0) return null;

    const lastUsage = recentUsages[recentUsages.length - 1];
    const lastUsageTime = new Date(lastUsage.timestamp);
    const now = new Date();
    const timeDiff = now - lastUsageTime;

    return {
      lastUsed: lastUsageTime,
      withinHour: timeDiff < 60 * 60 * 1000,
      withinDay: timeDiff < 24 * 60 * 60 * 1000,
      totalUsages: recentUsages.length
    };
  }

  /**
   * 生成个性化推荐原因
   * @param {Object} recommendation - 推荐项
   * @param {Number} usageCount - 使用次数
   * @param {String} difficultyLevel - 难度等级
   * @returns {String} 推荐原因
   */
  generatePersonalizationReason(recommendation, usageCount, difficultyLevel) {
    const reasons = [];

    if (usageCount > 10) {
      reasons.push('您经常使用此符号');
    } else if (usageCount > 0) {
      reasons.push('您之前使用过此符号');
    }

    if (recommendation.contextReason) {
      reasons.push(recommendation.contextReason);
    }

    if (difficultyLevel === 'beginner' && recommendation.category === 'basic') {
      reasons.push('适合当前学习水平');
    } else if (difficultyLevel === 'advanced' && recommendation.category === 'calculus') {
      reasons.push('符合高级学习需求');
    }

    return reasons.length > 0 ? reasons.join('，') : '系统推荐';
  }

  /**
   * 获取自适应推荐
   * @param {Object} studentModel - 学生模型
   * @param {Object} context - 上下文信息
   * @returns {Promise<Array>} 自适应推荐列表
   */
  async getAdaptiveRecommendations(studentModel, context) {
    try {
      const response = await axios.post('/api/symbols/recommend/adaptive', {
        user_id: studentModel.studentId,
        question_text: context.questionText || '',
        current_input: context.currentInput || '',
        current_topic: context.currentTopic || '',
        difficulty_level: context.difficultyLevel || 'medium',
        cursor_position: context.cursorPosition || 0,
        usage_history: studentModel.features?.preferences?.symbolUsagePatterns || {}
      });

      if (response.data && response.data.success) {
        const data = response.data.data;
        return {
          symbols: data.symbols || [],
          learningPattern: data.learning_pattern || {},
          learningSuggestions: data.learning_suggestions || [],
          adaptationApplied: data.adaptation_applied || false
        };
      }

      return {
        symbols: [],
        learningPattern: {},
        learningSuggestions: [],
        adaptationApplied: false
      };

    } catch (error) {
      console.error('Failed to get adaptive recommendations:', error);
      // 回退到普通推荐
      const fallbackSymbols = await this.getSymbolRecommendations(studentModel, context);
      return {
        symbols: fallbackSymbols,
        learningPattern: {},
        learningSuggestions: ['使用基础推荐模式'],
        adaptationApplied: false
      };
    }
  }

  /**
   * 获取用户学习分析
   * @param {Object} studentModel - 学生模型
   * @returns {Promise<Object>} 学习分析数据
   */
  async getUserLearningAnalytics(studentModel) {
    try {
      const response = await axios.get(`/api/symbols/analytics/${studentModel.studentId}`);

      if (response.data && response.data.success) {
        return response.data.data;
      }

      return null;

    } catch (error) {
      console.error('Failed to get learning analytics:', error);
      return null;
    }
  }

  /**
   * 获取学习洞察
   * @param {Object} studentModel - 学生模型
   * @returns {Promise<Object>} 学习洞察数据
   */
  async getLearningInsights(studentModel) {
    try {
      const response = await axios.get(`/api/symbols/learning-insights/${studentModel.studentId}`);

      if (response.data && response.data.success) {
        return response.data.data;
      }

      return null;

    } catch (error) {
      console.error('Failed to get learning insights:', error);
      return null;
    }
  }

  /**
   * 获取带解释的推荐
   * @param {Object} studentModel - 学生模型
   * @param {Object} context - 上下文信息
   * @returns {Promise<Object>} 带解释的推荐数据
   */
  async getExplainedRecommendations(studentModel, context) {
    try {
      const response = await axios.post('/api/symbols/recommend/explained', {
        user_id: studentModel.studentId,
        question_text: context.questionText || '',
        current_input: context.currentInput || '',
        current_topic: context.currentTopic || '',
        difficulty_level: context.difficultyLevel || 'medium',
        cursor_position: context.cursorPosition || 0,
        usage_history: studentModel.features?.preferences?.symbolUsagePatterns || {}
      });

      if (response.data && response.data.success) {
        return response.data.data;
      }

      return {
        symbols: [],
        context: {},
        user_statistics: {},
        total_count: 0
      };

    } catch (error) {
      console.error('Failed to get explained recommendations:', error);
      // 回退到普通推荐
      const fallbackSymbols = await this.getSymbolRecommendations(studentModel, context);
      return {
        symbols: fallbackSymbols.map(symbol => ({
          ...symbol,
          explanation: '基础推荐'
        })),
        context: {},
        user_statistics: {},
        total_count: fallbackSymbols.length
      };
    }
  }

  /**
   * 记录符号选择（增强版）
   * @param {Object} studentModel - 学生模型
   * @param {Object} symbol - 选择的符号
   * @param {Object} context - 上下文信息
   */
  async recordEnhancedSymbolUsage(studentModel, symbol, context) {
    try {
      // 调用后端记录API
      await axios.post('/api/symbols/usage', {
        user_id: studentModel.studentId,
        symbol_id: symbol.id,
        question_text: context.questionText || '',
        current_topic: context.currentTopic || '',
        current_input: context.currentInput || '',
        timestamp: new Date().toISOString()
      });

      // 同时更新本地学生模型
      this.recordSymbolUsage(studentModel, symbol.name || symbol.symbol, context);

    } catch (error) {
      console.error('Failed to record enhanced symbol usage:', error);
      // 至少更新本地模型
      this.recordSymbolUsage(studentModel, symbol.name || symbol.symbol, context);
    }
  }

  /**
   * 获取符号类别
   * @returns {Promise<Array>} 符号类别列表
   */
  async getSymbolCategories() {
    try {
      const response = await axios.get('/api/symbols/categories');

      if (response.data && response.data.success) {
        return response.data.data.categories || [];
      }

      return this._getDefaultCategories();

    } catch (error) {
      console.error('Failed to get symbol categories:', error);
      return this._getDefaultCategories();
    }
  }

  /**
   * 根据类别获取符号
   * @param {String} categoryId - 类别ID
   * @returns {Promise<Array>} 符号列表
   */
  async getSymbolsByCategory(categoryId) {
    try {
      const response = await axios.get(`/api/symbols/category/${categoryId}`);

      if (response.data && response.data.success) {
        return response.data.data.symbols || [];
      }

      return [];

    } catch (error) {
      console.error('Failed to get symbols by category:', error);
      return [];
    }
  }

  /**
   * 搜索符号
   * @param {String} query - 搜索查询
   * @returns {Promise<Array>} 搜索结果
   */
  async searchSymbols(query) {
    try {
      const response = await axios.post('/api/symbols/search', {
        query: query
      });

      if (response.data && response.data.success) {
        return response.data.data.symbols || [];
      }

      return [];

    } catch (error) {
      console.error('Failed to search symbols:', error);
      return [];
    }
  }

  /**
   * 获取默认类别
   * @returns {Array} 默认类别列表
   */
  _getDefaultCategories() {
    return [
      { id: 'basic', name: '基本运算', icon: 'icon-plus', description: '加减乘除等基本运算符号' },
      { id: 'relation', name: '关系符号', icon: 'icon-equals', description: '等于、不等于、大于小于等关系符号' },
      { id: 'greek', name: '希腊字母', icon: 'icon-alpha', description: 'α、β、π等希腊字母' },
      { id: 'calculus', name: '微积分', icon: 'icon-integral', description: '积分、求和、极限等微积分符号' },
      { id: 'geometry', name: '几何符号', icon: 'icon-triangle', description: '角度、三角形、圆等几何符号' },
      { id: 'algebra', name: '代数符号', icon: 'icon-function', description: '函数、集合、逻辑等代数符号' }
    ];
  }

  /**
   * 清理缓存
   */
  clearCache() {
    this.cache.clear();
  }
}

export default RecommendationService;