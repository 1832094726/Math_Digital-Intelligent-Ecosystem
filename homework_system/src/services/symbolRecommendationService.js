/**
 * 符号推荐服务
 * 为每个题目提供智能符号推荐
 * 集成后端AI推荐API
 */

import axios from 'axios'

// API基础配置
const API_BASE_URL = 'http://localhost:8081/api'

// 创建axios实例
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器 - 添加认证token
apiClient.interceptors.request.use(
  config => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  error => {
    return Promise.reject(error)
  }
)

// 模拟的题目符号推荐数据库（作为后备数据）
const questionSymbolDatabase = {
  // 一元二次方程题目
  'hw1_q1': {
    symbols: [
      { id: 'sym1', symbol: 'x', description: '未知数x', category: 'variable', relevance: 0.95 },
      { id: 'sym2', symbol: '²', description: '平方', category: 'operator', relevance: 0.90 },
      { id: 'sym3', symbol: '=', description: '等号', category: 'operator', relevance: 0.85 },
      { id: 'sym4', symbol: '±', description: '正负号', category: 'operator', relevance: 0.80 },
      { id: 'sym5', symbol: '√', description: '根号', category: 'operator', relevance: 0.75 }
    ]
  },
  
  // 梯形面积题目
  'hw1_q2': {
    symbols: [
      { id: 'sym6', symbol: 'S', description: '面积S', category: 'variable', relevance: 0.95 },
      { id: 'sym7', symbol: '=', description: '等号', category: 'operator', relevance: 0.90 },
      { id: 'sym8', symbol: '(', description: '左括号', category: 'bracket', relevance: 0.85 },
      { id: 'sym9', symbol: ')', description: '右括号', category: 'bracket', relevance: 0.85 },
      { id: 'sym10', symbol: '+', description: '加号', category: 'operator', relevance: 0.80 },
      { id: 'sym11', symbol: '×', description: '乘号', category: 'operator', relevance: 0.75 },
      { id: 'sym12', symbol: '÷', description: '除号', category: 'operator', relevance: 0.70 },
      { id: 'sym13', symbol: '2', description: '数字2', category: 'number', relevance: 0.65 }
    ]
  },
  
  // 圆的面积题目
  'hw1_q3': {
    symbols: [
      { id: 'sym14', symbol: 'π', description: '圆周率', category: 'constant', relevance: 0.95 },
      { id: 'sym15', symbol: 'r', description: '半径r', category: 'variable', relevance: 0.90 },
      { id: 'sym16', symbol: '²', description: '平方', category: 'operator', relevance: 0.85 },
      { id: 'sym17', symbol: '×', description: '乘号', category: 'operator', relevance: 0.80 },
      { id: 'sym18', symbol: '=', description: '等号', category: 'operator', relevance: 0.75 },
      { id: 'sym19', symbol: 'S', description: '面积S', category: 'variable', relevance: 0.70 }
    ]
  },
  
  // 分数运算题目
  'hw2_q1': {
    symbols: [
      { id: 'sym20', symbol: '/', description: '分数线', category: 'operator', relevance: 0.95 },
      { id: 'sym21', symbol: '+', description: '加号', category: 'operator', relevance: 0.90 },
      { id: 'sym22', symbol: '-', description: '减号', category: 'operator', relevance: 0.85 },
      { id: 'sym23', symbol: '=', description: '等号', category: 'operator', relevance: 0.80 },
      { id: 'sym24', symbol: '(', description: '左括号', category: 'bracket', relevance: 0.75 },
      { id: 'sym25', symbol: ')', description: '右括号', category: 'bracket', relevance: 0.75 }
    ]
  },
  
  // 百分比题目
  'hw2_q2': {
    symbols: [
      { id: 'sym26', symbol: '%', description: '百分号', category: 'operator', relevance: 0.95 },
      { id: 'sym27', symbol: '×', description: '乘号', category: 'operator', relevance: 0.90 },
      { id: 'sym28', symbol: '=', description: '等号', category: 'operator', relevance: 0.85 },
      { id: 'sym29', symbol: '+', description: '加号', category: 'operator', relevance: 0.80 },
      { id: 'sym30', symbol: '-', description: '减号', category: 'operator', relevance: 0.75 }
    ]
  },
  
  // 三角形角度题目
  'hw2_q3': {
    symbols: [
      { id: 'sym31', symbol: '∠', description: '角', category: 'geometry', relevance: 0.95 },
      { id: 'sym32', symbol: '△', description: '三角形', category: 'geometry', relevance: 0.90 },
      { id: 'sym33', symbol: '°', description: '度', category: 'unit', relevance: 0.85 },
      { id: 'sym34', symbol: '=', description: '等号', category: 'operator', relevance: 0.80 },
      { id: 'sym35', symbol: '+', description: '加号', category: 'operator', relevance: 0.75 },
      { id: 'sym36', symbol: '180', description: '180度', category: 'constant', relevance: 0.70 }
    ]
  }
};

// 基于题目内容的符号推荐算法
const contentBasedRecommendation = (questionText) => {
  const content = questionText.toLowerCase();
  const symbols = [];
  
  // 方程相关
  if (content.includes('方程') || content.includes('解') || content.includes('x')) {
    symbols.push(
      { id: 'cb1', symbol: 'x', description: '未知数x', category: 'variable', relevance: 0.90 },
      { id: 'cb2', symbol: 'y', description: '未知数y', category: 'variable', relevance: 0.85 },
      { id: 'cb3', symbol: '=', description: '等号', category: 'operator', relevance: 0.80 }
    );
  }
  
  // 面积相关
  if (content.includes('面积') || content.includes('周长')) {
    symbols.push(
      { id: 'cb4', symbol: 'S', description: '面积S', category: 'variable', relevance: 0.90 },
      { id: 'cb5', symbol: '²', description: '平方', category: 'operator', relevance: 0.85 },
      { id: 'cb6', symbol: '×', description: '乘号', category: 'operator', relevance: 0.80 }
    );
  }
  
  // 圆相关
  if (content.includes('圆') || content.includes('半径') || content.includes('直径')) {
    symbols.push(
      { id: 'cb7', symbol: 'π', description: '圆周率', category: 'constant', relevance: 0.95 },
      { id: 'cb8', symbol: 'r', description: '半径r', category: 'variable', relevance: 0.90 },
      { id: 'cb9', symbol: 'd', description: '直径d', category: 'variable', relevance: 0.85 }
    );
  }
  
  // 角度相关
  if (content.includes('角') || content.includes('三角形') || content.includes('度')) {
    symbols.push(
      { id: 'cb10', symbol: '∠', description: '角', category: 'geometry', relevance: 0.90 },
      { id: 'cb11', symbol: '°', description: '度', category: 'unit', relevance: 0.85 },
      { id: 'cb12', symbol: '△', description: '三角形', category: 'geometry', relevance: 0.80 }
    );
  }
  
  // 分数相关
  if (content.includes('分数') || content.includes('分子') || content.includes('分母')) {
    symbols.push(
      { id: 'cb13', symbol: '/', description: '分数线', category: 'operator', relevance: 0.90 },
      { id: 'cb14', symbol: '(', description: '左括号', category: 'bracket', relevance: 0.80 },
      { id: 'cb15', symbol: ')', description: '右括号', category: 'bracket', relevance: 0.80 }
    );
  }
  
  // 百分比相关
  if (content.includes('%') || content.includes('百分') || content.includes('折扣')) {
    symbols.push(
      { id: 'cb16', symbol: '%', description: '百分号', category: 'operator', relevance: 0.95 },
      { id: 'cb17', symbol: '×', description: '乘号', category: 'operator', relevance: 0.85 }
    );
  }
  
  return symbols;
};

/**
 * 获取题目的符号推荐
 * @param {Object} params - 参数对象
 * @param {number} params.user_id - 用户ID
 * @param {string} params.question_text - 题目文本
 * @param {string} params.question_id - 题目ID（可选）
 * @param {string} params.current_topic - 当前知识点（可选）
 * @param {string} params.difficulty_level - 难度级别（可选）
 * @returns {Promise} - 返回符号推荐结果
 */
export const getSymbolRecommendations = async (params) => {
  try {
    const { question_id, question_text, user_id } = params;

    // 首先尝试调用后端AI推荐API
    const response = await apiClient.post('/recommend/symbols', {
      context: question_text || '',
      question_id: question_id,
      limit: 8
    });

    if (response.data.success && response.data.recommendations) {
      // 转换API响应格式为前端期望的格式
      const symbols = response.data.recommendations.map(rec => ({
        id: rec.id,
        symbol: rec.symbol_text,
        description: rec.symbol_name,
        category: rec.category,
        relevance: rec.confidence,
        latex_code: rec.latex_code
      }));

      return {
        data: {
          symbols: symbols,
          total: symbols.length,
          question_id: question_id,
          timestamp: new Date().toISOString(),
          source: 'ai_api'
        }
      };
    }
  } catch (error) {
    console.warn('AI推荐API调用失败，使用本地推荐:', error.message);
  }

  // 如果API调用失败，使用本地推荐作为后备
  return getLocalSymbolRecommendations(params);
};

/**
 * 本地符号推荐（后备方案）
 * @param {Object} params - 参数对象
 * @returns {Promise} - 返回符号推荐结果
 */
const getLocalSymbolRecommendations = async (params) => {
  return new Promise((resolve) => {
    setTimeout(() => {
      const { question_id, question_text } = params;

      let symbols = [];

      // 首先尝试从数据库中获取预定义的推荐
      if (question_id && questionSymbolDatabase[question_id]) {
        symbols = questionSymbolDatabase[question_id].symbols;
      } else {
        // 如果没有预定义推荐，使用基于内容的推荐
        symbols = contentBasedRecommendation(question_text || '');
      }

      // 按相关性排序
      symbols.sort((a, b) => b.relevance - a.relevance);

      // 限制返回数量（最多8个）
      symbols = symbols.slice(0, 8);

      resolve({
        data: {
          symbols: symbols,
          total: symbols.length,
          question_id: question_id,
          timestamp: new Date().toISOString(),
          source: 'local'
        }
      });
    }, 300); // 模拟网络延迟
  });
};

/**
 * 批量获取多个题目的符号推荐
 * @param {Array} questions - 题目数组
 * @returns {Promise} - 返回批量推荐结果
 */
export const getBatchSymbolRecommendations = async (questions) => {
  const results = {};
  
  for (const question of questions) {
    const response = await getSymbolRecommendations({
      question_id: question.id,
      question_text: question.content,
      user_id: 1
    });
    
    results[question.id] = response.data.symbols;
  }
  
  return {
    data: {
      recommendations: results,
      timestamp: new Date().toISOString()
    }
  };
};

/**
 * 更新符号使用统计（用于改进推荐算法）
 * @param {Object} params - 参数对象
 * @param {number} params.user_id - 用户ID
 * @param {string} params.question_id - 题目ID
 * @param {string} params.symbol - 使用的符号
 * @returns {Promise} - 返回更新结果
 */
export const updateSymbolUsage = async (params) => {
  try {
    // 调用后端API记录符号使用
    const response = await apiClient.post('/recommend/symbols/usage', {
      symbol_text: params.symbol,
      question_id: params.question_id,
      context: params.context || ''
    });

    if (response.data.success) {
      return {
        data: {
          success: true,
          message: '符号使用统计已更新'
        }
      };
    }
  } catch (error) {
    console.warn('符号使用统计API调用失败:', error.message);
  }

  // 如果API调用失败，使用本地记录
  return new Promise((resolve) => {
    setTimeout(() => {
      console.log('符号使用统计已更新(本地):', params);

      resolve({
        data: {
          success: true,
          message: '统计已更新(本地)'
        }
      });
    }, 100);
  });
};
