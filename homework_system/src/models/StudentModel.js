/**
 * 学生模型类
 * 用于存储和更新学生的特征维度，为推荐系统提供数据支持
 */

class StudentModel {
  /**
   * 构造函数
   * @param {String} studentId - 学生ID
   * @param {Object} initialData - 初始数据
   */
  constructor(studentId, initialData = {}) {
    this.studentId = studentId;
    this.lastUpdateTime = new Date();
    this.updateCount = 0;
    
    // 初始化特征维度
    this.features = {
      // 基本信息维度
      basic: {
        grade: initialData.grade || 0, // 年级
        subject: initialData.subject || '', // 学科偏好
        learningStyle: initialData.learningStyle || 'visual', // 学习风格：visual, auditory, kinesthetic
        devicePreference: initialData.devicePreference || 'pc', // 设备偏好：pc, tablet, mobile, robot
        averageSessionDuration: initialData.averageSessionDuration || 0, // 平均学习时长(分钟)
      },
      
      // 知识掌握度维度 (按知识点分类)
      knowledgeMastery: initialData.knowledgeMastery || {},
      
      // 学习行为维度
      behavior: {
        completionRate: initialData.completionRate || 0, // 作业完成率
        accuracyRate: initialData.accuracyRate || 0, // 正确率
        averageAttempts: initialData.averageAttempts || 1, // 平均尝试次数
        responseTime: initialData.responseTime || 0, // 平均响应时间(秒)
        engagementLevel: initialData.engagementLevel || 0, // 参与度(0-100)
        persistenceScore: initialData.persistenceScore || 0, // 坚持度(0-100)
        helpSeekingFrequency: initialData.helpSeekingFrequency || 0, // 寻求帮助频率
        errorPatterns: initialData.errorPatterns || [], // 错误模式列表
      },
      
      // 社交维度
      social: {
        collaborationFrequency: initialData.collaborationFrequency || 0, // 协作频率
        peerComparisonPercentile: initialData.peerComparisonPercentile || 50, // 同伴比较百分位
        teacherInteractions: initialData.teacherInteractions || 0, // 师生互动次数
        socialInfluence: initialData.socialInfluence || 0, // 社交影响力(0-100)
      },
      
      // 情感维度
      emotional: {
        frustrationLevel: initialData.frustrationLevel || 0, // 挫折水平(0-100)
        confidenceLevel: initialData.confidenceLevel || 50, // 信心水平(0-100)
        motivationLevel: initialData.motivationLevel || 50, // 动机水平(0-100)
        satisfactionLevel: initialData.satisfactionLevel || 50, // 满意度(0-100)
      },
      
      // 认知维度
      cognitive: {
        attentionSpan: initialData.attentionSpan || 0, // 注意力持续时间(分钟)
        cognitiveLoad: initialData.cognitiveLoad || 50, // 认知负荷(0-100)
        memoryRetention: initialData.memoryRetention || 0, // 记忆保持率(0-100)
        processingSpeed: initialData.processingSpeed || 0, // 处理速度(相对值)
      },
      
      // 元认知维度
      metacognitive: {
        selfRegulation: initialData.selfRegulation || 0, // 自我调节能力(0-100)
        selfAssessmentAccuracy: initialData.selfAssessmentAccuracy || 0, // 自我评估准确度(0-100)
        planningSkill: initialData.planningSkill || 0, // 规划能力(0-100)
        reflectionFrequency: initialData.reflectionFrequency || 0, // 反思频率
      },
      
      // 学习偏好维度
      preferences: {
        contentTypes: initialData.contentTypes || [], // 内容类型偏好
        difficultyPreference: initialData.difficultyPreference || 'medium', // 难度偏好
        feedbackPreference: initialData.feedbackPreference || 'immediate', // 反馈偏好
        symbolUsagePatterns: initialData.symbolUsagePatterns || {}, // 符号使用模式
        formulaPreferences: initialData.formulaPreferences || [], // 公式偏好
      },
      
      // 时间维度
      temporal: {
        studyTimePreference: initialData.studyTimePreference || [], // 学习时间偏好
        seasonalPerformance: initialData.seasonalPerformance || {}, // 季节性表现
        progressionRate: initialData.progressionRate || 0, // 进步速率
        consistencyScore: initialData.consistencyScore || 0, // 一致性得分(0-100)
      },
      
      // 环境维度
      environmental: {
        noiseToleranceLevel: initialData.noiseToleranceLevel || 'medium', // 噪音容忍度
        lightingPreference: initialData.lightingPreference || 'bright', // 光照偏好
        temperaturePreference: initialData.temperaturePreference || 'moderate', // 温度偏好
        spacePreference: initialData.spacePreference || 'open', // 空间偏好
      },
      
      // 技术维度
      technological: {
        deviceProficiency: initialData.deviceProficiency || {}, // 设备熟练度
        toolUsageFrequency: initialData.toolUsageFrequency || {}, // 工具使用频率
        digitalLiteracyScore: initialData.digitalLiteracyScore || 0, // 数字素养得分(0-100)
        adaptabilityToNewTools: initialData.adaptabilityToNewTools || 0, // 新工具适应性(0-100)
      },
      
      // 历史维度
      historical: {
        previousPerformance: initialData.previousPerformance || {}, // 历史表现
        improvementAreas: initialData.improvementAreas || [], // 改进领域
        strengthAreas: initialData.strengthAreas || [], // 优势领域
        consistentChallenges: initialData.consistentChallenges || [], // 持续挑战
      }
    };
    
    // 计算特征总数
    this.featureCount = this._countFeatures(this.features);
    
    // 初始化更新计划
    this._initUpdateSchedule();
  }
  
  /**
   * 计算特征总数
   * @param {Object} obj - 特征对象
   * @param {Number} count - 当前计数
   * @returns {Number} - 特征总数
   * @private
   */
  _countFeatures(obj, count = 0) {
    if (typeof obj !== 'object' || obj === null) {
      return 1;
    }
    
    let total = 0;
    for (const key in obj) {
      if (Array.isArray(obj[key])) {
        total += 1; // 数组算作一个特征
      } else if (typeof obj[key] === 'object' && obj[key] !== null) {
        total += this._countFeatures(obj[key]);
      } else {
        total += 1;
      }
    }
    
    return total;
  }
  
  /**
   * 初始化更新计划
   * @private
   */
  _initUpdateSchedule() {
    // 每5分钟更新一次学生模型
    this.updateInterval = setInterval(() => {
      this.update();
    }, 5 * 60 * 1000); // 5分钟
  }
  
  /**
   * 更新学生模型
   * @param {Object} newData - 新数据
   */
  update(newData = {}) {
    this.lastUpdateTime = new Date();
    this.updateCount++;
    
    // 合并新数据到特征中
    this._mergeData(this.features, newData);
    
    // 触发更新事件
    if (typeof this.onUpdate === 'function') {
      this.onUpdate(this);
    }
    
    return this;
  }
  
  /**
   * 递归合并数据
   * @param {Object} target - 目标对象
   * @param {Object} source - 源对象
   * @private
   */
  _mergeData(target, source) {
    for (const key in source) {
      if (source.hasOwnProperty(key)) {
        if (typeof source[key] === 'object' && source[key] !== null && !Array.isArray(source[key])) {
          if (!target[key]) target[key] = {};
          this._mergeData(target[key], source[key]);
        } else {
          target[key] = source[key];
        }
      }
    }
  }
  
  /**
   * 获取特定维度的特征
   * @param {String} dimension - 维度名称
   * @returns {Object} - 维度特征
   */
  getDimensionFeatures(dimension) {
    return this.features[dimension] || null;
  }
  
  /**
   * 获取特定知识点的掌握度
   * @param {String} knowledgePoint - 知识点ID
   * @returns {Number} - 掌握度(0-100)
   */
  getKnowledgeMastery(knowledgePoint) {
    return (this.features.knowledgeMastery[knowledgePoint] || 0);
  }
  
  /**
   * 更新知识点掌握度
   * @param {String} knowledgePoint - 知识点ID
   * @param {Number} masteryLevel - 掌握度(0-100)
   */
  updateKnowledgeMastery(knowledgePoint, masteryLevel) {
    if (!this.features.knowledgeMastery) {
      this.features.knowledgeMastery = {};
    }
    
    const currentLevel = this.features.knowledgeMastery[knowledgePoint] || 0;
    // 使用加权平均平滑更新
    const weight = 0.7; // 新数据权重
    const newLevel = currentLevel * (1 - weight) + masteryLevel * weight;
    
    this.features.knowledgeMastery[knowledgePoint] = Math.min(100, Math.max(0, newLevel));
    
    return this;
  }
  
  /**
   * 记录学习行为
   * @param {String} behaviorType - 行为类型
   * @param {Object} data - 行为数据
   */
  recordBehavior(behaviorType, data) {
    switch (behaviorType) {
      case 'assignment_completion':
        this._updateCompletionRate(data);
        break;
      case 'question_attempt':
        this._updateAccuracyAndAttempts(data);
        break;
      case 'help_request':
        this._updateHelpSeeking(data);
        break;
      case 'error_made':
        this._updateErrorPatterns(data);
        break;
      case 'engagement_action':
        this._updateEngagement(data);
        break;
      default:
        console.warn(`Unknown behavior type: ${behaviorType}`);
    }
    
    return this;
  }
  
  /**
   * 更新完成率
   * @param {Object} data - 完成数据
   * @private
   */
  _updateCompletionRate(data) {
    const { completed, total } = data;
    const currentRate = this.features.behavior.completionRate;
    
    // 计算新的完成率
    const newCompletionCount = (currentRate * total) + (completed ? 1 : 0);
    const newTotal = total + 1;
    this.features.behavior.completionRate = newCompletionCount / newTotal;
  }
  
  /**
   * 更新准确率和尝试次数
   * @param {Object} data - 尝试数据
   * @private
   */
  _updateAccuracyAndAttempts(data) {
    const { correct, attempts } = data;
    
    // 更新准确率
    const currentAccuracy = this.features.behavior.accuracyRate;
    const currentAttempts = this.features.behavior.averageAttempts;
    const weight = 0.3; // 新数据权重
    
    this.features.behavior.accuracyRate = currentAccuracy * (1 - weight) + (correct ? 1 : 0) * weight;
    this.features.behavior.averageAttempts = currentAttempts * (1 - weight) + attempts * weight;
  }
  
  /**
   * 更新寻求帮助频率
   * @param {Object} data - 帮助数据
   * @private
   */
  _updateHelpSeeking(data) {
    const { frequency } = data;
    const currentFrequency = this.features.behavior.helpSeekingFrequency;
    const weight = 0.2; // 新数据权重
    
    this.features.behavior.helpSeekingFrequency = currentFrequency * (1 - weight) + frequency * weight;
  }
  
  /**
   * 更新错误模式
   * @param {Object} data - 错误数据
   * @private
   */
  _updateErrorPatterns(data) {
    const { errorType, context } = data;
    
    // 检查是否已存在该错误模式
    const existingPatternIndex = this.features.behavior.errorPatterns.findIndex(
      pattern => pattern.type === errorType
    );
    
    if (existingPatternIndex >= 0) {
      // 更新已有错误模式
      const pattern = this.features.behavior.errorPatterns[existingPatternIndex];
      pattern.frequency = (pattern.frequency || 0) + 1;
      pattern.contexts = [...(pattern.contexts || []), context].slice(-5); // 保留最近5个上下文
    } else {
      // 添加新错误模式
      this.features.behavior.errorPatterns.push({
        type: errorType,
        frequency: 1,
        contexts: [context],
        firstObserved: new Date()
      });
    }
    
    // 限制错误模式数量，保留频率最高的
    if (this.features.behavior.errorPatterns.length > 10) {
      this.features.behavior.errorPatterns.sort((a, b) => b.frequency - a.frequency);
      this.features.behavior.errorPatterns = this.features.behavior.errorPatterns.slice(0, 10);
    }
  }
  
  /**
   * 更新参与度
   * @param {Object} data - 参与数据
   * @private
   */
  _updateEngagement(data) {
    const { engagementScore, duration } = data;
    const currentEngagement = this.features.behavior.engagementLevel;
    const weight = 0.25; // 新数据权重
    
    this.features.behavior.engagementLevel = currentEngagement * (1 - weight) + engagementScore * weight;
    
    // 更新平均学习时长
    const currentDuration = this.features.basic.averageSessionDuration;
    this.features.basic.averageSessionDuration = currentDuration * (1 - weight) + duration * weight;
  }
  
  /**
   * 获取学生模型摘要
   * @returns {Object} - 模型摘要
   */
  getSummary() {
    return {
      studentId: this.studentId,
      lastUpdateTime: this.lastUpdateTime,
      updateCount: this.updateCount,
      featureCount: this.featureCount,
      knowledgePoints: Object.keys(this.features.knowledgeMastery).length,
      averageKnowledgeMastery: this._calculateAverageKnowledgeMastery(),
      topStrengths: this._getTopStrengths(3),
      topWeaknesses: this._getTopWeaknesses(3),
      learningStyle: this.features.basic.learningStyle,
      devicePreference: this.features.basic.devicePreference,
      engagementLevel: this.features.behavior.engagementLevel,
      accuracyRate: this.features.behavior.accuracyRate,
      completionRate: this.features.behavior.completionRate
    };
  }
  
  /**
   * 计算平均知识掌握度
   * @returns {Number} - 平均掌握度
   * @private
   */
  _calculateAverageKnowledgeMastery() {
    const masteryValues = Object.values(this.features.knowledgeMastery);
    if (masteryValues.length === 0) return 0;
    
    const sum = masteryValues.reduce((acc, val) => acc + val, 0);
    return sum / masteryValues.length;
  }
  
  /**
   * 获取最强的知识点
   * @param {Number} count - 返回数量
   * @returns {Array} - 最强知识点列表
   * @private
   */
  _getTopStrengths(count = 3) {
    const knowledgePoints = Object.entries(this.features.knowledgeMastery)
      .map(([id, mastery]) => ({ id, mastery }))
      .sort((a, b) => b.mastery - a.mastery);
    
    return knowledgePoints.slice(0, count);
  }
  
  /**
   * 获取最弱的知识点
   * @param {Number} count - 返回数量
   * @returns {Array} - 最弱知识点列表
   * @private
   */
  _getTopWeaknesses(count = 3) {
    const knowledgePoints = Object.entries(this.features.knowledgeMastery)
      .filter(([_, mastery]) => mastery > 0) // 只考虑已学习过的知识点
      .map(([id, mastery]) => ({ id, mastery }))
      .sort((a, b) => a.mastery - b.mastery);
    
    return knowledgePoints.slice(0, count);
  }
  
  /**
   * 清理资源
   */
  dispose() {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
  }
}

export default StudentModel; 