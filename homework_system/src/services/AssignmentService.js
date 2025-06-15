/**
 * 作业服务类
 * 实现作业管理的核心功能，包括作业的接收、跟踪、提交和评分
 */

class AssignmentService {
  /**
   * 构造函数
   * @param {Object} options - 配置选项
   */
  constructor(options = {}) {
    this.options = {
      apiEndpoint: '/api/assignments',
      autoSaveInterval: 30000, // 30秒自动保存一次
      ...options
    };
    
    // 当前作业列表
    this.assignments = [];
    
    // 当前活动的作业
    this.activeAssignment = null;
    
    // 作业进度缓存
    this.progressCache = new Map();
    
    // 自动保存定时器
    this.autoSaveTimer = null;
  }
  
  /**
   * 初始化服务
   * @param {String} studentId - 学生ID
   * @returns {Promise<void>}
   */
  async initialize(studentId) {
    this.studentId = studentId;
    
    try {
      // 加载学生的作业列表
      await this.loadAssignments();
      
      // 设置自动保存
      this._setupAutoSave();
      
      return true;
    } catch (error) {
      console.error('Failed to initialize assignment service:', error);
      return false;
    }
  }
  
  /**
   * 加载学生的作业列表
   * @returns {Promise<Array>} - 作业列表
   */
  async loadAssignments() {
    try {
      // 实际应用中，这里应该是API调用
      // 现在使用模拟数据进行演示
      this.assignments = await this._mockApiCall(`${this.options.apiEndpoint}/list`, {
        studentId: this.studentId
      });
      
      return this.assignments;
    } catch (error) {
      console.error('Failed to load assignments:', error);
      throw error;
    }
  }
  
  /**
   * 获取作业列表
   * @param {Object} filters - 过滤条件
   * @returns {Array} - 过滤后的作业列表
   */
  getAssignments(filters = {}) {
    let filteredAssignments = [...this.assignments];
    
    // 应用过滤器
    if (filters.status) {
      filteredAssignments = filteredAssignments.filter(
        assignment => assignment.status === filters.status
      );
    }
    
    if (filters.subject) {
      filteredAssignments = filteredAssignments.filter(
        assignment => assignment.subject === filters.subject
      );
    }
    
    if (filters.dueDate) {
      const now = new Date();
      const dueDateFilter = filters.dueDate;
      
      if (dueDateFilter === 'today') {
        filteredAssignments = filteredAssignments.filter(assignment => {
          const dueDate = new Date(assignment.dueDate);
          return dueDate.toDateString() === now.toDateString();
        });
      } else if (dueDateFilter === 'thisWeek') {
        const weekStart = new Date(now);
        weekStart.setDate(now.getDate() - now.getDay());
        const weekEnd = new Date(weekStart);
        weekEnd.setDate(weekStart.getDate() + 6);
        
        filteredAssignments = filteredAssignments.filter(assignment => {
          const dueDate = new Date(assignment.dueDate);
          return dueDate >= weekStart && dueDate <= weekEnd;
        });
      } else if (dueDateFilter === 'overdue') {
        filteredAssignments = filteredAssignments.filter(assignment => {
          const dueDate = new Date(assignment.dueDate);
          return dueDate < now && assignment.status !== 'completed';
        });
      }
    }
    
    // 排序
    if (filters.sortBy) {
      filteredAssignments.sort((a, b) => {
        if (filters.sortBy === 'dueDate') {
          return new Date(a.dueDate) - new Date(b.dueDate);
        } else if (filters.sortBy === 'priority') {
          return b.priority - a.priority;
        } else if (filters.sortBy === 'subject') {
          return a.subject.localeCompare(b.subject);
        }
        return 0;
      });
      
      if (filters.sortDirection === 'desc') {
        filteredAssignments.reverse();
      }
    }
    
    return filteredAssignments;
  }
  
  /**
   * 获取作业详情
   * @param {String} assignmentId - 作业ID
   * @returns {Promise<Object>} - 作业详情
   */
  async getAssignmentDetails(assignmentId) {
    try {
      // 检查是否已有详情
      const existingAssignment = this.assignments.find(a => a.id === assignmentId);
      if (existingAssignment && existingAssignment.details) {
        return existingAssignment.details;
      }
      
      // 获取作业详情
      const details = await this._mockApiCall(`${this.options.apiEndpoint}/${assignmentId}/details`, {
        studentId: this.studentId
      });
      
      // 更新缓存
      const assignmentIndex = this.assignments.findIndex(a => a.id === assignmentId);
      if (assignmentIndex >= 0) {
        this.assignments[assignmentIndex].details = details;
      }
      
      return details;
    } catch (error) {
      console.error(`Failed to get assignment details for ${assignmentId}:`, error);
      throw error;
    }
  }
  
  /**
   * 开始作业
   * @param {String} assignmentId - 作业ID
   * @returns {Promise<Object>} - 作业内容
   */
  async startAssignment(assignmentId) {
    try {
      // 获取作业详情
      const details = await this.getAssignmentDetails(assignmentId);
      
      // 获取作业进度
      const progress = await this.getAssignmentProgress(assignmentId);
      
      // 设置当前活动的作业
      this.activeAssignment = {
        id: assignmentId,
        details,
        progress,
        startTime: new Date()
      };
      
      // 更新作业状态
      await this.updateAssignmentStatus(assignmentId, 'in_progress');
      
      return this.activeAssignment;
    } catch (error) {
      console.error(`Failed to start assignment ${assignmentId}:`, error);
      throw error;
    }
  }
  
  /**
   * 获取作业进度
   * @param {String} assignmentId - 作业ID
   * @returns {Promise<Object>} - 作业进度
   */
  async getAssignmentProgress(assignmentId) {
    try {
      // 检查缓存
      if (this.progressCache.has(assignmentId)) {
        return this.progressCache.get(assignmentId);
      }
      
      // 获取作业进度
      const progress = await this._mockApiCall(`${this.options.apiEndpoint}/${assignmentId}/progress`, {
        studentId: this.studentId
      });
      
      // 更新缓存
      this.progressCache.set(assignmentId, progress);
      
      return progress;
    } catch (error) {
      console.error(`Failed to get assignment progress for ${assignmentId}:`, error);
      throw error;
    }
  }
  
  /**
   * 更新作业进度
   * @param {String} assignmentId - 作业ID
   * @param {Object} progressData - 进度数据
   * @returns {Promise<Object>} - 更新后的进度
   */
  async updateAssignmentProgress(assignmentId, progressData) {
    try {
      // 获取当前进度
      const currentProgress = await this.getAssignmentProgress(assignmentId);
      
      // 合并进度数据
      const newProgress = {
        ...currentProgress,
        ...progressData,
        lastUpdated: new Date().toISOString()
      };
      
      // 更新缓存
      this.progressCache.set(assignmentId, newProgress);
      
      // 如果是当前活动的作业，更新活动作业的进度
      if (this.activeAssignment && this.activeAssignment.id === assignmentId) {
        this.activeAssignment.progress = newProgress;
      }
      
      // 实际应用中，这里应该是API调用
      // 现在使用模拟数据进行演示
      await this._mockApiCall(`${this.options.apiEndpoint}/${assignmentId}/progress`, {
        studentId: this.studentId,
        progress: newProgress
      }, 'PUT');
      
      return newProgress;
    } catch (error) {
      console.error(`Failed to update assignment progress for ${assignmentId}:`, error);
      throw error;
    }
  }
  
  /**
   * 保存作业答案
   * @param {String} assignmentId - 作业ID
   * @param {String} questionId - 问题ID
   * @param {Object} answer - 答案数据
   * @returns {Promise<Object>} - 保存结果
   */
  async saveAnswer(assignmentId, questionId, answer) {
    try {
      // 获取当前进度
      const currentProgress = await this.getAssignmentProgress(assignmentId);
      
      // 更新答案
      if (!currentProgress.answers) {
        currentProgress.answers = {};
      }
      
      currentProgress.answers[questionId] = {
        ...currentProgress.answers[questionId],
        ...answer,
        lastUpdated: new Date().toISOString()
      };
      
      // 计算完成度
      const details = await this.getAssignmentDetails(assignmentId);
      const totalQuestions = details.questions.length;
      const answeredQuestions = Object.keys(currentProgress.answers).length;
      currentProgress.completionRate = answeredQuestions / totalQuestions;
      
      // 更新进度
      await this.updateAssignmentProgress(assignmentId, {
        answers: currentProgress.answers,
        completionRate: currentProgress.completionRate
      });
      
      return currentProgress.answers[questionId];
    } catch (error) {
      console.error(`Failed to save answer for ${assignmentId}, question ${questionId}:`, error);
      throw error;
    }
  }
  
  /**
   * 获取问题答案
   * @param {String} assignmentId - 作业ID
   * @param {String} questionId - 问题ID
   * @returns {Promise<Object>} - 答案数据
   */
  async getAnswer(assignmentId, questionId) {
    try {
      // 获取当前进度
      const currentProgress = await this.getAssignmentProgress(assignmentId);
      
      // 返回答案
      return currentProgress.answers && currentProgress.answers[questionId] 
        ? currentProgress.answers[questionId] 
        : null;
    } catch (error) {
      console.error(`Failed to get answer for ${assignmentId}, question ${questionId}:`, error);
      throw error;
    }
  }
  
  /**
   * 提交作业
   * @param {String} assignmentId - 作业ID
   * @returns {Promise<Object>} - 提交结果
   */
  async submitAssignment(assignmentId) {
    try {
      // 获取当前进度
      const currentProgress = await this.getAssignmentProgress(assignmentId);
      
      // 检查完成度
      if (currentProgress.completionRate < 1) {
        const confirmSubmit = confirm('你还有未完成的题目，确定要提交吗？');
        if (!confirmSubmit) {
          return { submitted: false, message: '提交已取消' };
        }
      }
      
      // 提交作业
      const result = await this._mockApiCall(`${this.options.apiEndpoint}/${assignmentId}/submit`, {
        studentId: this.studentId,
        progress: currentProgress
      }, 'POST');
      
      // 更新作业状态
      await this.updateAssignmentStatus(assignmentId, 'submitted');
      
      // 清除活动作业
      if (this.activeAssignment && this.activeAssignment.id === assignmentId) {
        this.activeAssignment = null;
      }
      
      return result;
    } catch (error) {
      console.error(`Failed to submit assignment ${assignmentId}:`, error);
      throw error;
    }
  }
  
  /**
   * 获取作业评分
   * @param {String} assignmentId - 作业ID
   * @returns {Promise<Object>} - 评分结果
   */
  async getAssignmentGrade(assignmentId) {
    try {
      // 获取评分结果
      const grade = await this._mockApiCall(`${this.options.apiEndpoint}/${assignmentId}/grade`, {
        studentId: this.studentId
      });
      
      // 更新缓存
      const assignmentIndex = this.assignments.findIndex(a => a.id === assignmentId);
      if (assignmentIndex >= 0) {
        this.assignments[assignmentIndex].grade = grade;
      }
      
      return grade;
    } catch (error) {
      console.error(`Failed to get grade for assignment ${assignmentId}:`, error);
      throw error;
    }
  }
  
  /**
   * 更新作业状态
   * @param {String} assignmentId - 作业ID
   * @param {String} status - 新状态
   * @returns {Promise<Object>} - 更新结果
   */
  async updateAssignmentStatus(assignmentId, status) {
    try {
      // 更新状态
      const result = await this._mockApiCall(`${this.options.apiEndpoint}/${assignmentId}/status`, {
        studentId: this.studentId,
        status
      }, 'PUT');
      
      // 更新缓存
      const assignmentIndex = this.assignments.findIndex(a => a.id === assignmentId);
      if (assignmentIndex >= 0) {
        this.assignments[assignmentIndex].status = status;
      }
      
      return result;
    } catch (error) {
      console.error(`Failed to update status for assignment ${assignmentId}:`, error);
      throw error;
    }
  }
  
  /**
   * 设置自动保存
   * @private
   */
  _setupAutoSave() {
    // 清除现有的定时器
    if (this.autoSaveTimer) {
      clearInterval(this.autoSaveTimer);
    }
    
    // 设置新的定时器
    this.autoSaveTimer = setInterval(() => {
      this._autoSave();
    }, this.options.autoSaveInterval);
  }
  
  /**
   * 自动保存当前作业进度
   * @private
   */
  async _autoSave() {
    if (!this.activeAssignment) return;
    
    try {
      // 保存当前进度
      await this._mockApiCall(`${this.options.apiEndpoint}/${this.activeAssignment.id}/progress`, {
        studentId: this.studentId,
        progress: this.activeAssignment.progress
      }, 'PUT');
      
      console.log(`Auto-saved progress for assignment ${this.activeAssignment.id}`);
    } catch (error) {
      console.error('Auto-save failed:', error);
    }
  }
  
  /**
   * 模拟API调用
   * @param {String} endpoint - API端点
   * @param {Object} params - 请求参数
   * @param {String} method - 请求方法
   * @returns {Promise<Object>} - 响应数据
   * @private
   */
  async _mockApiCall(endpoint, params, method = 'GET') {
    // 在实际应用中，这里应该是真实的API调用
    // 现在使用模拟数据进行演示
    
    // 模拟网络延迟
    await new Promise(resolve => setTimeout(resolve, 100 + Math.random() * 300));
    
    // 根据不同的端点返回不同的模拟数据
    if (endpoint === `${this.options.apiEndpoint}/list`) {
      return [
        {
          id: 'assign1',
          title: '代数基础练习',
          subject: '数学',
          description: '二次方程和函数练习',
          dueDate: '2023-12-15T23:59:59',
          assignedDate: '2023-12-01T10:00:00',
          status: 'assigned',
          priority: 2,
          completionRate: 0
        },
        {
          id: 'assign2',
          title: '三角函数应用',
          subject: '数学',
          description: '三角函数在物理中的应用',
          dueDate: '2023-12-10T23:59:59',
          assignedDate: '2023-12-02T10:00:00',
          status: 'in_progress',
          priority: 3,
          completionRate: 0.5
        },
        {
          id: 'assign3',
          title: '概率统计基础',
          subject: '数学',
          description: '概率计算和统计分析',
          dueDate: '2023-12-05T23:59:59',
          assignedDate: '2023-11-28T10:00:00',
          status: 'submitted',
          priority: 1,
          completionRate: 1
        },
        {
          id: 'assign4',
          title: '微积分入门',
          subject: '数学',
          description: '导数和积分的基本概念',
          dueDate: '2023-12-20T23:59:59',
          assignedDate: '2023-12-05T10:00:00',
          status: 'assigned',
          priority: 2,
          completionRate: 0
        }
      ];
    } else if (endpoint.includes('/details')) {
      const assignmentId = endpoint.split('/')[2];
      
      if (assignmentId === 'assign1') {
        return {
          id: 'assign1',
          title: '代数基础练习',
          subject: '数学',
          description: '二次方程和函数练习',
          instructions: '完成以下关于二次方程和函数的练习题。请在每道题下方的答题区域填写你的答案和解题步骤。',
          dueDate: '2023-12-15T23:59:59',
          assignedDate: '2023-12-01T10:00:00',
          estimatedTime: 60, // 分钟
          totalPoints: 100,
          questions: [
            {
              id: 'q1',
              type: 'equation',
              content: '解方程：x² - 5x + 6 = 0',
              points: 20,
              knowledgePoints: ['二次方程', '因式分解']
            },
            {
              id: 'q2',
              type: 'function',
              content: '求函数 f(x) = x² - 4x + 3 的最小值',
              points: 25,
              knowledgePoints: ['二次函数', '求导', '函数最值']
            },
            {
              id: 'q3',
              type: 'application',
              content: '一个长方形的周长是20米，求长方形的面积最大值',
              points: 30,
              knowledgePoints: ['二次函数', '最值问题', '几何应用']
            },
            {
              id: 'q4',
              type: 'proof',
              content: '证明：对于任意实数a和b，(a + b)² ≤ 2(a² + b²)',
              points: 25,
              knowledgePoints: ['不等式', '代数证明']
            }
          ],
          resources: [
            {
              id: 'res1',
              type: 'pdf',
              title: '二次方程复习资料',
              url: '/resources/quadratic_equations.pdf'
            },
            {
              id: 'res2',
              type: 'video',
              title: '二次函数图像分析',
              url: '/resources/quadratic_functions_video.mp4'
            }
          ]
        };
      } else if (assignmentId === 'assign2') {
        return {
          id: 'assign2',
          title: '三角函数应用',
          subject: '数学',
          description: '三角函数在物理中的应用',
          instructions: '完成以下关于三角函数在物理中应用的练习题。请在每道题下方的答题区域填写你的答案和解题步骤。',
          dueDate: '2023-12-10T23:59:59',
          assignedDate: '2023-12-02T10:00:00',
          estimatedTime: 90, // 分钟
          totalPoints: 100,
          questions: [
            {
              id: 'q1',
              type: 'calculation',
              content: '一个物体在简谐运动中，其位移方程为 x = 5sin(2πt)，其中 x 的单位是米，t 的单位是秒。求物体的周期和最大速度。',
              points: 25,
              knowledgePoints: ['简谐运动', '三角函数', '导数']
            },
            {
              id: 'q2',
              type: 'calculation',
              content: '一个波的方程为 y = 3sin(4πx - 2πt)，其中 y 和 x 的单位是米，t 的单位是秒。求波的波长、周期和传播速度。',
              points: 25,
              knowledgePoints: ['波动方程', '三角函数', '物理应用']
            },
            {
              id: 'q3',
              type: 'application',
              content: '一个摆长为1米的单摆，在小角度摆动时，求其周期。（取重力加速度 g = 9.8 m/s²）',
              points: 25,
              knowledgePoints: ['单摆', '三角函数', '物理应用']
            },
            {
              id: 'q4',
              type: 'proof',
              content: '证明：在交流电路中，如果电压为 v = V₀sin(ωt)，电流为 i = I₀sin(ωt - φ)，则平均功率为 P = (V₀I₀/2)cosφ',
              points: 25,
              knowledgePoints: ['交流电', '三角函数', '物理应用']
            }
          ],
          resources: [
            {
              id: 'res1',
              type: 'pdf',
              title: '三角函数在物理中的应用',
              url: '/resources/trig_physics_applications.pdf'
            },
            {
              id: 'res2',
              type: 'simulation',
              title: '简谐运动模拟器',
              url: '/resources/harmonic_motion_simulator.html'
            }
          ]
        };
      } else {
        return {
          id: assignmentId,
          title: '默认作业',
          subject: '数学',
          description: '默认作业描述',
          instructions: '完成以下练习题。',
          questions: [
            {
              id: 'q1',
              type: 'text',
              content: '示例问题',
              points: 10
            }
          ]
        };
      }
    } else if (endpoint.includes('/progress')) {
      const assignmentId = endpoint.split('/')[2];
      
      if (method === 'PUT') {
        // 保存进度
        console.log('Saving progress:', params);
        return { success: true, message: 'Progress saved' };
      } else {
        // 获取进度
        if (assignmentId === 'assign1') {
          return {
            completionRate: 0,
            lastUpdated: '2023-12-01T10:30:00',
            answers: {}
          };
        } else if (assignmentId === 'assign2') {
          return {
            completionRate: 0.5,
            lastUpdated: '2023-12-03T15:45:00',
            answers: {
              q1: {
                content: '周期 T = 1秒，最大速度 vmax = 10π m/s',
                steps: [
                  'x = 5sin(2πt) 是简谐运动方程',
                  '周期 T = 2π/ω = 2π/(2π) = 1秒',
                  '速度 v = dx/dt = 5·2π·cos(2πt)',
                  '最大速度 vmax = 5·2π = 10π m/s'
                ],
                lastUpdated: '2023-12-03T14:20:00'
              },
              q2: {
                content: '波长 λ = 0.5米，周期 T = 1秒，传播速度 v = 0.5 m/s',
                steps: [
                  'y = 3sin(4πx - 2πt) 是波动方程',
                  '波长 λ = 2π/(4π) = 0.5米',
                  '周期 T = 2π/(2π) = 1秒',
                  '传播速度 v = λ/T = 0.5/1 = 0.5 m/s'
                ],
                lastUpdated: '2023-12-03T15:45:00'
              }
            }
          };
        } else {
          return {
            completionRate: 0,
            lastUpdated: null,
            answers: {}
          };
        }
      }
    } else if (endpoint.includes('/submit')) {
      // 提交作业
      return {
        submitted: true,
        submissionTime: new Date().toISOString(),
        message: '作业已成功提交，等待评分'
      };
    } else if (endpoint.includes('/grade')) {
      const assignmentId = endpoint.split('/')[2];
      
      if (assignmentId === 'assign3') {
        return {
          score: 85,
          totalPoints: 100,
          feedback: '整体表现良好，但在概率计算部分有一些错误。',
          gradedAt: '2023-12-06T14:30:00',
          questionFeedback: {
            q1: {
              score: 18,
              maxPoints: 20,
              feedback: '计算正确，但缺少解释'
            },
            q2: {
              score: 25,
              maxPoints: 25,
              feedback: '完美解答'
            },
            q3: {
              score: 22,
              maxPoints: 30,
              feedback: '概率计算有误，但思路正确'
            },
            q4: {
              score: 20,
              maxPoints: 25,
              feedback: '证明基本正确，但缺少一些关键步骤'
            }
          }
        };
      } else {
        return {
          score: null,
          message: '作业尚未评分'
        };
      }
    } else if (endpoint.includes('/status')) {
      // 更新状态
      return {
        success: true,
        message: `作业状态已更新为 ${params.status}`
      };
    }
    
    return { error: 'Invalid endpoint' };
  }
  
  /**
   * 清理资源
   */
  dispose() {
    if (this.autoSaveTimer) {
      clearInterval(this.autoSaveTimer);
      this.autoSaveTimer = null;
    }
  }
}

export default AssignmentService; 