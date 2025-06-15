/**
 * 作业服务
 * 处理作业相关的API请求
 */

// 注释掉未使用的导入和变量
// import axios from 'axios';

// API基础URL，实际应用中应从环境变量获取
// const API_BASE_URL = '/api';

/**
 * 获取作业列表
 * @returns {Promise} - 返回作业列表
 */
export const fetchHomeworkList = () => {
  // 实际应用中应调用真实API
  // 这里返回模拟数据
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        data: [
          {
            id: 'hw1',
            title: '一元二次方程练习',
            description: '本作业包含一元二次方程的基本概念和解法',
            deadline: '2023-07-01T23:59:59',
            status: 'in_progress',
            difficulty: 2,
            lastActivity: '2023-06-14T15:30:00',
            problems: [
              { id: 'p1', title: '一元二次方程的概念' },
              { id: 'p2', title: '因式分解法' },
              { id: 'p3', title: '配方法' },
              { id: 'p4', title: '公式法' },
              { id: 'p5', title: '根与系数的关系' }
            ],
            savedAnswers: {
              'p1': 'x^2 + 2x + 1 = 0',
              'p2': '(x+1)^2 = 0'
            }
          },
          {
            id: 'hw2',
            title: '三角函数基础',
            description: '三角函数的定义、性质和应用',
            deadline: '2023-07-05T23:59:59',
            status: 'not_started',
            difficulty: 3,
            lastActivity: null,
            problems: [
              { id: 'p6', title: '三角函数的定义' },
              { id: 'p7', title: '三角函数的基本性质' },
              { id: 'p8', title: '三角恒等式' },
              { id: 'p9', title: '三角函数的图像' }
            ],
            savedAnswers: {}
          },
          {
            id: 'hw3',
            title: '数列基础',
            description: '数列的基本概念和等差、等比数列',
            deadline: '2023-06-20T23:59:59',
            status: 'submitted',
            difficulty: 2,
            lastActivity: '2023-06-10T10:15:00',
            problems: [
              { id: 'p10', title: '数列的基本概念' },
              { id: 'p11', title: '等差数列' },
              { id: 'p12', title: '等比数列' }
            ],
            savedAnswers: {
              'p10': '数列是按照一定顺序排列的数的序列',
              'p11': 'a_n = a_1 + (n-1)d',
              'p12': 'a_n = a_1 * q^(n-1)'
            }
          },
          {
            id: 'hw4',
            title: '概率统计入门',
            description: '概率与统计的基本概念和计算方法',
            deadline: '2023-06-25T23:59:59',
            status: 'graded',
            difficulty: 4,
            lastActivity: '2023-06-05T14:20:00',
            problems: [
              { id: 'p13', title: '概率的基本概念' },
              { id: 'p14', title: '古典概型' },
              { id: 'p15', title: '条件概率' },
              { id: 'p16', title: '统计量的计算' }
            ],
            savedAnswers: {
              'p13': '概率是对随机事件发生可能性的度量',
              'p14': 'P(A) = m/n',
              'p15': 'P(A|B) = P(AB)/P(B)',
              'p16': '平均值 = (x_1 + x_2 + ... + x_n) / n'
            },
            feedback: {
              overallScore: 85,
              totalScore: 100,
              accuracyScore: 0.85,
              completionScore: 1.0,
              methodScore: 0.8,
              overallComment: '整体表现良好，对概率基本概念掌握扎实，但在条件概率的应用上还需加强。',
              problems: [
                {
                  id: 'p13',
                  title: '概率的基本概念',
                  statement: '请解释概率的基本概念及其应用场景。',
                  isCorrect: true,
                  score: 25,
                  totalScore: 25,
                  studentAnswer: '概率是对随机事件发生可能性的度量',
                  comments: [
                    {
                      type: 'success',
                      text: '概念理解正确，表述简洁明了。'
                    }
                  ]
                },
                {
                  id: 'p14',
                  title: '古典概型',
                  statement: '请给出古典概型的计算公式并举例说明。',
                  isCorrect: true,
                  score: 25,
                  totalScore: 25,
                  studentAnswer: 'P(A) = m/n',
                  comments: [
                    {
                      type: 'success',
                      text: '公式正确。'
                    },
                    {
                      type: 'info',
                      text: '可以补充一个具体例子会更好。'
                    }
                  ]
                },
                {
                  id: 'p15',
                  title: '条件概率',
                  statement: '请给出条件概率的定义和计算公式。',
                  isCorrect: false,
                  score: 15,
                  totalScore: 25,
                  studentAnswer: 'P(A|B) = P(AB)/P(B)',
                  correctAnswer: 'P(A|B) = P(A∩B)/P(B)，其中P(B)>0。条件概率P(A|B)表示在事件B已经发生的条件下，事件A发生的概率。',
                  comments: [
                    {
                      type: 'warning',
                      text: '公式正确但表述不完整，缺少条件P(B)>0和概念解释。'
                    },
                    {
                      type: 'error',
                      text: '没有说明条件概率的实际含义。'
                    }
                  ]
                },
                {
                  id: 'p16',
                  title: '统计量的计算',
                  statement: '请给出样本平均值的计算公式。',
                  isCorrect: true,
                  score: 20,
                  totalScore: 25,
                  studentAnswer: '平均值 = (x_1 + x_2 + ... + x_n) / n',
                  comments: [
                    {
                      type: 'success',
                      text: '计算公式正确。'
                    },
                    {
                      type: 'info',
                      text: '可以使用求和符号表示会更简洁：x̄ = (1/n)∑x_i'
                    }
                  ]
                }
              ],
              suggestions: [
                '加强条件概率相关概念的理解和应用',
                '练习使用数学符号规范表达公式',
                '在回答问题时注意概念的完整性',
                '多做一些统计量计算的实际应用题'
              ]
            }
          }
        ]
      });
    }, 500);
  });
};

/**
 * 获取作业详情
 * @param {string} homeworkId - 作业ID
 * @returns {Promise} - 返回作业详情
 */
export const fetchHomeworkDetail = (homeworkId) => {
  // 实际应用中应调用真实API
  // 这里返回模拟数据
  return new Promise((resolve) => {
    setTimeout(() => {
      if (homeworkId === 'hw1') {
        resolve({
          data: {
            id: 'hw1',
            title: '一元二次方程练习',
            description: '本作业包含一元二次方程的基本概念和解法，通过本次作业，你将掌握一元二次方程的各种解法和应用。',
            deadline: '2023-07-01T23:59:59',
            status: 'in_progress',
            difficulty: 2,
            lastActivity: '2023-06-14T15:30:00',
            problems: [
              {
                id: 'p1',
                title: '一元二次方程的概念',
                content: '请写出一元二次方程的一般形式，并解释其中各项的含义。',
                points: 20,
                type: 'algebra'
              },
              {
                id: 'p2',
                title: '因式分解法',
                content: '使用因式分解法解方程：x² - 4 = 0',
                points: 20,
                type: 'algebra'
              },
              {
                id: 'p3',
                title: '配方法',
                content: '使用配方法解方程：x² + 6x + 5 = 0',
                points: 20,
                type: 'algebra'
              },
              {
                id: 'p4',
                title: '公式法',
                content: '使用公式法解方程：2x² - 7x + 3 = 0',
                points: 20,
                type: 'algebra'
              },
              {
                id: 'p5',
                title: '根与系数的关系',
                content: '已知一元二次方程x² + px + q = 0的两根为3和-2，求p和q的值。',
                points: 20,
                type: 'algebra'
              }
            ],
            savedAnswers: {
              'p1': 'ax^2 + bx + c = 0，其中a、b、c是实数，且a ≠ 0',
              'p2': '(x+2)(x-2) = 0，所以x = 2或x = -2'
            }
          }
        });
      } else {
        // 返回默认数据
        resolve({
          data: {
            id: homeworkId,
            title: '作业标题',
            description: '作业描述',
            deadline: '2023-07-01T23:59:59',
            status: 'not_started',
            difficulty: 2,
            problems: []
          }
        });
      }
    }, 500);
  });
};

/**
 * 提交作业答案
 * @param {Object} homeworkData - 作业数据，包含homeworkId和answers
 * @returns {Promise} - 返回提交结果
 */
export const submitHomeworkAnswer = (homeworkData) => {
  // 实际应用中应调用真实API
  // 这里返回模拟数据
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
          success: true,
        message: '提交成功'
      });
    }, 500);
  });
};

/**
 * 保存作业进度
 * @param {Object} homeworkData - 作业数据，包含homeworkId和answers
 * @returns {Promise} - 返回保存结果
 */
export const saveHomeworkProgress = (homeworkData) => {
  // 实际应用中应调用真实API
  // 这里返回模拟数据
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
          success: true,
        message: '保存成功'
      });
    }, 500);
  });
};