/**
 * 用户服务
 * 处理用户相关的API请求
 */

/**
 * 获取用户上下文信息
 * @param {string} userId - 用户ID
 * @returns {Promise} - 返回用户上下文信息
 */
export const getUserContext = (userId) => {
  // 实际应用中应调用真实API
  // 这里返回模拟数据
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        data: {
          userId,
          recentTopics: ['一元二次方程', '因式分解', '配方法', '公式法'],
          weakPoints: [
            {
              id: 1,
              name: '判别式应用',
              proficiency: 60,
              lastPracticeDate: '2023-06-10'
            },
            {
              id: 2,
              name: '配方法',
              proficiency: 65,
              lastPracticeDate: '2023-06-12'
            }
          ],
          learningStyle: 'visual',
          proficiency: {
            algebra: 0.75,
            geometry: 0.82,
            calculus: 0.68,
            probability: 0.70
          },
          // 学生个性化数据，用于推荐
          personalData: {
            preferredSymbols: ['√', '∫', '∑', 'π'],
            commonErrors: ['配方不完整', '符号错误', '计算错误'],
            learningPace: 'medium',
            interactionHistory: [
              {
                date: '2023-06-14',
                topic: '一元二次方程',
                performance: 0.85
              },
              {
                date: '2023-06-12',
                topic: '因式分解',
                performance: 0.92
              },
              {
                date: '2023-06-10',
                topic: '配方法',
                performance: 0.65
              }
            ]
          }
        }
      });
    }, 300);
  });
};
