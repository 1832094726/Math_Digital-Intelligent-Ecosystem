/**
 * 知识点服务
 * 处理知识点相关的API请求
 */

import axios from 'axios';

// API基础URL
const API_BASE_URL = '/api';

/**
 * 获取题目相关的知识点
 * @param {number} questionId - 题目ID
 * @returns {Promise} - 返回知识点列表
 */
export const getQuestionKnowledgePoints = (questionId) => {
  return axios.get(`${API_BASE_URL}/knowledge/question`, {
    params: { questionId }
  });
};

/**
 * 根据题目内容获取相关知识点
 * @param {string} questionText - 题目内容
 * @returns {Promise} - 返回知识点列表
 */
export const getKnowledgePointsByText = (questionText) => {
  return axios.get(`${API_BASE_URL}/knowledge/question`, {
    params: { text: questionText }
  });
};

/**
 * 提交题目内容获取相关知识点
 * @param {Object} data - 包含questionId或text的数据对象
 * @returns {Promise} - 返回知识点列表
 */
export const submitQuestionForKnowledge = (data) => {
  return axios.post(`${API_BASE_URL}/knowledge/question`, data);
}; 