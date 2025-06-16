/**
 * 作业服务
 * 处理作业相关的API请求
 */

import axios from 'axios';

// API基础URL
const API_BASE_URL = '/api';

/**
 * 获取作业列表
 * @returns {Promise} - 返回作业列表
 */
export const fetchHomeworkList = () => {
  return axios.get(`${API_BASE_URL}/homework/list`);
};

/**
 * 获取作业详情
 * @param {string} homeworkId - 作业ID
 * @returns {Promise} - 返回作业详情
 */
export const fetchHomeworkDetail = (homeworkId) => {
  return axios.get(`${API_BASE_URL}/homework/detail/${homeworkId}`);
};

/**
 * 提交作业答案
 * @param {Object} homeworkData - 作业数据，包含homeworkId和answers
 * @returns {Promise} - 返回提交结果
 */
export const submitHomeworkAnswer = (homeworkData) => {
  return axios.post(`${API_BASE_URL}/homework/submit`, homeworkData);
};

/**
 * 保存作业进度
 * @param {Object} homeworkData - 作业数据，包含homeworkId和answers
 * @returns {Promise} - 返回保存结果
 */
export const saveHomeworkProgress = (homeworkData) => {
  return axios.post(`${API_BASE_URL}/homework/save`, homeworkData);
};