import Vue from 'vue';
import Vuex from 'vuex';

Vue.use(Vuex);

export default new Vuex.Store({
  state: {
    // 用户信息
    user: null,
    
    // 当前作业
    currentHomework: null,
    
    // 作业列表
    homeworks: [],
    
    // 用户上下文（用于推荐）
    userContext: null
  },
  
  mutations: {
    // 设置用户信息
    SET_USER(state, user) {
      state.user = user;
    },
    
    // 设置当前作业
    SET_CURRENT_HOMEWORK(state, homework) {
      state.currentHomework = homework;
    },
    
    // 设置作业列表
    SET_HOMEWORKS(state, homeworks) {
      state.homeworks = homeworks;
    },
    
    // 设置用户上下文
    SET_USER_CONTEXT(state, context) {
      state.userContext = context;
    },
    
    // 更新作业状态
    UPDATE_HOMEWORK_STATUS(state, { homeworkId, status }) {
      const homework = state.homeworks.find(hw => hw.id === homeworkId);
      if (homework) {
        homework.status = status;
      }
      
      if (state.currentHomework && state.currentHomework.id === homeworkId) {
        state.currentHomework.status = status;
      }
    },
    
    // 保存作业答案
    SAVE_HOMEWORK_ANSWERS(state, { homeworkId, answers }) {
      const homework = state.homeworks.find(hw => hw.id === homeworkId);
      if (homework) {
        homework.savedAnswers = { ...homework.savedAnswers, ...answers };
      }
      
      if (state.currentHomework && state.currentHomework.id === homeworkId) {
        state.currentHomework.savedAnswers = { ...state.currentHomework.savedAnswers, ...answers };
      }
    }
  },
  
  actions: {
    // 登录
    async login({ commit }) {
      try {
        // 实际应用中应调用API进行登录
        // 这里使用模拟数据
        const user = {
          id: 'user123',
          name: '张三',
          avatar: 'https://example.com/avatar.jpg',
          grade: '初二',
          class: '3班'
        };
        
        commit('SET_USER', user);
        return user;
      } catch (error) {
        console.error('登录失败', error);
        throw error;
      }
    },
    
    // 获取作业列表
    async fetchHomeworks({ commit }) {
      try {
        // 实际应用中应调用API获取作业列表
        // 这里使用模拟数据
        const response = await import('../services/homeworkService').then(
          module => module.fetchHomeworkList()
        );
        
        commit('SET_HOMEWORKS', response.data);
        return response.data;
      } catch (error) {
        console.error('获取作业列表失败', error);
        throw error;
      }
    },
    
    // 获取作业详情
    async fetchHomeworkDetail({ commit }, homeworkId) {
      try {
        // 实际应用中应调用API获取作业详情
        // 这里使用模拟数据
        const response = await import('../services/homeworkService').then(
          module => module.fetchHomeworkDetail(homeworkId)
        );
        
        commit('SET_CURRENT_HOMEWORK', response.data);
        return response.data;
      } catch (error) {
        console.error('获取作业详情失败', error);
        throw error;
      }
    },
    
    // 获取用户上下文
    async fetchUserContext({ commit, state }) {
      try {
        if (!state.user) return null;
        
        // 实际应用中应调用API获取用户上下文
        // 这里使用模拟数据
        const response = await import('../services/userService').then(
          module => module.getUserContext(state.user.id)
        );
        
        commit('SET_USER_CONTEXT', response.data);
        return response.data;
      } catch (error) {
        console.error('获取用户上下文失败', error);
        throw error;
      }
    },
    
    // 提交作业
    async submitHomework({ commit }, { homeworkId, answers }) {
      try {
        // 实际应用中应调用API提交作业
        // 这里使用模拟数据
        const response = await import('../services/homeworkService').then(
          module => module.submitHomeworkAnswer({ homeworkId, answers })
        );
        
        commit('UPDATE_HOMEWORK_STATUS', { homeworkId, status: 'submitted' });
        return response.data;
      } catch (error) {
        console.error('提交作业失败', error);
        throw error;
      }
    },
    
    // 保存作业进度
    async saveHomeworkProgress({ commit }, { homeworkId, answers }) {
      try {
        // 实际应用中应调用API保存进度
        // 这里使用模拟数据
        const response = await import('../services/homeworkService').then(
          module => module.saveHomeworkProgress({ homeworkId, answers })
        );
        
        commit('SAVE_HOMEWORK_ANSWERS', { homeworkId, answers });
        commit('UPDATE_HOMEWORK_STATUS', { homeworkId, status: 'in_progress' });
        return response.data;
      } catch (error) {
        console.error('保存作业进度失败', error);
        throw error;
      }
    }
  },
  
  getters: {
    // 获取用户信息
    getUser: state => state.user,
    
    // 获取当前作业
    getCurrentHomework: state => state.currentHomework,
    
    // 获取作业列表
    getHomeworks: state => state.homeworks,
    
    // 获取用户上下文
    getUserContext: state => state.userContext,
    
    // 获取未完成作业
    getIncompleteHomeworks: state => {
      return state.homeworks.filter(
        hw => hw.status === 'not_started' || hw.status === 'in_progress'
      );
    },
    
    // 获取已提交作业
    getSubmittedHomeworks: state => {
      return state.homeworks.filter(
        hw => hw.status === 'submitted' || hw.status === 'graded'
      );
    }
  }
});
