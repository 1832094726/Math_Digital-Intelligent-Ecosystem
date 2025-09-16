<!--
  文件名: IntegratedHomeworkPage.vue
  描述: 集成四个模块的主页面，包括作业管理、知识推荐、练习推荐和反馈模块
  创建日期: 2023-06-15
-->

<template>
  <div class="integrated-homework-page homework-container">
    <!-- 顶部导航栏 -->
    <header class="page-header">
      <div class="header-left">
        <!-- 移动端左侧菜单按钮 -->
        <el-button
          class="mobile-menu-btn"
          icon="el-icon-menu"
          @click="toggleLeftPanel"
          v-show="isMobile"
          type="text"
          size="medium"
        ></el-button>
        <h2>智能作业系统</h2>
      </div>
      <div class="header-right">
        <!-- 用户信息 -->
        <div class="user-info" v-if="user">
          <span class="user-name">{{ user.real_name || user.username }}</span>
          <span class="user-role">{{ getRoleText(user.role) }}</span>
        </div>

        <!-- 退出登录按钮 -->
        <el-button
          class="logout-btn"
          icon="el-icon-switch-button"
          @click="handleLogout"
          type="text"
          size="medium"
        >退出</el-button>

        <!-- 移动端右侧面板按钮 -->
        <el-button
          class="mobile-panel-btn"
          icon="el-icon-s-grid"
          @click="toggleRightPanel"
          v-show="isMobile"
          type="text"
          size="medium"
        ></el-button>
      </div>
    </header>
    
    <!-- 主要内容区域 -->
    <main class="page-content">
      <!-- 左侧作业管理 -->
      <div class="left-panel" :class="{ 'panel-hidden': isMobile && !showLeftPanel }">
        <!-- 移动端遮罩层 -->
        <div class="panel-overlay" v-if="isMobile && showLeftPanel" @click="hideLeftPanel"></div>
        <div class="panel-content">
          <HomeworkManagement
            :homeworks="homeworks"
            :activeHomeworkId="currentHomework ? currentHomework.id : ''"
            @select-homework="selectHomework"
            @refresh-homeworks="fetchHomeworks"
          />
        </div>
      </div>
      
      <!-- 中间作业内容 -->
      <div class="center-panel">
        <div v-if="loading" class="loading-container">
          <el-skeleton :rows="10" animated />
        </div>
        
        <div v-else-if="!currentHomework" class="empty-state">
          <i class="el-icon-document"></i>
          <p>请从左侧选择一个作业</p>
        </div>
        
        <div v-else class="homework-content">
          <div class="homework-header">
            <h3>{{ currentHomework.title }}</h3>
            <div class="homework-meta">
              <span><i class="el-icon-date"></i> 截止日期: {{ formatDate(currentHomework.deadline) }}</span>
              <span><i class="el-icon-medal"></i> 难度: {{ getDifficultyText(currentHomework.difficulty) }}</span>
            </div>
            <p class="homework-description">{{ currentHomework.description }}</p>
          </div>
          
          <div class="problem-list">
            <el-collapse v-model="activeProblemIds">
              <el-collapse-item
                v-for="question in currentHomework.questions"
                :key="question.id"
                :title="`问题 ${question.id} (${question.score}分)`"
                :name="question.id"
                :class="{ 'selected-question': selectedQuestionId === question.id }"
                @click.native="selectQuestion(question.id)"
              >
                <div class="problem-content">
                  <div class="problem-statement" v-html="question.content"></div>
                  
                  <div v-if="question.options && question.options.length > 0" class="options-area">
                    <el-radio-group v-model="answers[question.id]">
                      <el-radio 
                        v-for="(option, index) in question.options" 
                        :key="index"
                        :label="option"
                      >{{ option }}</el-radio>
                    </el-radio-group>
                  </div>
                  <div v-else class="answer-area">
                    <div
                      class="answer-input-container"
                      :class="{ 'panel-active': showSymbolPanel && currentQuestion && currentQuestion.id === question.id }"
                    >
                      <el-input
                        type="textarea"
                        :rows="4"
                        placeholder="在此输入答案..."
                        v-model="answers[question.id]"
                        @input="saveProgress"
                        @focus="onAnswerFocus(question, $event)"
                        ref="answerInput"
                        class="answer-input"
                      ></el-input>

                      <!-- 符号推荐面板 -->
                      <div
                        v-if="showSymbolPanel && currentQuestion && currentQuestion.id === question.id"
                        class="symbol-recommendation-panel"
                      >
                        <div class="panel-header">
                          <h4><i class="el-icon-magic-stick"></i> 推荐符号</h4>
                          <div class="header-actions">
                            <!-- 状态指示器 -->
                            <div class="status-indicator">
                              <span v-if="saving" class="status-item">
                                <i class="el-icon-loading"></i>
                                <span>保存中</span>
                              </span>
                              <span v-else-if="symbolLoading" class="status-item">
                                <i class="el-icon-loading"></i>
                                <span>加载中</span>
                              </span>
                              <span v-else class="status-item ready">
                                <i class="el-icon-check"></i>
                                <span>就绪</span>
                              </span>
                            </div>
                            <el-button
                              type="text"
                              icon="el-icon-close"
                              @click="closeSymbolPanel"
                              class="close-btn"
                            ></el-button>
                          </div>
                        </div>

                        <div v-if="symbolLoading" class="loading-state">
                          <i class="el-icon-loading"></i>
                          <span>加载推荐符号中...</span>
                        </div>

                        <div v-else class="symbol-content">
                          <!-- 智能推荐符号 - 改名并放在最上面 -->
                          <div v-if="getCustomRecommendedSymbols(question).length > 0" class="symbol-category">
                            <h5>💡 智能推荐</h5>
                            <div class="symbol-grid">
                              <button
                                v-for="symbol in getCustomRecommendedSymbols(question)"
                                :key="symbol.id"
                                class="symbol-btn"
                                :title="symbol.description"
                                @click="insertSymbol(symbol.symbol, question.id)"
                              >
                                {{ symbol.symbol }}
                              </button>
                            </div>
                          </div>

                          <!-- 基础数学符号 -->
                          <div class="symbol-category">
                            <h5>➕ 基础运算</h5>
                            <div class="symbol-grid">
                              <button
                                v-for="symbol in basicSymbols"
                                :key="symbol.id"
                                class="symbol-btn"
                                :title="symbol.description"
                                @click="insertSymbol(symbol.symbol, question.id)"
                              >
                                {{ symbol.symbol }}
                              </button>
                            </div>
                          </div>

                          <!-- 几何符号 -->
                          <div class="symbol-category">
                            <h5>📐 几何符号</h5>
                            <div class="symbol-grid">
                              <button
                                v-for="symbol in geometrySymbols"
                                :key="symbol.id"
                                class="symbol-btn"
                                :title="symbol.description"
                                @click="insertSymbol(symbol.symbol, question.id)"
                              >
                                {{ symbol.symbol }}
                              </button>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </el-collapse-item>
            </el-collapse>
          </div>
          
          <div class="homework-actions">
            <el-button 
              type="primary" 
              :disabled="!canSubmit"
              @click="submitHomework"
              :loading="submitting"
            >提交作业</el-button>
            <el-button 
              type="info" 
              @click="saveProgress"
              :loading="saving"
            >保存进度</el-button>
          </div>
          
          <!-- 反馈区域 -->
          <div v-if="feedback" class="feedback-container">
            <FeedbackModule :feedback="feedback" />
          </div>
        </div>
      </div>
      
      <!-- 右侧推荐面板 -->
      <div class="right-panel" :class="{ 'panel-hidden': isMobile && !showRightPanel }">
        <!-- 移动端遮罩层 -->
        <div class="panel-overlay" v-if="isMobile && showRightPanel" @click="hideRightPanel"></div>
        <div class="panel-content">
          <el-tabs v-model="activeTab" type="card">
            <el-tab-pane label="知识推荐" name="knowledge">
              <KnowledgeRecommendation
                :currentProblem="currentProblem"
                :userContext="userContext"
                @knowledge-selected="applyKnowledge"
              />
            </el-tab-pane>
            <el-tab-pane label="练习推荐" name="exercise">
              <ExerciseRecommendation
                :currentProblem="currentProblem"
                :userContext="userContext"
                @exercise-selected="selectExercise"
              />
            </el-tab-pane>
          </el-tabs>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { mapGetters, mapActions } from 'vuex';
import HomeworkManagement from './HomeworkManagement.vue';
import KnowledgeRecommendation from './KnowledgeRecommendation.vue';
import ExerciseRecommendation from './ExerciseRecommendation.vue';
import FeedbackModule from './FeedbackModule.vue';
import { formatDate } from '../utils/dateFormat';
import axios from 'axios';

export default {
  name: 'IntegratedHomeworkPage',

  components: {
    HomeworkManagement,
    KnowledgeRecommendation,
    ExerciseRecommendation,
    FeedbackModule
  },


  
  data() {
    return {
      loading: false,
      submitting: false,
      saving: false,
      activeProblemIds: [],
      activeTab: 'knowledge',
      answers: {},
      feedback: null,
      saveTimeout: null,

      // 当前选中的题目ID（用于右侧推荐）
      selectedQuestionId: null,

      // 响应式布局相关
      isMobile: false,
      showLeftPanel: false,
      showRightPanel: false,

      // 符号推荐相关
      showSymbolPanel: false,
      symbolLoading: false,
      currentQuestion: null,
      recommendedSymbols: [],

      // 基础数学符号
      basicSymbols: [
        { id: 1, symbol: '+', description: '加号' },
        { id: 2, symbol: '-', description: '减号' },
        { id: 3, symbol: '×', description: '乘号' },
        { id: 4, symbol: '÷', description: '除号' },
        { id: 5, symbol: '=', description: '等号' },
        { id: 6, symbol: '≠', description: '不等号' },
        { id: 7, symbol: '>', description: '大于' },
        { id: 8, symbol: '<', description: '小于' },
        { id: 9, symbol: '≥', description: '大于等于' },
        { id: 10, symbol: '≤', description: '小于等于' },
        { id: 11, symbol: '²', description: '平方' },
        { id: 12, symbol: '³', description: '立方' },
        { id: 13, symbol: '√', description: '根号' },
        { id: 14, symbol: 'π', description: '圆周率' }
      ],

      // 几何符号
      geometrySymbols: [
        { id: 15, symbol: '∠', description: '角' },
        { id: 16, symbol: '△', description: '三角形' },
        { id: 17, symbol: '□', description: '正方形' },
        { id: 18, symbol: '○', description: '圆' },
        { id: 19, symbol: '∥', description: '平行' },
        { id: 20, symbol: '⊥', description: '垂直' },
        { id: 21, symbol: '∽', description: '相似' },
        { id: 22, symbol: '≅', description: '全等' }
      ]
    };
  },
  
  computed: {
    ...mapGetters({
      user: 'getUser',
      homeworks: 'getHomeworks',
      currentHomework: 'getCurrentHomework',
      userContext: 'getUserContext'
    }),
    
    // 当前选中的问题（用于右侧推荐）
    currentProblem() {
      if (!this.currentHomework || !this.selectedQuestionId) return null;

      return this.currentHomework.questions.find(
        question => question.id === this.selectedQuestionId
      );
    },
    
    // 是否可以提交
    canSubmit() {
      if (!this.currentHomework) return false;
      
      // 检查是否所有问题都已回答
      return this.currentHomework.questions.every(
        question => this.answers[question.id]
      );
    }
  },
  
  created() {
    // 配置axios实例
    this.$http = axios.create({
      baseURL: 'http://localhost:8081',
      timeout: 10000
    });

    // 从localStorage获取token并设置到axios header
    const token = localStorage.getItem('token');
    if (token) {
      this.$http.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    }

    // 初始化数据
    this.initializeData();
  },

  mounted() {
    // 初始化响应式检查
    this.checkMobile();

    // 监听窗口大小变化
    window.addEventListener('resize', this.handleResize);
  },

  beforeDestroy() {
    // 移除事件监听
    window.removeEventListener('resize', this.handleResize);
  },
  
  methods: {
    ...mapActions([
      'login',
      'fetchHomeworks',
      'fetchHomeworkDetail',
      'fetchUserContext',
      'submitHomework',
      'saveHomeworkProgress'
    ]),
    
    // 初始化数据
    async initializeData() {
      try {
        // 检查是否已登录
        const token = localStorage.getItem('token');
        if (!token) {
          this.$router.push('/login');
          return;
        }

        // 获取作业列表
        await this.fetchHomeworks();

        // 获取用户上下文
        await this.fetchUserContext();
      } catch (error) {
        console.error('初始化数据失败', error);

        // 如果是认证错误，跳转到登录页
        if (error.response && error.response.status === 401) {
          localStorage.removeItem('token');
          this.$router.push('/login');
        } else {
          this.$message.error('加载数据失败，请刷新页面重试');
        }
      }
    },
    
    // 选择作业
    async selectHomework(homeworkId) {
      this.loading = true;
      this.answers = {};
      this.feedback = null;
      
      try {
        // 获取作业详情
        const homework = await this.fetchHomeworkDetail(homeworkId);
        
        // 初始化答案
        if (homework.savedAnswers) {
          this.answers = { ...homework.savedAnswers };
        }
        
        // 如果是已批改的作业，显示反馈
        if (homework.status === 'graded' && homework.feedback) {
          this.feedback = homework.feedback;
        }
        
        // 默认展开所有问题
        if (homework.questions && homework.questions.length > 0) {
          this.activeProblemIds = homework.questions.map(q => q.id);
          // 默认选中第一个题目用于右侧推荐
          this.selectedQuestionId = homework.questions[0].id;
        }
      } catch (error) {
        console.error('获取作业详情失败', error);
        this.$message.error('获取作业详情失败');
      } finally {
        this.loading = false;
      }
    },
    
    // 保存进度
    saveProgress() {
      // 防抖处理，避免频繁保存
      clearTimeout(this.saveTimeout);
      this.saveTimeout = setTimeout(async () => {
        if (!this.currentHomework) return;

        this.saving = true;

        try {
          await this.saveHomeworkProgress({
            homeworkId: this.currentHomework.id,
            answers: this.answers
          });

          // 移除频繁的成功提示
          // this.$message.success('进度已保存');
        } catch (error) {
          console.error('保存进度失败', error);
          this.$message.error('保存进度失败');
        } finally {
          this.saving = false;
        }
      }, 1000);
    },
    
    // 提交作业
    async submitHomework() {
      if (!this.currentHomework) return;
      
      this.submitting = true;
      
      try {
        const result = await this.submitHomework({
          homeworkId: this.currentHomework.id,
          answers: this.answers
        });
        
        this.$message.success('作业提交成功');
        
        // 显示反馈
        if (result.feedback) {
          this.feedback = result.feedback;
        }
      } catch (error) {
        console.error('提交作业失败', error);
        this.$message.error('提交作业失败');
      } finally {
        this.submitting = false;
      }
    },
    
    // 应用知识点
    applyKnowledge(knowledge) {
      if (!this.currentProblem) return;
      
      // 在当前答案中插入知识点
      const problemId = this.currentProblem.id;
      const currentAnswer = this.answers[problemId] || '';
      
      // 根据知识点类型应用不同的内容
      let contentToInsert = '';
      
      if (knowledge.formulas && knowledge.formulas.length > 0) {
        contentToInsert = knowledge.formulas[0];
      } else {
        contentToInsert = knowledge.title;
      }
      
      this.answers[problemId] = currentAnswer + '\n' + contentToInsert;
      this.saveProgress();
    },
    
    // 选择练习
    selectExercise(exercise) {
      // 实际应用中应跳转到练习页面
      this.$message.info(`即将开始练习：${exercise.title}`);
    },

    // 选择题目（用于右侧推荐）
    selectQuestion(questionId) {
      this.selectedQuestionId = questionId;
    },

    // 响应式布局方法

    // 检查是否为移动端
    checkMobile() {
      this.isMobile = window.innerWidth <= 768;
    },

    // 切换左侧面板
    toggleLeftPanel() {
      console.log('toggleLeftPanel called, current state:', this.showLeftPanel);
      console.log('isMobile:', this.isMobile);
      this.showLeftPanel = !this.showLeftPanel;
      if (this.showLeftPanel) {
        this.showRightPanel = false; // 关闭右侧面板
      }
      console.log('new showLeftPanel state:', this.showLeftPanel);
    },

    // 切换右侧面板
    toggleRightPanel() {
      console.log('toggleRightPanel called, current state:', this.showRightPanel);
      console.log('isMobile:', this.isMobile);
      this.showRightPanel = !this.showRightPanel;
      if (this.showRightPanel) {
        this.showLeftPanel = false; // 关闭左侧面板
      }
      console.log('new showRightPanel state:', this.showRightPanel);
    },

    // 隐藏左侧面板
    hideLeftPanel() {
      this.showLeftPanel = false;
    },

    // 隐藏右侧面板
    hideRightPanel() {
      this.showRightPanel = false;
    },

    // 窗口大小变化处理
    handleResize() {
      this.checkMobile();

      // 如果切换到桌面端，重置面板状态
      if (!this.isMobile) {
        this.showLeftPanel = false;
        this.showRightPanel = false;
      }
    },

    // 符号推荐相关方法

    // 答案输入框获得焦点时
    async onAnswerFocus(question, event) {
      this.currentQuestion = question;
      this.showSymbolPanel = true;

      // 更新选中的题目（用于右侧推荐）
      this.selectedQuestionId = question.id;

      // 平滑滚动确保面板可见
      this.$nextTick(() => {
        setTimeout(() => {
          const inputElement = event.target;
          if (inputElement) {
            inputElement.scrollIntoView({
              behavior: 'smooth',
              block: 'center'
            });
          }
        }, 100); // 等待面板渲染完成
      });

      // 获取智能推荐符号
      await this.loadRecommendedSymbols(question);
    },

    // 关闭符号面板
    closeSymbolPanel() {
      this.showSymbolPanel = false;
      this.currentQuestion = null;
      this.recommendedSymbols = [];
    },

    // 加载推荐符号
    async loadRecommendedSymbols(question) {
      this.symbolLoading = true;

      try {
        // 调用符号推荐服务
        const symbolService = await import('../services/symbolRecommendationService');

        const response = await symbolService.getSymbolRecommendations({
          user_id: this.user?.id || 1,
          question_id: question.id,
          question_text: question.content,
          current_topic: question.knowledge_points?.[0] || '',
          difficulty_level: this.currentHomework?.difficulty || 'medium'
        });

        if (response.data && response.data.symbols) {
          this.recommendedSymbols = response.data.symbols.map(symbol => ({
            id: symbol.id,
            symbol: symbol.symbol,
            description: symbol.description,
            category: symbol.category,
            relevance: symbol.relevance
          }));
        } else {
          // 如果没有返回符号，使用默认推荐
          this.recommendedSymbols = this.getDefaultRecommendedSymbols(question);
        }
      } catch (error) {
        console.error('获取推荐符号失败:', error);
        // 使用默认推荐符号
        this.recommendedSymbols = this.getDefaultRecommendedSymbols(question);
      } finally {
        this.symbolLoading = false;
      }
    },

    // 获取定制推荐符号（基于题目解法的写死接口）
    getCustomRecommendedSymbols(question) {
      const content = question.content.toLowerCase();
      const customRecommended = [];

      // 根据题目类型和解法推荐特定符号

      // 一元二次方程类题目
      if (content.includes('方程') && (content.includes('x²') || content.includes('x^2') || content.includes('二次'))) {
        customRecommended.push(
          { id: 'custom1', symbol: 'x', description: '未知数x' },
          { id: 'custom2', symbol: '²', description: '平方' },
          { id: 'custom3', symbol: '=', description: '等号' },
          { id: 'custom4', symbol: '±', description: '正负号' },
          { id: 'custom5', symbol: '√', description: '根号' }
        );
      }

      // 梯形面积类题目
      else if (content.includes('梯形') && content.includes('面积')) {
        customRecommended.push(
          { id: 'custom6', symbol: 'S', description: '面积S' },
          { id: 'custom7', symbol: '=', description: '等号' },
          { id: 'custom8', symbol: '(', description: '左括号' },
          { id: 'custom9', symbol: ')', description: '右括号' },
          { id: 'custom10', symbol: '+', description: '加号' },
          { id: 'custom11', symbol: '×', description: '乘号' },
          { id: 'custom12', symbol: '÷', description: '除号' },
          { id: 'custom13', symbol: '2', description: '数字2' }
        );
      }

      // 圆形相关题目
      else if (content.includes('圆') && (content.includes('面积') || content.includes('周长'))) {
        customRecommended.push(
          { id: 'custom14', symbol: 'π', description: '圆周率' },
          { id: 'custom15', symbol: 'r', description: '半径r' },
          { id: 'custom16', symbol: '²', description: '平方' },
          { id: 'custom17', symbol: '×', description: '乘号' },
          { id: 'custom18', symbol: '=', description: '等号' }
        );
      }

      // 三角形相关题目
      else if (content.includes('三角形') && (content.includes('面积') || content.includes('角'))) {
        customRecommended.push(
          { id: 'custom19', symbol: '△', description: '三角形' },
          { id: 'custom20', symbol: '∠', description: '角' },
          { id: 'custom21', symbol: '°', description: '度' },
          { id: 'custom22', symbol: '=', description: '等号' },
          { id: 'custom23', symbol: '÷', description: '除号' }
        );
      }

      // 分数相关题目
      else if (content.includes('分数') || content.includes('分子') || content.includes('分母')) {
        customRecommended.push(
          { id: 'custom24', symbol: '/', description: '分数线' },
          { id: 'custom25', symbol: '+', description: '加号' },
          { id: 'custom26', symbol: '-', description: '减号' },
          { id: 'custom27', symbol: '=', description: '等号' },
          { id: 'custom28', symbol: '(', description: '左括号' },
          { id: 'custom29', symbol: ')', description: '右括号' }
        );
      }

      // 百分比相关题目
      else if (content.includes('%') || content.includes('百分') || content.includes('折扣') || content.includes('利率')) {
        customRecommended.push(
          { id: 'custom30', symbol: '%', description: '百分号' },
          { id: 'custom31', symbol: '×', description: '乘号' },
          { id: 'custom32', symbol: '=', description: '等号' },
          { id: 'custom33', symbol: '+', description: '加号' },
          { id: 'custom34', symbol: '-', description: '减号' }
        );
      }

      // 一般计算题
      else if (content.includes('计算') || content.includes('求') || content.includes('多少')) {
        customRecommended.push(
          { id: 'custom35', symbol: '=', description: '等号' },
          { id: 'custom36', symbol: '+', description: '加号' },
          { id: 'custom37', symbol: '-', description: '减号' },
          { id: 'custom38', symbol: '×', description: '乘号' },
          { id: 'custom39', symbol: '÷', description: '除号' }
        );
      }

      return customRecommended;
    },

    // 获取默认推荐符号（基于题目内容的简单匹配）
    getDefaultRecommendedSymbols(question) {
      const content = question.content.toLowerCase();
      const recommended = [];

      // 根据题目内容推荐符号
      if (content.includes('方程') || content.includes('解')) {
        recommended.push(
          { id: 'rec1', symbol: 'x', description: '未知数x' },
          { id: 'rec2', symbol: 'y', description: '未知数y' },
          { id: 'rec3', symbol: '=', description: '等号' }
        );
      }

      if (content.includes('面积') || content.includes('周长')) {
        recommended.push(
          { id: 'rec4', symbol: '²', description: '平方' },
          { id: 'rec5', symbol: 'π', description: '圆周率' },
          { id: 'rec6', symbol: '×', description: '乘号' }
        );
      }

      if (content.includes('角') || content.includes('三角形')) {
        recommended.push(
          { id: 'rec7', symbol: '∠', description: '角' },
          { id: 'rec8', symbol: '△', description: '三角形' },
          { id: 'rec9', symbol: '°', description: '度' }
        );
      }

      return recommended;
    },

    // 插入符号到答案中
    async insertSymbol(symbol, questionId) {
      const currentAnswer = this.answers[questionId] || '';

      // 获取当前光标位置（简化处理，追加到末尾）
      this.answers[questionId] = currentAnswer + symbol;

      // 保存进度
      this.saveProgress();

      // 更新符号使用统计
      try {
        const symbolService = await import('../services/symbolRecommendationService');
        await symbolService.updateSymbolUsage({
          user_id: this.user?.id || 1,
          question_id: questionId,
          symbol: symbol
        });
      } catch (error) {
        console.error('更新符号使用统计失败:', error);
      }

      // 聚焦回答案输入框
      this.$nextTick(() => {
        const inputRef = this.$refs.answerInput;
        if (inputRef && Array.isArray(inputRef)) {
          // 找到对应的输入框
          const targetInput = inputRef.find(input =>
            input.$el.closest('.problem-content')?.querySelector(`[name="${questionId}"]`)
          );
          if (targetInput) {
            targetInput.focus();
          }
        }
      });
    },
    
    // 获取难度文本
    getDifficultyText(difficulty) {
      const difficultyMap = {
        1: '简单',
        2: '中等',
        3: '困难',
        4: '挑战',
        5: '极难'
      };
      return difficultyMap[difficulty] || difficulty;
    },
    
    // 格式化日期
    formatDate(date) {
      return formatDate(date, 'YYYY-MM-DD HH:mm');
    },

    // 获取角色文本
    getRoleText(role) {
      const roleMap = {
        'student': '学生',
        'teacher': '教师',
        'admin': '管理员',
        'parent': '家长'
      };
      return roleMap[role] || '用户';
    },

    // 退出登录
    async handleLogout() {
      try {
        await this.$confirm('确定要退出登录吗？', '提示', {
          confirmButtonText: '确定',
          cancelButtonText: '取消',
          type: 'warning'
        });

        // 清除本地存储
        localStorage.removeItem('token');
        localStorage.removeItem('user');

        // 清除axios默认header
        delete this.$http.defaults.headers.common['Authorization'];

        // 跳转到登录页
        this.$router.push('/login');

        this.$message.success('已退出登录');

      } catch (error) {
        // 用户取消退出
        console.log('用户取消退出登录');
      }
    }
  }
};
</script>

<style>
.integrated-homework-page {
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 20px;
  height: 60px;
  background-color: #409EFF;
  color: white;
  position: relative;
  z-index: 1000;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 10px;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 15px;
}

.user-info {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  font-size: 14px;
}

.user-name {
  font-weight: 500;
  color: #333;
}

.user-role {
  font-size: 12px;
  color: #666;
  background: #f0f2f5;
  padding: 2px 8px;
  border-radius: 10px;
  margin-top: 2px;
}

.logout-btn {
  color: #666;
  font-size: 14px;
}

.logout-btn:hover {
  color: #409eff;
}

.mobile-menu-btn,
.mobile-panel-btn {
  display: none;
  color: white;
}

.header-left h2 {
  margin: 0;
  font-size: 20px;
}

.el-dropdown-link {
  color: white;
  cursor: pointer;
}

.page-content {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.left-panel {
  width: 280px;
  border-right: 1px solid #e6e6e6;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  position: relative;
  transition: transform 0.3s ease;
}

.left-panel .panel-content {
  height: 100%;
  flex: 1;
  position: relative;
  z-index: 1000;
  background: white;
}

.left-panel .panel-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  z-index: 998;
  display: none;
  pointer-events: auto;
}

.center-panel {
  flex: 1;
  padding: 20px;
  overflow-y: auto;
}

.right-panel {
  width: 320px;
  border-left: 1px solid #e6e6e6;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  position: relative;
  transition: transform 0.3s ease;
}

.right-panel .panel-content {
  height: 100%;
  flex: 1;
  position: relative;
  z-index: 1000;
  background: white;
}

.right-panel .panel-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  z-index: 998;
  display: none;
  pointer-events: auto;
}

.loading-container {
  padding: 20px;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 300px;
  color: #909399;
}

.empty-state i {
  font-size: 48px;
  margin-bottom: 20px;
}

.homework-header {
  margin-bottom: 20px;
}

.homework-header h3 {
  margin: 0 0 10px 0;
  font-size: 22px;
  color: #303133;
}

.homework-meta {
  display: flex;
  gap: 20px;
  font-size: 14px;
  color: #606266;
  margin-bottom: 10px;
}

.homework-description {
  color: #606266;
  line-height: 1.6;
}

.problem-list {
  margin-bottom: 20px;
}

.problem-content {
  padding: 10px 0;
}

.problem-statement {
  margin-bottom: 15px;
  line-height: 1.6;
}

.answer-area {
  margin-top: 10px;
}

.homework-actions {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
}

.feedback-container {
  margin-top: 30px;
  padding-top: 20px;
  border-top: 1px solid #e6e6e6;
}

/* 让内容区域自适应高度 */
.problem-content {
  height: auto !important;
  min-height: auto !important;
  max-height: none !important;
}

/* 调整 el-collapse-item 的阴影效果 */
.el-collapse-item {
  box-shadow: none !important; /* 去掉阴影 */
  border-radius: 6px;
  margin-bottom: 8px;
}

.el-collapse-item__header {
  padding: 12px 20px !important; /* 与内容区域保持一致的左右间距 */
  background-color: #fafafa;
  border-radius: 6px 6px 0 0;
}

.el-collapse-item.is-active .el-collapse-item__header {
  border-bottom: 1px solid #e6e6e6;
}

.el-collapse-item__wrap {
  border-radius: 0 0 6px 6px;
  overflow: hidden;
}

/* 选中题目的样式 */
.selected-question .el-collapse-item__header {
  background-color: #ECF5FF !important;
  border-left: 4px solid #409EFF !important;
  color: #409EFF !important;
  font-weight: 600 !important;
}

.selected-question .el-collapse-item__header:hover {
  background-color: #D9ECFF !important;
}

/* 符号推荐面板样式 - IEEE风格 */
.answer-input-container {
  position: relative;
  transition: margin-bottom 0.3s ease;
}

.answer-input-container.panel-active {
  margin-bottom: 35vh; /* 使用视窗高度的50%，更灵活 */
  position: relative;
}



.answer-input {
  width: 100%;
}

.symbol-recommendation-panel {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  background: #fefefe;
  border: 2px solid #409EFF;
  border-radius: 8px;
  box-shadow: none; /* 去掉阴影 */
  z-index: 1000;
  margin-top: 8px;
  max-height: 400px;
  overflow-y: auto;
  font-family: 'Times New Roman', serif;
  /* 确保面板不会被遮挡 */
  min-height: 200px;
}



.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: #fefefe;
  color: #606266;
  border-radius: 6px 6px 0 0;
  border-bottom: 1px solid #E4E7ED;
}

.panel-header h4 {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 6px;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.status-indicator {
  font-size: 12px;
}

.status-item {
  display: flex;
  align-items: center;
  gap: 4px;
  color: #909399;
}

.status-item.ready {
  color: #67C23A;
}

.status-item i {
  font-size: 12px;
}

.status-item .el-icon-loading {
  animation: rotating 2s linear infinite;
}

@keyframes rotating {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}

.close-btn {
  color: #909399 !important;
  padding: 4px !important;
}

.close-btn:hover {
  background-color: rgba(144, 147, 153, 0.1) !important;
  color: #606266 !important;
}

.loading-state {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
  color: #606266;
  gap: 8px;
}

.symbol-content {
  padding: 16px;
}

.symbol-category {
  margin-bottom: 16px;
}

.symbol-category:last-child {
  margin-bottom: 0;
}

.symbol-category h5 {
  margin: 0 0 8px 0;
  font-size: 12px;
  color: #606266;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  border-bottom: 1px solid #E4E7ED;
  padding-bottom: 4px;
  font-family: 'Times New Roman', serif;
}

.symbol-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(40px, 1fr));
  gap: 6px;
}

.symbol-btn {
  width: 40px;
  height: 40px;
  border: 1px solid #DCDFE6;
  background: #ffffff;
  border-radius: 6px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  font-weight: 500;
  transition: all 0.2s ease;
  color: #303133;
}

.symbol-btn:hover {
  border-color: #409EFF;
  background: #ECF5FF;
  color: #409EFF;
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(64, 158, 255, 0.2);
}

.symbol-btn:active {
  transform: translateY(0);
}



/* 响应式设计 */
@media (max-width: 768px) {
  /* 显示移动端菜单按钮 */
  .mobile-menu-btn,
  .mobile-panel-btn {
    display: inline-flex !important;
  }

  /* 主内容区域调整 */
  .main-content {
    position: relative;
  }

  /* 左侧面板移动端样式 */
  .left-panel {
    position: fixed;
    top: 60px; /* 头部高度 */
    left: 0;
    bottom: 0;
    width: 280px;
    background: white;
    z-index: 999;
    transform: translateX(-100%);
    border-right: 1px solid #e6e6e6;
    box-shadow: 2px 0 8px rgba(0, 0, 0, 0.15);
  }

  .left-panel:not(.panel-hidden) {
    transform: translateX(0);
  }

  .left-panel.panel-hidden {
    transform: translateX(-100%);
  }

  .left-panel .panel-overlay {
    display: block !important;
    z-index: 998;
  }

  /* 右侧面板移动端样式 */
  .right-panel {
    position: fixed;
    top: 60px;
    right: 0;
    bottom: 0;
    width: 320px;
    background: white;
    z-index: 999;
    transform: translateX(100%);
    border-left: 1px solid #e6e6e6;
    box-shadow: -2px 0 8px rgba(0, 0, 0, 0.15);
  }

  .right-panel:not(.panel-hidden) {
    transform: translateX(0);
  }

  .right-panel.panel-hidden {
    transform: translateX(100%);
  }

  .right-panel .panel-overlay {
    display: block !important;
    z-index: 998;
  }

  /* 中间内容区域占满 */
  .center-panel {
    width: 100%;
    margin: 0;
  }

  /* 符号推荐面板移动端调整 */
  .answer-input-container.panel-active {
    margin-bottom: 40vh;
  }

  .symbol-recommendation-panel {
    max-height: 300px;
  }

  .symbol-grid {
    grid-template-columns: repeat(auto-fill, minmax(35px, 1fr));
    gap: 4px;
  }

  .symbol-btn {
    width: 35px;
    height: 35px;
    font-size: 14px;
  }

  .panel-header {
    padding: 8px 12px;
  }

  .panel-header h4 {
    font-size: 12px;
  }

  .status-indicator {
    font-size: 10px;
  }
}

/* 隐藏面板的样式 */
.panel-hidden {
  transform: translateX(-100%) !important;
}

.right-panel.panel-hidden {
  transform: translateX(100%) !important;
}
</style>
