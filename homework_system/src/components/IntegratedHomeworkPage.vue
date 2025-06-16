<!--
  文件名: IntegratedHomeworkPage.vue
  描述: 集成四个模块的主页面，包括作业管理、知识推荐、练习推荐和反馈模块
  创建日期: 2023-06-15
-->

<template>
  <div class="integrated-homework-page">
    <!-- 顶部导航栏 -->
    <header class="page-header">
      <div class="header-left">
        <h2>智能作业系统</h2>
      </div>
      <div class="header-right">
        <el-dropdown v-if="user" trigger="click">
          <span class="el-dropdown-link">
            {{ user.name }} <i class="el-icon-arrow-down el-icon--right"></i>
          </span>
          <el-dropdown-menu slot="dropdown">
            <el-dropdown-item>个人中心</el-dropdown-item>
            <el-dropdown-item>设置</el-dropdown-item>
            <el-dropdown-item divided>退出登录</el-dropdown-item>
          </el-dropdown-menu>
        </el-dropdown>
      </div>
    </header>
    
    <!-- 主要内容区域 -->
    <main class="page-content">
      <!-- 左侧作业管理 -->
      <div class="left-panel">
        <HomeworkManagement 
          :homeworks="homeworks"
          :activeHomeworkId="currentHomework ? currentHomework.id : ''"
          @select-homework="selectHomework"
          @refresh-homeworks="fetchHomeworks"
        />
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
                    <el-input
                      type="textarea"
                      :rows="4"
                      placeholder="在此输入答案..."
                      v-model="answers[question.id]"
                      @input="saveProgress"
                    ></el-input>
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
      <div class="right-panel">
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
      saveTimeout: null
    };
  },
  
  computed: {
    ...mapGetters({
      user: 'getUser',
      homeworks: 'getHomeworks',
      currentHomework: 'getCurrentHomework',
      userContext: 'getUserContext'
    }),
    
    // 当前选中的问题
    currentProblem() {
      if (!this.currentHomework || !this.activeProblemIds.length) return null;
      
      return this.currentHomework.questions.find(
        question => question.id === this.activeProblemIds[0]
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
    // 初始化数据
    this.initializeData();
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
        // 模拟登录
        await this.login({ username: 'student', password: 'password' });
        
        // 获取作业列表
        await this.fetchHomeworks();
        
        // 获取用户上下文
        await this.fetchUserContext();
      } catch (error) {
        console.error('初始化数据失败', error);
        this.$message.error('加载数据失败，请刷新页面重试');
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
        
        // 默认展开第一个问题
        if (homework.questions && homework.questions.length > 0) {
          this.activeProblemIds = [homework.questions[0].id];
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
          
          this.$message.success('进度已保存');
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
    }
  }
};
</script>

<style scoped>
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
}

.center-panel {
  flex: 1;
  padding: 20px;
  overflow-y: auto;
}

.right-panel {
  width: 320px;
  border-left: 1px solid #e6e6e6;
  overflow: hidden;
  display: flex;
  flex-direction: column;
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
</style>
