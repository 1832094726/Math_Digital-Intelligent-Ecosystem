<!--
  文件名: ExerciseRecommendation.vue
  描述: 练习推荐模块组件，根据学生学习情况推荐适合的练习题
  创建日期: 2023-06-15
-->

<template>
  <div class="exercise-recommendation">
    <div class="module-header">
      <h3>推荐练习</h3>
      <el-tooltip content="基于当前问题和您的学习情况，智能推荐相关练习题" placement="top">
        <i class="el-icon-question"></i>
      </el-tooltip>
    </div>
    
    <div v-if="loading" class="loading-state">
      <el-skeleton :rows="5" animated />
    </div>
    
    <div v-else-if="!currentProblem" class="empty-state">
      <i class="el-icon-document"></i>
      <p>请选择一道题目以获取相关练习推荐</p>
    </div>
    
    <template v-else>
      <!-- 推荐策略选择 -->
      <div class="recommendation-strategy">
        <span>推荐策略:</span>
        <el-radio-group v-model="strategy" size="mini" @change="fetchExercises">
          <el-radio-button label="similar">相似题目</el-radio-button>
          <el-radio-button label="complementary">互补知识</el-radio-button>
          <el-radio-button label="adaptive">自适应</el-radio-button>
        </el-radio-group>
      </div>
      
      <!-- 难度选择 -->
      <div class="difficulty-filter">
        <span>难度范围:</span>
        <el-slider
          v-model="difficultyRange"
          range
          :min="1"
          :max="5"
          :step="1"
          :marks="difficultyMarks"
          @change="fetchExercises"
        ></el-slider>
      </div>
      
      <!-- 练习列表 -->
      <div class="exercises-container">
        <div v-if="recommendedExercises.length === 0" class="no-exercises">
          <i class="el-icon-warning-outline"></i>
          <p>暂无符合条件的推荐练习</p>
        </div>
        
        <el-card 
          v-for="exercise in recommendedExercises" 
          :key="exercise.id"
          class="exercise-card"
          :body-style="{ padding: '12px' }"
          shadow="hover"
        >
          <div class="exercise-header">
            <span class="exercise-title">{{ exercise.title }}</span>
            <div class="exercise-tags">
              <el-tag size="mini" :type="getDifficultyType(exercise.difficulty)">
                {{ getDifficultyText(exercise.difficulty) }}
              </el-tag>
              <el-tag size="mini" type="info" v-if="exercise.type">
                {{ exercise.type }}
              </el-tag>
            </div>
          </div>
          
          <div class="exercise-preview" v-html="exercise.preview"></div>
          
          <div class="exercise-footer">
            <div class="exercise-meta">
              <span v-if="exercise.recommendReason" class="recommend-reason">
                <i class="el-icon-info"></i>
                {{ exercise.recommendReason }}
              </span>
              <span v-if="exercise.matchScore" class="match-score">
                匹配度: {{ Math.round(exercise.matchScore * 100) }}%
              </span>
            </div>
            
            <div class="exercise-actions">
              <el-button 
                type="primary" 
                size="mini" 
                @click="selectExercise(exercise)"
              >练习</el-button>
              <el-button 
                type="text" 
                size="mini" 
                @click="saveExercise(exercise)"
              >收藏</el-button>
            </div>
          </div>
        </el-card>
      </div>
      
      <!-- 历史练习记录 -->
      <div class="section-header">
        <span>最近练习</span>
        <el-button type="text" size="mini" @click="showAllHistory">查看全部</el-button>
      </div>
      
      <div class="history-container">
        <el-timeline>
          <el-timeline-item
            v-for="(history, index) in exerciseHistory"
            :key="history.id"
            :timestamp="formatDate(history.date)"
            :type="getHistoryItemType(history)"
            :size="index === 0 ? 'large' : 'normal'"
            :hide-timestamp="false"
          >
            <span class="history-title">{{ history.title }}</span>
            <div class="history-meta">
              <span class="history-score">
                得分: {{ history.score }}/{{ history.totalScore }}
              </span>
              <el-button 
                type="text" 
                size="mini" 
                @click="reviewHistory(history)"
              >复习</el-button>
            </div>
          </el-timeline-item>
        </el-timeline>
      </div>
    </template>
  </div>
</template>

<script>
export default {
  name: 'ExerciseRecommendation',
  
  props: {
    // 当前问题
    currentProblem: {
      type: Object,
      default: null
    },
    
    // 用户上下文
    userContext: {
      type: Object,
      required: true
    }
  },
  
  data() {
    return {
      // 加载状态
      loading: false,
      
      // 推荐策略
      strategy: 'similar',
      
      // 难度范围
      difficultyRange: [1, 3],
      
      // 难度标记
      difficultyMarks: {
        1: '简单',
        2: '中等',
        3: '困难',
        4: '挑战',
        5: '极难'
      },
      
      // 推荐练习
      recommendedExercises: [],
      
      // 练习历史
      exerciseHistory: []
    };
  },
  
  watch: {
    // 监听当前问题变化，更新推荐
    currentProblem: {
      handler(newVal) {
        if (newVal) {
          this.fetchExercises();
          this.fetchExerciseHistory();
        } else {
          this.recommendedExercises = [];
          this.exerciseHistory = [];
        }
      },
      immediate: true
    }
  },
  
  methods: {
    // 获取推荐练习
    async fetchExercises() {
      if (!this.currentProblem) return;
      
      this.loading = true;
      
      try {
        // 实际应用中应调用API获取推荐
        // 这里使用模拟数据
        await this.simulateFetch();
      } catch (error) {
        console.error('获取推荐练习失败', error);
        this.$message.error('获取推荐练习失败');
      } finally {
        this.loading = false;
      }
    },
    
    // 获取练习历史
    async fetchExerciseHistory() {
      if (!this.currentProblem) return;
      
      try {
        // 实际应用中应调用API获取历史
        // 这里使用模拟数据
        await this.simulateHistoryFetch();
      } catch (error) {
        console.error('获取练习历史失败', error);
      }
    },
    
    // 模拟获取推荐数据
    async simulateFetch() {
      // 模拟API延迟
      await new Promise(resolve => setTimeout(resolve, 800));
      
      // 根据当前问题和策略生成模拟数据
      const exercises = [];
      
      if (this.strategy === 'similar') {
        // 相似题目推荐
        exercises.push({
          id: 'e1',
          title: '相似题目 - 一元二次方程应用',
          preview: '一个长方形的周长是20厘米，面积是24平方厘米，求这个长方形的长和宽。',
          difficulty: 2,
          type: '应用题',
          recommendReason: '与当前问题使用相同解题方法',
          matchScore: 0.92
        });
        
        exercises.push({
          id: 'e2',
          title: '相似题目 - 因式分解',
          preview: '分解因式：x² - 9',
          difficulty: 1,
          type: '计算题',
          recommendReason: '与当前问题涉及相同知识点',
          matchScore: 0.85
        });
      } else if (this.strategy === 'complementary') {
        // 互补知识推荐
        exercises.push({
          id: 'e3',
          title: '互补知识 - 判别式应用',
          preview: '讨论方程 ax² + bx + c = 0 的根与判别式的关系。',
          difficulty: 3,
          type: '讨论题',
          recommendReason: '补充您的弱点知识',
          matchScore: 0.78
        });
        
        exercises.push({
          id: 'e4',
          title: '互补知识 - 韦达定理',
          preview: '已知一元二次方程 x² + px + q = 0 的两根为 α 和 β，求 α² + β²。',
          difficulty: 3,
          type: '计算题',
          recommendReason: '拓展相关知识点',
          matchScore: 0.75
        });
      } else if (this.strategy === 'adaptive') {
        // 自适应推荐
        exercises.push({
          id: 'e5',
          title: '自适应 - 配方法练习',
          preview: '使用配方法解一元二次方程：x² - 6x + 8 = 0',
          difficulty: 2,
          type: '计算题',
          recommendReason: '针对您的学习进度定制',
          matchScore: 0.95
        });
        
        exercises.push({
          id: 'e6',
          title: '自适应 - 一元二次方程综合',
          preview: '解方程并验证：2x² - 7x + 3 = 0',
          difficulty: this.userContext.proficiency?.algebra > 0.7 ? 3 : 2,
          type: '综合题',
          recommendReason: '基于您的能力水平推荐',
          matchScore: 0.88
        });
      }
      
      // 根据难度范围过滤
      this.recommendedExercises = exercises.filter(
        ex => ex.difficulty >= this.difficultyRange[0] && ex.difficulty <= this.difficultyRange[1]
      );
    },
    
    // 模拟获取历史数据
    async simulateHistoryFetch() {
      // 模拟API延迟
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // 生成模拟历史数据
      this.exerciseHistory = [
        {
          id: 'h1',
          title: '一元二次方程综合练习',
          date: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000), // 2天前
          score: 85,
          totalScore: 100,
          status: 'completed'
        },
        {
          id: 'h2',
          title: '因式分解专项训练',
          date: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000), // 5天前
          score: 92,
          totalScore: 100,
          status: 'completed'
        },
        {
          id: 'h3',
          title: '二次函数图像分析',
          date: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), // 7天前
          score: 78,
          totalScore: 100,
          status: 'completed'
        }
      ];
    },
    
    // 选择练习
    selectExercise(exercise) {
      this.$emit('exercise-selected', exercise);
    },
    
    // 收藏练习
    saveExercise(exercise) {
      // 实际应用中应调用API保存收藏
      this.$message.success(`已收藏练习：${exercise.title}`);
    },
    
    // 复习历史练习
    reviewHistory(history) {
      // 实际应用中应跳转到练习详情页
      this.$message.info(`开始复习：${history.title}`);
    },
    
    // 查看全部历史
    showAllHistory() {
      // 实际应用中应跳转到历史页面
      this.$message.info('查看全部练习历史');
    },
    
    // 获取难度类型
    getDifficultyType(difficulty) {
      const typeMap = {
        1: 'success',
        2: '',
        3: 'warning',
        4: 'danger',
        5: 'danger'
      };
      return typeMap[difficulty] || '';
    },
    
    // 获取难度文本
    getDifficultyText(difficulty) {
      return this.difficultyMarks[difficulty] || '未知';
    },
    
    // 获取历史项目类型
    getHistoryItemType(history) {
      if (history.score / history.totalScore >= 0.9) return 'success';
      if (history.score / history.totalScore >= 0.6) return 'warning';
      return 'danger';
    },
    
    // 格式化日期
    formatDate(date) {
      if (!date) return '';
      
      const now = new Date();
      const diff = now - date;
      const day = 24 * 60 * 60 * 1000;
      
      if (diff < day) {
        return '今天';
      } else if (diff < 2 * day) {
        return '昨天';
      } else if (diff < 7 * day) {
        return `${Math.floor(diff / day)}天前`;
      } else {
        const d = new Date(date);
        const month = String(d.getMonth() + 1).padStart(2, '0');
        const day = String(d.getDate()).padStart(2, '0');
        return `${month}-${day}`;
      }
    }
  }
};
</script>

<style scoped>
.exercise-recommendation {
  height: 100%;
  padding: 15px;
  display: flex;
  flex-direction: column;
  overflow-y: hidden;
}

.module-header {
  display: flex;
  align-items: center;
  margin-bottom: 15px;
}

.module-header h3 {
  margin: 0;
  font-size: 18px;
  color: #303133;
  margin-right: 8px;
}

.module-header i {
  color: #909399;
  cursor: help;
}

.loading-state {
  padding: 10px;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 150px;
  color: #909399;
}

.empty-state i {
  font-size: 36px;
  margin-bottom: 10px;
}

.recommendation-strategy {
  display: flex;
  align-items: center;
  margin-bottom: 15px;
}

.recommendation-strategy span {
  margin-right: 10px;
  color: #606266;
}

.difficulty-filter {
  margin-bottom: 15px;
}

.difficulty-filter span {
  color: #606266;
}

.exercises-container {
  flex: 1;
  overflow-y: auto;
  margin-bottom: 15px;
}

.no-exercises {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100px;
  color: #909399;
}

.no-exercises i {
  font-size: 24px;
  margin-bottom: 8px;
}

.exercise-card {
  margin-bottom: 15px;
}

.exercise-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.exercise-title {
  font-weight: 500;
  color: #303133;
}

.exercise-tags {
  display: flex;
  gap: 5px;
}

.exercise-preview {
  margin-bottom: 10px;
  color: #606266;
  font-size: 14px;
  line-height: 1.5;
  max-height: 60px;
  overflow: hidden;
  text-overflow: ellipsis;
}

.exercise-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.exercise-meta {
  display: flex;
  flex-direction: column;
  font-size: 12px;
  color: #909399;
}

.recommend-reason {
  margin-bottom: 5px;
}

.recommend-reason i {
  margin-right: 5px;
}

.exercise-actions {
  display: flex;
  gap: 10px;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin: 15px 0 10px;
  padding-bottom: 5px;
  border-bottom: 1px solid #ebeef5;
  font-weight: 500;
  color: #303133;
}

.history-container {
  overflow-y: auto;
  max-height: 200px;
}

.history-title {
  font-weight: 500;
  color: #303133;
}

.history-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 5px;
  font-size: 12px;
  color: #909399;
}
</style>
