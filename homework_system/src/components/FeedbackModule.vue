<!--
  文件名: FeedbackModule.vue
  描述: 反馈模块组件，提供实时评分和针对性反馈
  创建日期: 2023-06-15
-->

<template>
    <div class="feedback-module">
      <div class="module-header">
        <h3>作业反馈</h3>
        <el-tooltip content="基于您的作业表现提供的评分和建议" placement="top">
          <i class="el-icon-question"></i>
        </el-tooltip>
      </div>
      
      <!-- 总体评分 -->
      <div class="overall-score">
        <el-row :gutter="20">
          <el-col :span="8">
            <div class="score-card">
              <div class="score-value">{{ feedback.overallScore }}</div>
              <div class="score-label">总分</div>
              <div class="score-total">满分 {{ feedback.totalScore }}</div>
            </div>
          </el-col>
          
          <el-col :span="16">
            <div class="score-details">
              <div class="score-item">
                <span class="score-item-label">准确性</span>
                <el-progress 
                  :percentage="Math.round(feedback.accuracyScore * 100)" 
                  :color="getScoreColor(feedback.accuracyScore)"
                ></el-progress>
              </div>
              <div class="score-item">
                <span class="score-item-label">完整性</span>
                <el-progress 
                  :percentage="Math.round(feedback.completionScore * 100)" 
                  :color="getScoreColor(feedback.completionScore)"
                ></el-progress>
              </div>
              <div class="score-item">
                <span class="score-item-label">方法应用</span>
                <el-progress 
                  :percentage="Math.round(feedback.methodScore * 100)" 
                  :color="getScoreColor(feedback.methodScore)"
                ></el-progress>
              </div>
            </div>
          </el-col>
        </el-row>
      </div>
      
      <!-- 总体评语 -->
      <div class="overall-comment">
        <el-alert
          type="info"
          :closable="false"
          show-icon
        >
          <div class="comment-content">{{ feedback.overallComment }}</div>
        </el-alert>
      </div>
      
      <!-- 问题反馈 -->
      <div class="problems-feedback">
        <div class="section-header">
          <span>题目反馈</span>
        </div>
        
        <el-collapse>
          <el-collapse-item 
            v-for="problem in feedback.problems" 
            :key="problem.id"
            :title="getProblemTitle(problem)"
            :name="problem.id"
          >
            <div class="problem-feedback">
              <div class="problem-statement">{{ problem.statement }}</div>
              
              <div class="answer-comparison">
                <div class="student-answer">
                  <div class="answer-label">您的答案:</div>
                  <div class="answer-content" v-html="formatAnswer(problem.studentAnswer)"></div>
                </div>
                
                <div v-if="!problem.isCorrect" class="correct-answer">
                  <div class="answer-label">参考答案:</div>
                  <div class="answer-content" v-html="formatAnswer(problem.correctAnswer)"></div>
                </div>
              </div>
              
              <div class="problem-comments">
                <div 
                  v-for="(comment, index) in problem.comments" 
                  :key="index"
                  class="comment-item"
                >
                  <el-alert
                    :type="comment.type"
                    :closable="false"
                    show-icon
                  >
                    {{ comment.text }}
                  </el-alert>
                </div>
              </div>
            </div>
          </el-collapse-item>
        </el-collapse>
      </div>
      
      <!-- 学习建议 -->
      <div v-if="feedback.suggestions && feedback.suggestions.length > 0" class="learning-suggestions">
        <div class="section-header">
          <span>学习建议</span>
        </div>
        
        <el-card shadow="never">
          <ul class="suggestion-list">
            <li v-for="(suggestion, index) in feedback.suggestions" :key="index">
              {{ suggestion }}
            </li>
          </ul>
        </el-card>
      </div>
    </div>
  </template>
  
  <script>
  export default {
    name: 'FeedbackModule',
    
    props: {
      // 反馈数据
      feedback: {
        type: Object,
        required: true
      }
    },
    
    methods: {
      // 获取分数颜色
      getScoreColor(score) {
        if (score >= 0.8) return '#67C23A';
        if (score >= 0.6) return '#E6A23C';
        return '#F56C6C';
      },
      
      // 获取问题标题
      getProblemTitle(problem) {
        const scoreText = `${problem.score}/${problem.totalScore}分`;
        const statusIcon = problem.isCorrect 
          ? '<i class="el-icon-check" style="color: #67C23A;"></i>' 
          : '<i class="el-icon-close" style="color: #F56C6C;"></i>';
        
        return `${statusIcon} ${problem.title} (${scoreText})`;
      },
      
      // 格式化答案（保留换行）
      formatAnswer(answer) {
        if (!answer) return '未作答';
        return answer.replace(/\n/g, '<br>');
      }
    }
  };
  </script>
  
  <style scoped>
  .feedback-module {
    padding: 15px;
  }
  
  .module-header {
    display: flex;
    align-items: center;
    margin-bottom: 20px;
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
  
  .overall-score {
    margin-bottom: 20px;
  }
  
  .score-card {
    background-color: #f5f7fa;
    border-radius: 4px;
    padding: 15px;
    text-align: center;
    height: 100%;
  }
  
  .score-value {
    font-size: 32px;
    font-weight: bold;
    color: #409EFF;
    margin-bottom: 5px;
  }
  
  .score-label {
    font-size: 14px;
    color: #606266;
    margin-bottom: 5px;
  }
  
  .score-total {
    font-size: 12px;
    color: #909399;
  }
  
  .score-details {
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-around;
  }
  
  .score-item {
    margin-bottom: 10px;
  }
  
  .score-item-label {
    display: inline-block;
    width: 70px;
    font-size: 14px;
    color: #606266;
  }
  
  .overall-comment {
    margin-bottom: 20px;
  }
  
  .comment-content {
    line-height: 1.6;
  }
  
  .section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 20px 0 10px;
    padding-bottom: 5px;
    border-bottom: 1px solid #ebeef5;
    font-weight: 500;
    color: #303133;
  }
  
  .problem-feedback {
    padding: 10px 0;
  }
  
  .problem-statement {
    margin-bottom: 15px;
    color: #606266;
    line-height: 1.6;
  }
  
  .answer-comparison {
    margin-bottom: 15px;
  }
  
  .student-answer, .correct-answer {
    margin-bottom: 10px;
  }
  
  .answer-label {
    font-weight: 500;
    margin-bottom: 5px;
    color: #303133;
  }
  
  .answer-content {
    background-color: #f5f7fa;
    border-radius: 4px;
    padding: 10px;
    color: #606266;
    line-height: 1.6;
    white-space: pre-wrap;
  }
  
  .problem-comments {
    margin-top: 15px;
  }
  
  .comment-item {
    margin-bottom: 10px;
  }
  
  .suggestion-list {
    padding-left: 20px;
    margin: 0;
    line-height: 1.8;
    color: #606266;
  }
  
  .suggestion-list li {
    margin-bottom: 8px;
  }
  
  .suggestion-list li:last-child {
    margin-bottom: 0;
  }
  </style>