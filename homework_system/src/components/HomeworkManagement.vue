<!--
  文件名: HomeworkManagement.vue
  描述: 作业管理模块组件，负责作业的接收、存储、状态跟踪和提交
  作者: Claude
  创建日期: 2025-06-15
-->

<template>
  <div class="homework-management">
    <div class="header">
      <h3>我的作业</h3>
      <el-button 
        type="text" 
        icon="el-icon-refresh" 
        @click="refreshHomeworks"
        :loading="loading"
      >刷新</el-button>
    </div>
    
    <!-- 作业过滤器 -->
    <div class="filter-container">
      <el-select v-model="statusFilter" placeholder="状态" size="small" @change="applyFilters">
        <el-option label="全部" value="all"></el-option>
        <el-option label="未完成" value="incomplete"></el-option>
        <el-option label="已提交" value="submitted"></el-option>
        <el-option label="已批改" value="graded"></el-option>
      </el-select>
      
      <el-select v-model="sortBy" placeholder="排序" size="small" @change="applyFilters">
        <el-option label="截止日期" value="deadline"></el-option>
        <el-option label="最近活动" value="activity"></el-option>
        <el-option label="难度" value="difficulty"></el-option>
      </el-select>
    </div>
    
    <!-- 作业列表 -->
    <div class="homework-list" v-loading="loading">
      <template v-if="filteredHomeworks.length > 0">
        <div 
          v-for="homework in filteredHomeworks" 
          :key="homework.id"
          class="homework-item"
          :class="{ 'active': activeHomeworkId === homework.id }"
          @click="selectHomework(homework.id)"
        >
          <div class="homework-status">
            <el-tag :type="getStatusType(homework.status)" size="mini">
              {{ getStatusText(homework.status) }}
            </el-tag>
          </div>
          <div class="homework-info">
            <h4 class="homework-title">{{ homework.title }}</h4>
            <div class="homework-meta">
              <span><i class="el-icon-date"></i> {{ formatDate(homework.deadline) }}</span>
              <span><i class="el-icon-medal"></i> {{ getDifficultyText(homework.difficulty) }}</span>
            </div>
          </div>
          <div class="homework-progress">
            <el-progress 
              type="circle" 
              :percentage="calculateProgress(homework)" 
              :width="36"
              :status="getProgressStatus(homework)"
            ></el-progress>
          </div>
        </div>
      </template>
      
      <div v-else-if="!loading" class="empty-state">
        <i class="el-icon-document"></i>
        <p>{{ getEmptyStateText() }}</p>
      </div>
    </div>
    
    <!-- 作业统计 -->
    <div class="homework-stats">
      <div class="stat-item">
        <span class="stat-label">待完成</span>
        <span class="stat-value">{{ getIncompleteCount() }}</span>
      </div>
      <div class="stat-item">
        <span class="stat-label">已提交</span>
        <span class="stat-value">{{ getSubmittedCount() }}</span>
      </div>
      <div class="stat-item">
        <span class="stat-label">已批改</span>
        <span class="stat-value">{{ getGradedCount() }}</span>
      </div>
    </div>
    
    <!-- 提醒设置 -->
    <div class="reminder-settings">
      <el-collapse>
        <el-collapse-item title="提醒设置" name="reminders">
          <div class="reminder-options">
            <el-checkbox v-model="reminders.deadline">截止前提醒</el-checkbox>
            <el-select 
              v-model="reminders.deadlineTime" 
              placeholder="提前时间" 
              size="small"
              :disabled="!reminders.deadline"
            >
              <el-option label="1小时" value="1h"></el-option>
              <el-option label="3小时" value="3h"></el-option>
              <el-option label="6小时" value="6h"></el-option>
              <el-option label="12小时" value="12h"></el-option>
              <el-option label="1天" value="1d"></el-option>
            </el-select>
          </div>
          <div class="reminder-options">
            <el-checkbox v-model="reminders.daily">每日提醒</el-checkbox>
            <el-time-picker
              v-model="reminders.dailyTime"
              placeholder="提醒时间"
              size="small"
              :disabled="!reminders.daily"
              format="HH:mm"
              value-format="HH:mm"
            ></el-time-picker>
          </div>
          <el-button 
            type="primary" 
            size="small" 
            @click="saveReminderSettings"
            style="margin-top: 10px"
          >保存设置</el-button>
        </el-collapse-item>
      </el-collapse>
    </div>
  </div>
</template>

<script>
import { formatDate } from '../utils/dateFormat';

export default {
  name: 'HomeworkManagement',
  
  props: {
    // 作业列表
    homeworks: {
      type: Array,
      required: true
    },
    
    // 当前激活的作业ID
    activeHomeworkId: {
      type: String,
      default: ''
    }
  },
  
  data() {
    return {
      // 加载状态
      loading: false,
      
      // 过滤器
      statusFilter: 'all',
      sortBy: 'deadline',
      
      // 提醒设置
      reminders: {
        deadline: true,
        deadlineTime: '12h',
        daily: false,
        dailyTime: '20:00'
      }
    };
  },
  
  computed: {
    // 根据过滤条件筛选并排序作业
    filteredHomeworks() {
      // 先筛选
      let result = [...this.homeworks];
      
      if (this.statusFilter !== 'all') {
        if (this.statusFilter === 'incomplete') {
          // 未完成包括未开始和进行中
          result = result.filter(hw => 
            hw.status === 'not_started' || hw.status === 'in_progress'
          );
        } else if (this.statusFilter === 'submitted') {
          // 已提交包括已提交和已批改
          result = result.filter(hw => 
            hw.status === 'submitted' || hw.status === 'graded'
          );
        } else if (this.statusFilter === 'graded') {
          // 只包括已批改
          result = result.filter(hw => hw.status === 'graded');
        }
      }
      
      // 再排序
      if (this.sortBy === 'deadline') {
        // 按截止日期排序（从近到远）
        result.sort((a, b) => new Date(a.deadline) - new Date(b.deadline));
      } else if (this.sortBy === 'activity') {
        // 按最近活动排序（从新到旧）
        result.sort((a, b) => new Date(b.lastActivity) - new Date(a.lastActivity));
      } else if (this.sortBy === 'difficulty') {
        // 按难度排序（从低到高）
        result.sort((a, b) => a.difficulty - b.difficulty);
      }
      
      return result;
    }
  },
  
  methods: {
    // 选择作业
    selectHomework(homeworkId) {
      this.$emit('select-homework', homeworkId);
    },
    
    // 刷新作业列表
    refreshHomeworks() {
      this.loading = true;
      this.$emit('refresh-homeworks');
      
      // 模拟加载结束
      setTimeout(() => {
        this.loading = false;
      }, 800);
    },
    
    // 应用过滤条件
    applyFilters() {
      // 过滤和排序是通过计算属性实现的，这里不需要额外操作
    },
    
    // 计算作业完成进度
    calculateProgress(homework) {
      if (!homework.problems || homework.problems.length === 0) {
        return 0;
      }
      
      if (homework.status === 'submitted' || homework.status === 'graded') {
        return 100;
      }
      
      // 根据已回答的问题计算进度
      const totalProblems = homework.problems.length;
      const answeredProblems = homework.problems.filter(
        problem => homework.savedAnswers && homework.savedAnswers[problem.id]
      ).length;
      
      return Math.round((answeredProblems / totalProblems) * 100);
    },
    
    // 获取进度状态
    getProgressStatus(homework) {
      const progress = this.calculateProgress(homework);
      
      if (progress === 0) return 'exception';
      if (progress < 100) return '';
      return 'success';
    },
    
    // 获取状态标签类型
    getStatusType(status) {
      const typeMap = {
        'not_started': 'info',
        'in_progress': 'warning',
        'submitted': 'primary',
        'graded': 'success'
      };
      return typeMap[status] || 'info';
    },
    
    // 获取状态文本
    getStatusText(status) {
      const textMap = {
        'not_started': '未开始',
        'in_progress': '进行中',
        'submitted': '已提交',
        'graded': '已批改'
      };
      return textMap[status] || status;
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
    
    // 获取未完成作业数量
    getIncompleteCount() {
      return this.homeworks.filter(
        hw => hw.status === 'not_started' || hw.status === 'in_progress'
      ).length;
    },
    
    // 获取已提交作业数量
    getSubmittedCount() {
      return this.homeworks.filter(
        hw => hw.status === 'submitted'
      ).length;
    },
    
    // 获取已批改作业数量
    getGradedCount() {
      return this.homeworks.filter(
        hw => hw.status === 'graded'
      ).length;
    },
    
    // 获取空状态文本
    getEmptyStateText() {
      if (this.statusFilter === 'all') {
        return '暂无作业';
      } else if (this.statusFilter === 'incomplete') {
        return '暂无未完成作业';
      } else if (this.statusFilter === 'submitted') {
        return '暂无已提交作业';
      } else if (this.statusFilter === 'graded') {
        return '暂无已批改作业';
      }
      return '暂无作业';
    },
    
    // 保存提醒设置
    saveReminderSettings() {
      // 实际应用中应该调用API保存设置
      this.$message.success('提醒设置已保存');
      
      // 如果启用了提醒，可以在本地设置提醒
      if (this.reminders.deadline) {
        // 设置截止日期提醒
        console.log('设置截止日期提醒', this.reminders.deadlineTime);
      }
      
      if (this.reminders.daily) {
        // 设置每日提醒
        console.log('设置每日提醒', this.reminders.dailyTime);
      }
    },
    
    // 格式化日期
    formatDate(date) {
      return formatDate(date, 'MM-DD HH:mm');
    }
  }
};
</script>

<style scoped>
.homework-management {
  display: flex;
  flex-direction: column;
  height: 100%;
  padding: 15px;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.header h3 {
  margin: 0;
  font-size: 18px;
  color: #303133;
}

.filter-container {
  display: flex;
  gap: 10px;
  margin-bottom: 15px;
}

.homework-list {
  flex: 1;
  overflow-y: auto;
  margin-bottom: 15px;
}

.homework-item {
  display: flex;
  padding: 12px;
  margin-bottom: 10px;
  background-color: #fff;
  border-radius: 4px;
  border-left: 3px solid #e6e6e6;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
  cursor: pointer;
  transition: all 0.3s;
}

.homework-item:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.homework-item.active {
  border-left-color: #409EFF;
  background-color: #f0f9ff;
}

.homework-status {
  width: 60px;
  display: flex;
  align-items: flex-start;
  justify-content: center;
}

.homework-info {
  flex: 1;
  padding-right: 10px;
}

.homework-title {
  margin: 0 0 5px 0;
  font-size: 15px;
  font-weight: 500;
  color: #303133;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.homework-meta {
  display: flex;
  gap: 10px;
  font-size: 12px;
  color: #909399;
}

.homework-progress {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
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

.homework-stats {
  display: flex;
  border-top: 1px solid #e6e6e6;
  border-bottom: 1px solid #e6e6e6;
  padding: 10px 0;
  margin-bottom: 15px;
}

.stat-item {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.stat-label {
  font-size: 12px;
  color: #909399;
  margin-bottom: 5px;
}

.stat-value {
  font-size: 18px;
  font-weight: bold;
  color: #303133;
}

.reminder-settings {
  margin-top: auto;
}

.reminder-options {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
}
</style>

