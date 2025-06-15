<template>
  <div class="homework-container">
    <div class="homework-header">
      <h2>我的作业</h2>
      <div class="homework-filters">
        <div class="filter-group">
          <label for="status-filter">状态:</label>
          <select id="status-filter" v-model="filters.status">
            <option value="">全部</option>
            <option value="assigned">待完成</option>
            <option value="in_progress">进行中</option>
            <option value="submitted">已提交</option>
            <option value="graded">已评分</option>
          </select>
        </div>
        
        <div class="filter-group">
          <label for="subject-filter">学科:</label>
          <select id="subject-filter" v-model="filters.subject">
            <option value="">全部</option>
            <option v-for="subject in subjects" :key="subject" :value="subject">
              {{ subject }}
            </option>
          </select>
        </div>
        
        <div class="filter-group">
          <label for="due-date-filter">截止日期:</label>
          <select id="due-date-filter" v-model="filters.dueDate">
            <option value="">全部</option>
            <option value="today">今天</option>
            <option value="thisWeek">本周</option>
            <option value="overdue">已逾期</option>
          </select>
        </div>
        
        <div class="filter-group">
          <label for="sort-by">排序:</label>
          <select id="sort-by" v-model="filters.sortBy">
            <option value="dueDate">截止日期</option>
            <option value="priority">优先级</option>
            <option value="subject">学科</option>
          </select>
          <button 
            class="sort-direction-btn" 
            @click="toggleSortDirection"
            :title="filters.sortDirection === 'asc' ? '升序' : '降序'"
          >
            <i :class="filters.sortDirection === 'asc' ? 'icon-sort-asc' : 'icon-sort-desc'"></i>
          </button>
        </div>
      </div>
    </div>
    
    <div v-if="loading" class="loading">
      <div class="loading-spinner"></div>
      <p>加载作业中...</p>
    </div>
    
    <div v-else-if="filteredAssignments.length === 0" class="empty-state">
      <p>没有找到符合条件的作业</p>
    </div>
    
    <div v-else class="homework-list">
      <div 
        v-for="assignment in filteredAssignments" 
        :key="assignment.id"
        class="homework-card"
        :class="{
          'urgent': isUrgent(assignment),
          'completed': assignment.status === 'submitted' || assignment.status === 'graded'
        }"
        @click="viewAssignment(assignment.id)"
      >
        <div class="homework-status-indicator" :class="assignment.status"></div>
        <h3>{{ assignment.title }}</h3>
        
        <div class="homework-meta">
          <span class="subject">{{ assignment.subject }}</span>
          <span class="due-date" :class="{ 'overdue': isOverdue(assignment) }">
            {{ formatDueDate(assignment.dueDate) }}
          </span>
        </div>
        
        <p class="homework-description">{{ assignment.description }}</p>
        
        <div class="homework-tags">
          <span class="homework-tag status">{{ getStatusText(assignment.status) }}</span>
          <span v-if="assignment.priority >= 3" class="homework-tag priority-high">高优先级</span>
          <span v-else-if="assignment.priority === 2" class="homework-tag priority-medium">中优先级</span>
          <span v-if="isOverdue(assignment)" class="homework-tag overdue">已逾期</span>
          <span v-else-if="isDueSoon(assignment)" class="homework-tag due-soon">即将截止</span>
        </div>
        
        <div class="homework-progress">
          <div class="progress-bar" :style="{ width: `${assignment.completionRate * 100}%` }"></div>
        </div>
        
        <div class="homework-completion">
          {{ Math.round(assignment.completionRate * 100) }}% 完成
        </div>
        
        <div class="homework-actions">
          <button 
            v-if="assignment.status === 'assigned'"
            class="btn btn-primary"
            @click.stop="startAssignment(assignment.id)"
          >
            开始
          </button>
          
          <button 
            v-else-if="assignment.status === 'in_progress'"
            class="btn btn-primary"
            @click.stop="continueAssignment(assignment.id)"
          >
            继续
          </button>
          
          <button 
            v-else-if="assignment.status === 'submitted' && !assignment.grade"
            class="btn btn-outline"
            disabled
          >
            等待评分
          </button>
          
          <button 
            v-else-if="assignment.status === 'graded' || assignment.grade"
            class="btn btn-secondary"
            @click.stop="viewGrade(assignment.id)"
          >
            查看评分
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'HomeworkList',
  
  props: {
    assignmentService: {
      type: Object,
      required: true
    }
  },
  
  data() {
    return {
      loading: true,
      assignments: [],
      subjects: [],
      filters: {
        status: '',
        subject: '',
        dueDate: '',
        sortBy: 'dueDate',
        sortDirection: 'asc'
      }
    };
  },
  
  computed: {
    filteredAssignments() {
      return this.assignmentService.getAssignments(this.filters);
    }
  },
  
  async created() {
    await this.loadAssignments();
  },
  
  methods: {
    async loadAssignments() {
      this.loading = true;
      try {
        await this.assignmentService.loadAssignments();
        this.assignments = this.assignmentService.assignments;
        
        // 提取所有学科
        this.subjects = [...new Set(this.assignments.map(a => a.subject))];
      } catch (error) {
        console.error('Failed to load assignments:', error);
        this.$emit('error', '加载作业失败，请稍后再试');
      } finally {
        this.loading = false;
      }
    },
    
    toggleSortDirection() {
      this.filters.sortDirection = this.filters.sortDirection === 'asc' ? 'desc' : 'asc';
    },
    
    formatDueDate(dateString) {
      const dueDate = new Date(dateString);
      const now = new Date();
      
      // 如果是今天
      if (dueDate.toDateString() === now.toDateString()) {
        return `今天 ${dueDate.getHours()}:${dueDate.getMinutes().toString().padStart(2, '0')}`;
      }
      
      // 如果是明天
      const tomorrow = new Date(now);
      tomorrow.setDate(now.getDate() + 1);
      if (dueDate.toDateString() === tomorrow.toDateString()) {
        return `明天 ${dueDate.getHours()}:${dueDate.getMinutes().toString().padStart(2, '0')}`;
      }
      
      // 如果是本周
      const weekStart = new Date(now);
      weekStart.setDate(now.getDate() - now.getDay());
      const weekEnd = new Date(weekStart);
      weekEnd.setDate(weekStart.getDate() + 6);
      
      if (dueDate >= weekStart && dueDate <= weekEnd) {
        const days = ['周日', '周一', '周二', '周三', '周四', '周五', '周六'];
        return `${days[dueDate.getDay()]} ${dueDate.getHours()}:${dueDate.getMinutes().toString().padStart(2, '0')}`;
      }
      
      // 其他情况
      return `${dueDate.getMonth() + 1}月${dueDate.getDate()}日`;
    },
    
    isOverdue(assignment) {
      const dueDate = new Date(assignment.dueDate);
      const now = new Date();
      return dueDate < now && assignment.status !== 'submitted' && assignment.status !== 'graded';
    },
    
    isDueSoon(assignment) {
      const dueDate = new Date(assignment.dueDate);
      const now = new Date();
      const threeDaysLater = new Date(now);
      threeDaysLater.setDate(now.getDate() + 3);
      
      return dueDate > now && dueDate <= threeDaysLater;
    },
    
    isUrgent(assignment) {
      return this.isOverdue(assignment) || 
             (this.isDueSoon(assignment) && assignment.priority >= 2);
    },
    
    getStatusText(status) {
      const statusMap = {
        'assigned': '待完成',
        'in_progress': '进行中',
        'submitted': '已提交',
        'graded': '已评分'
      };
      
      return statusMap[status] || status;
    },
    
    viewAssignment(assignmentId) {
      this.$emit('view-assignment', assignmentId);
    },
    
    startAssignment(assignmentId) {
      this.$emit('start-assignment', assignmentId);
    },
    
    continueAssignment(assignmentId) {
      this.$emit('continue-assignment', assignmentId);
    },
    
    viewGrade(assignmentId) {
      this.$emit('view-grade', assignmentId);
    }
  }
};
</script>

<style scoped>
.homework-container {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.homework-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 15px;
}

.homework-header h2 {
  margin: 0;
}

.homework-filters {
  display: flex;
  gap: 15px;
  flex-wrap: wrap;
}

.filter-group {
  display: flex;
  align-items: center;
  gap: 5px;
}

.filter-group select {
  padding: 8px;
  border-radius: 4px;
  border: 1px solid var(--border-color);
}

.sort-direction-btn {
  background: none;
  border: none;
  cursor: pointer;
  padding: 5px;
}

.homework-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
}

.homework-card {
  background-color: white;
  border-radius: 8px;
  box-shadow: var(--shadow);
  padding: 15px;
  transition: var(--transition);
  border-left: 4px solid var(--primary-color);
  position: relative;
  cursor: pointer;
}

.homework-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

.homework-card.urgent {
  border-left-color: #dc3545;
}

.homework-card.completed {
  border-left-color: var(--secondary-color);
}

.homework-card h3 {
  margin-top: 0;
  margin-bottom: 10px;
  color: var(--text-color);
}

.homework-meta {
  display: flex;
  justify-content: space-between;
  color: var(--light-text);
  font-size: 14px;
  margin-bottom: 15px;
}

.due-date.overdue {
  color: #dc3545;
  font-weight: bold;
}

.homework-description {
  margin-bottom: 15px;
  font-size: 14px;
  color: var(--text-color);
  line-height: 1.5;
}

.homework-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin-bottom: 15px;
}

.homework-tag {
  background-color: var(--background-dark);
  color: var(--light-text);
  font-size: 12px;
  padding: 4px 8px;
  border-radius: 20px;
}

.homework-tag.status {
  background-color: #e9ecef;
}

.homework-tag.priority-high {
  background-color: #dc3545;
  color: white;
}

.homework-tag.priority-medium {
  background-color: #fd7e14;
  color: white;
}

.homework-tag.overdue {
  background-color: #dc3545;
  color: white;
}

.homework-tag.due-soon {
  background-color: #ffc107;
  color: #212529;
}

.homework-progress {
  height: 6px;
  background-color: var(--background-dark);
  border-radius: 3px;
  margin-bottom: 5px;
  overflow: hidden;
}

.progress-bar {
  height: 100%;
  background-color: var(--primary-color);
  border-radius: 3px;
}

.homework-completion {
  font-size: 12px;
  color: var(--light-text);
  margin-bottom: 15px;
  text-align: right;
}

.homework-actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}

.btn {
  padding: 8px 15px;
  border-radius: 4px;
  border: none;
  cursor: pointer;
  font-weight: 500;
  transition: var(--transition);
  font-size: 14px;
}

.btn-primary {
  background-color: var(--primary-color);
  color: white;
}

.btn-primary:hover {
  background-color: #3a80d2;
}

.btn-secondary {
  background-color: var(--secondary-color);
  color: white;
}

.btn-secondary:hover {
  background-color: #4ca84c;
}

.btn-outline {
  background-color: transparent;
  border: 1px solid var(--border-color);
  color: var(--text-color);
}

.btn-outline:hover {
  background-color: var(--background-light);
}

.btn[disabled] {
  opacity: 0.6;
  cursor: not-allowed;
}

.empty-state {
  text-align: center;
  padding: 40px;
  background-color: var(--background-light);
  border-radius: 8px;
  color: var(--light-text);
}

.loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px;
}

.loading-spinner {
  border: 4px solid var(--background-dark);
  border-top: 4px solid var(--primary-color);
  border-radius: 50%;
  width: 30px;
  height: 30px;
  animation: spin 1s linear infinite;
  margin-bottom: 15px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

@media (max-width: 768px) {
  .homework-header {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .homework-filters {
    flex-direction: column;
    width: 100%;
  }
  
  .filter-group {
    width: 100%;
  }
  
  .filter-group select {
    flex-grow: 1;
  }
  
  .homework-list {
    grid-template-columns: 1fr;
  }
}
</style> 