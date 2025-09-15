<template>
  <div class="login-container">
    <div class="login-card">
      <div class="login-header">
        <h2>K-12数学教育系统</h2>
        <p>请登录您的账户</p>
      </div>
      
      <el-form 
        :model="loginForm" 
        :rules="loginRules" 
        ref="loginForm"
        class="login-form"
        @submit.native.prevent="handleLogin"
      >
        <el-form-item prop="username">
          <el-input
            v-model="loginForm.username"
            placeholder="请输入用户名"
            prefix-icon="el-icon-user"
            size="large"
          />
        </el-form-item>
        
        <el-form-item prop="password">
          <el-input
            v-model="loginForm.password"
            type="password"
            placeholder="请输入密码"
            prefix-icon="el-icon-lock"
            size="large"
            @keyup.enter.native="handleLogin"
          />
        </el-form-item>
        
        <el-form-item>
          <el-button 
            type="primary" 
            size="large"
            :loading="loading"
            @click="handleLogin"
            class="login-button"
          >
            {{ loading ? '登录中...' : '登录' }}
          </el-button>
        </el-form-item>
      </el-form>

      <!-- 注册链接 -->
      <div class="register-link">
        <span>还没有账户？</span>
        <router-link to="/register">立即注册</router-link>
      </div>

      <div class="demo-accounts">
        <h4>演示账户</h4>
        <div class="account-list">
          <div class="account-item" @click="quickLogin('test_student_001', 'password')">
            <span class="role">学生</span>
            <span class="username">test_student_001</span>
          </div>
          <div class="account-item" @click="quickLogin('teacher001', 'password')">
            <span class="role">教师</span>
            <span class="username">teacher001</span>
          </div>
          <div class="account-item" @click="quickLogin('admin', 'password')">
            <span class="role">管理员</span>
            <span class="username">admin</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { mapActions } from 'vuex';

export default {
  name: 'LoginPage',
  data() {
    return {
      loading: false,
      loginForm: {
        username: '',
        password: ''
      },
      loginRules: {
        username: [
          { required: true, message: '请输入用户名', trigger: 'blur' }
        ],
        password: [
          { required: true, message: '请输入密码', trigger: 'blur' }
        ]
      }
    };
  },
  methods: {
    ...mapActions(['login']),
    
    async handleLogin() {
      try {
        await this.$refs.loginForm.validate();
        this.loading = true;
        
        await this.login(this.loginForm);
        
        this.$message.success('登录成功');
        
        // 跳转到作业页面
        this.$router.push('/homework');
        
      } catch (error) {
        console.error('登录失败:', error);
        this.$message.error(error.message || '登录失败，请检查用户名和密码');
      } finally {
        this.loading = false;
      }
    },
    
    quickLogin(username, password) {
      this.loginForm.username = username;
      this.loginForm.password = password;
      this.handleLogin();
    }
  }
};
</script>

<style scoped>
.login-container {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 20px;
}

.login-card {
  background: white;
  border-radius: 12px;
  box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
  padding: 40px;
  width: 100%;
  max-width: 400px;
}

.login-header {
  text-align: center;
  margin-bottom: 30px;
}

.login-header h2 {
  color: #333;
  margin-bottom: 8px;
  font-weight: 600;
}

.login-header p {
  color: #666;
  margin: 0;
}

.login-form {
  margin-bottom: 30px;
}

.login-button {
  width: 100%;
  height: 45px;
  font-size: 16px;
  font-weight: 500;
}

.demo-accounts {
  border-top: 1px solid #eee;
  padding-top: 20px;
}

.demo-accounts h4 {
  margin: 0 0 15px 0;
  color: #666;
  font-size: 14px;
  text-align: center;
}

.account-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.account-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  background: #f8f9fa;
  border-radius: 6px;
  cursor: pointer;
  transition: background-color 0.2s;
}

.account-item:hover {
  background: #e9ecef;
}

.role {
  font-size: 12px;
  color: #666;
  background: #e3f2fd;
  padding: 2px 8px;
  border-radius: 12px;
}

.username {
  font-size: 13px;
  color: #333;
  font-family: monospace;
}

.register-link {
  text-align: center;
  margin: 20px 0;
  color: #666;
  font-size: 14px;
}

.register-link a {
  color: #409EFF;
  text-decoration: none;
  margin-left: 5px;
  font-weight: 500;
}

.register-link a:hover {
  text-decoration: underline;
}

/* 响应式设计 */
@media (max-width: 480px) {
  .login-card {
    padding: 30px 20px;
  }
  
  .login-header h2 {
    font-size: 20px;
  }
}
</style>
