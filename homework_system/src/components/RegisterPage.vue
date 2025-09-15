<template>
  <div class="register-container">
    <div class="register-card">
      <div class="register-header">
        <h2>用户注册</h2>
        <p>创建您的K-12数学教育账户</p>
      </div>

      <el-form
        ref="registerForm"
        :model="registerForm"
        :rules="registerRules"
        label-width="100px"
        class="register-form"
      >
        <!-- 基本信息 -->
        <div class="form-section">
          <h3>基本信息</h3>
          
          <el-form-item label="用户角色" prop="role">
            <el-select v-model="registerForm.role" placeholder="请选择用户角色" @change="onRoleChange">
              <el-option label="学生" value="student"></el-option>
              <el-option label="教师" value="teacher"></el-option>
              <el-option label="家长" value="parent"></el-option>
            </el-select>
          </el-form-item>

          <el-form-item label="用户名" prop="username">
            <el-input v-model="registerForm.username" placeholder="请输入用户名"></el-input>
          </el-form-item>

          <el-form-item label="邮箱" prop="email">
            <el-input v-model="registerForm.email" placeholder="请输入邮箱地址"></el-input>
          </el-form-item>

          <el-form-item label="真实姓名" prop="real_name">
            <el-input v-model="registerForm.real_name" placeholder="请输入真实姓名"></el-input>
          </el-form-item>

          <el-form-item label="密码" prop="password">
            <el-input
              v-model="registerForm.password"
              type="password"
              placeholder="请输入密码（至少8位，包含字母和数字）"
              show-password
            ></el-input>
          </el-form-item>

          <el-form-item label="确认密码" prop="confirmPassword">
            <el-input
              v-model="registerForm.confirmPassword"
              type="password"
              placeholder="请再次输入密码"
              show-password
            ></el-input>
          </el-form-item>
        </div>

        <!-- 角色特定信息 -->
        <div class="form-section" v-if="registerForm.role">
          <h3>{{ getRoleSectionTitle() }}</h3>

          <!-- 学生特定字段 -->
          <template v-if="registerForm.role === 'student'">
            <el-form-item label="年级" prop="grade">
              <el-select v-model="registerForm.grade" placeholder="请选择年级">
                <el-option v-for="grade in grades" :key="grade" :label="`${grade}年级`" :value="grade"></el-option>
              </el-select>
            </el-form-item>

            <el-form-item label="学校" prop="school">
              <el-input v-model="registerForm.school" placeholder="请输入学校名称"></el-input>
            </el-form-item>

            <el-form-item label="班级" prop="class_name">
              <el-input v-model="registerForm.class_name" placeholder="请输入班级（如：1班）"></el-input>
            </el-form-item>

            <el-form-item label="学号" prop="student_id">
              <el-input v-model="registerForm.student_id" placeholder="请输入学号"></el-input>
            </el-form-item>
          </template>

          <!-- 教师特定字段 -->
          <template v-if="registerForm.role === 'teacher'">
            <el-form-item label="学校" prop="school">
              <el-input v-model="registerForm.school" placeholder="请输入学校名称"></el-input>
            </el-form-item>

            <el-form-item label="教授年级" prop="teaching_grades">
              <el-select v-model="registerForm.teaching_grades" multiple placeholder="请选择教授年级">
                <el-option v-for="grade in grades" :key="grade" :label="`${grade}年级`" :value="grade"></el-option>
              </el-select>
            </el-form-item>

            <el-form-item label="联系电话" prop="phone">
              <el-input v-model="registerForm.phone" placeholder="请输入联系电话"></el-input>
            </el-form-item>
          </template>

          <!-- 家长特定字段 -->
          <template v-if="registerForm.role === 'parent'">
            <el-form-item label="联系电话" prop="phone">
              <el-input v-model="registerForm.phone" placeholder="请输入联系电话"></el-input>
            </el-form-item>

            <el-form-item label="孩子姓名" prop="child_name">
              <el-input v-model="registerForm.child_name" placeholder="请输入孩子姓名"></el-input>
            </el-form-item>

            <el-form-item label="孩子学校" prop="child_school">
              <el-input v-model="registerForm.child_school" placeholder="请输入孩子学校"></el-input>
            </el-form-item>

            <el-form-item label="孩子年级" prop="child_grade">
              <el-select v-model="registerForm.child_grade" placeholder="请选择孩子年级">
                <el-option v-for="grade in grades" :key="grade" :label="`${grade}年级`" :value="grade"></el-option>
              </el-select>
            </el-form-item>
          </template>
        </div>

        <!-- 提交按钮 -->
        <el-form-item class="submit-section">
          <el-button type="primary" @click="submitForm" :loading="loading" size="large">
            {{ loading ? '注册中...' : '立即注册' }}
          </el-button>
          <el-button @click="resetForm" size="large">重置</el-button>
        </el-form-item>

        <!-- 登录链接 -->
        <div class="login-link">
          <span>已有账户？</span>
          <router-link to="/login">立即登录</router-link>
        </div>
      </el-form>
    </div>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'RegisterPage',
  
  data() {
    // 密码验证规则
    const validatePassword = (rule, value, callback) => {
      if (!value) {
        callback(new Error('请输入密码'))
      } else if (value.length < 8) {
        callback(new Error('密码长度至少8位'))
      } else if (!/(?=.*[a-zA-Z])(?=.*\d)/.test(value)) {
        callback(new Error('密码必须包含字母和数字'))
      } else {
        callback()
      }
    }

    // 确认密码验证
    const validateConfirmPassword = (rule, value, callback) => {
      if (!value) {
        callback(new Error('请确认密码'))
      } else if (value !== this.registerForm.password) {
        callback(new Error('两次输入的密码不一致'))
      } else {
        callback()
      }
    }

    return {
      loading: false,
      grades: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
      
      registerForm: {
        role: '',
        username: '',
        email: '',
        real_name: '',
        password: '',
        confirmPassword: '',
        // 学生字段
        grade: null,
        school: '',
        class_name: '',
        student_id: '',
        // 教师字段
        teaching_grades: [],
        phone: '',
        // 家长字段
        child_name: '',
        child_school: '',
        child_grade: null
      },

      registerRules: {
        role: [
          { required: true, message: '请选择用户角色', trigger: 'change' }
        ],
        username: [
          { required: true, message: '请输入用户名', trigger: 'blur' },
          { min: 3, max: 20, message: '用户名长度在3到20个字符', trigger: 'blur' }
        ],
        email: [
          { required: true, message: '请输入邮箱地址', trigger: 'blur' },
          { type: 'email', message: '请输入正确的邮箱地址', trigger: 'blur' }
        ],
        real_name: [
          { required: true, message: '请输入真实姓名', trigger: 'blur' }
        ],
        password: [
          { required: true, validator: validatePassword, trigger: 'blur' }
        ],
        confirmPassword: [
          { required: true, validator: validateConfirmPassword, trigger: 'blur' }
        ],
        // 学生必填字段
        grade: [
          { required: true, message: '请选择年级', trigger: 'change' }
        ],
        school: [
          { required: true, message: '请输入学校名称', trigger: 'blur' }
        ],
        class_name: [
          { required: true, message: '请输入班级', trigger: 'blur' }
        ],
        student_id: [
          { required: true, message: '请输入学号', trigger: 'blur' }
        ],
        // 教师必填字段
        teaching_grades: [
          { required: true, message: '请选择教授年级', trigger: 'change' }
        ],
        phone: [
          { required: true, message: '请输入联系电话', trigger: 'blur' },
          { pattern: /^1[3-9]\d{9}$/, message: '请输入正确的手机号码', trigger: 'blur' }
        ],
        // 家长必填字段
        child_name: [
          { required: true, message: '请输入孩子姓名', trigger: 'blur' }
        ],
        child_school: [
          { required: true, message: '请输入孩子学校', trigger: 'blur' }
        ],
        child_grade: [
          { required: true, message: '请选择孩子年级', trigger: 'change' }
        ]
      }
    }
  },

  methods: {
    // 角色改变时重置相关字段
    onRoleChange() {
      // 清空角色特定字段
      this.registerForm.grade = null
      this.registerForm.school = ''
      this.registerForm.class_name = ''
      this.registerForm.student_id = ''
      this.registerForm.teaching_grades = []
      this.registerForm.phone = ''
      this.registerForm.child_name = ''
      this.registerForm.child_school = ''
      this.registerForm.child_grade = null
      
      // 清除验证错误
      this.$nextTick(() => {
        this.$refs.registerForm.clearValidate()
      })
    },

    // 获取角色特定信息标题
    getRoleSectionTitle() {
      const titles = {
        student: '学生信息',
        teacher: '教师信息',
        parent: '家长信息'
      }
      return titles[this.registerForm.role] || '角色信息'
    },

    // 提交表单
    async submitForm() {
      try {
        // 表单验证
        const valid = await this.$refs.registerForm.validate()
        if (!valid) return

        this.loading = true

        // 准备提交数据
        const submitData = { ...this.registerForm }
        delete submitData.confirmPassword // 移除确认密码字段

        // 调用注册API
        const response = await axios.post('http://localhost:5000/api/auth/register', submitData)

        if (response.data.success) {
          this.$message.success('注册成功！请查收激活邮件')
          
          // 跳转到登录页面
          setTimeout(() => {
            this.$router.push('/login')
          }, 2000)
        } else {
          this.$message.error(response.data.message || '注册失败')
        }

      } catch (error) {
        console.error('注册失败:', error)
        
        if (error.response && error.response.data) {
          this.$message.error(error.response.data.message || '注册失败')
        } else {
          this.$message.error('网络错误，请稍后重试')
        }
      } finally {
        this.loading = false
      }
    },

    // 重置表单
    resetForm() {
      this.$refs.registerForm.resetFields()
      this.registerForm.teaching_grades = []
    }
  }
}
</script>

<style scoped>
.register-container {
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
}

.register-card {
  background: white;
  border-radius: 10px;
  box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
  padding: 40px;
  width: 100%;
  max-width: 600px;
  max-height: 90vh;
  overflow-y: auto;
}

.register-header {
  text-align: center;
  margin-bottom: 30px;
}

.register-header h2 {
  color: #333;
  margin-bottom: 10px;
  font-size: 28px;
}

.register-header p {
  color: #666;
  font-size: 16px;
}

.form-section {
  margin-bottom: 30px;
  padding-bottom: 20px;
  border-bottom: 1px solid #f0f0f0;
}

.form-section:last-child {
  border-bottom: none;
}

.form-section h3 {
  color: #409EFF;
  margin-bottom: 20px;
  font-size: 18px;
}

.register-form .el-form-item {
  margin-bottom: 20px;
}

.submit-section {
  text-align: center;
  margin-top: 30px;
}

.submit-section .el-button {
  width: 120px;
  margin: 0 10px;
}

.login-link {
  text-align: center;
  margin-top: 20px;
  color: #666;
}

.login-link a {
  color: #409EFF;
  text-decoration: none;
  margin-left: 5px;
}

.login-link a:hover {
  text-decoration: underline;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .register-card {
    padding: 20px;
    margin: 10px;
  }
  
  .register-header h2 {
    font-size: 24px;
  }
  
  .submit-section .el-button {
    width: 100px;
    margin: 5px;
  }
}
</style>
