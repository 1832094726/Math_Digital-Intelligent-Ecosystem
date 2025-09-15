import Vue from 'vue';
import VueRouter from 'vue-router';
import IntegratedHomeworkPage from '../components/IntegratedHomeworkPage.vue';
import LoginPage from '../components/LoginPage.vue';
import RegisterPage from '../components/RegisterPage.vue';

Vue.use(VueRouter);

const routes = [
  {
    path: '/',
    redirect: '/homework'
  },
  {
    path: '/login',
    name: 'Login',
    component: LoginPage,
    meta: { requiresAuth: false }
  },
  {
    path: '/register',
    name: 'Register',
    component: RegisterPage,
    meta: { requiresAuth: false }
  },
  {
    path: '/homework',
    name: 'Homework',
    component: IntegratedHomeworkPage,
    meta: { requiresAuth: true }
  },
  // 其他路由可以在这里添加
];

const router = new VueRouter({
  mode: 'history',
  base: process.env.BASE_URL,
  routes
});

// 路由守卫
router.beforeEach((to, from, next) => {
  const token = localStorage.getItem('token');
  const requiresAuth = to.matched.some(record => record.meta.requiresAuth);

  if (requiresAuth && !token) {
    // 需要认证但没有token，跳转到登录页
    next('/login');
  } else if ((to.path === '/login' || to.path === '/register') && token) {
    // 已登录用户访问登录/注册页，跳转到作业页
    next('/homework');
  } else {
    // 正常访问
    next();
  }
});

export default router;