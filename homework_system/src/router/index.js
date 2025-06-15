import Vue from 'vue';
import VueRouter from 'vue-router';
import IntegratedHomeworkPage from '../components/IntegratedHomeworkPage.vue';

Vue.use(VueRouter);

const routes = [
  {
    path: '/',
    redirect: '/homework'
  },
  {
    path: '/homework',
    name: 'Homework',
    component: IntegratedHomeworkPage
  },
  // 其他路由可以在这里添加
];

const router = new VueRouter({
  mode: 'history',
  base: process.env.BASE_URL,
  routes
});

export default router; 