import Vue from 'vue';
import ElementUI from 'element-ui';
import 'element-ui/lib/theme-chalk/index.css';
import './assets/global-override.css'; // 引入全局样式覆盖
import App from './App.vue';
import router from './router';
import store from './store';

// 使用ElementUI
Vue.use(ElementUI);

Vue.config.productionTip = false;

new Vue({
  router,
  store,
  render: h => h(App)
}).$mount('#app'); 