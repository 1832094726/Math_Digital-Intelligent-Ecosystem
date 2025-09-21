module.exports = {
  devServer: {
    port: 8080,
    proxy: {
      '/api': {
        target: 'http://172.104.172.5:8081', // 后端API地址
        changeOrigin: true,
        pathRewrite: {
          '^/api': '/api' // 保留/api前缀以匹配后端路由
        }
      }
    }
  },
  // 输出目录
  outputDir: 'dist',
  // 静态资源目录
  assetsDir: 'static',
  // 生产环境是否生成 sourceMap 文件
  productionSourceMap: false
}; 