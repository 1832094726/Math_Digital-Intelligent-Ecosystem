module.exports = {
  devServer: {
    port: 8080,
    proxy: {
      '/api': {
        target: 'http://localhost:5000', // 后端API地址
        changeOrigin: true,
        pathRewrite: {
          '^/api': ''
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