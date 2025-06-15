/**
 * 格式化日期
 * @param {Date|string} date - 日期对象或日期字符串
 * @param {string} format - 格式化模板，如 'YYYY-MM-DD HH:mm:ss'
 * @returns {string} 格式化后的日期字符串
 */
export function formatDate(date, format = 'YYYY-MM-DD HH:mm:ss') {
  if (!date) return '';
  
  // 如果是字符串，转换为日期对象
  if (typeof date === 'string') {
    date = new Date(date);
  }
  
  const year = date.getFullYear();
  const month = date.getMonth() + 1;
  const day = date.getDate();
  const hour = date.getHours();
  const minute = date.getMinutes();
  const second = date.getSeconds();
  
  // 补零函数
  const pad = (n) => n.toString().padStart(2, '0');
  
  return format
    .replace(/YYYY/g, year)
    .replace(/MM/g, pad(month))
    .replace(/DD/g, pad(day))
    .replace(/HH/g, pad(hour))
    .replace(/mm/g, pad(minute))
    .replace(/ss/g, pad(second));
} 