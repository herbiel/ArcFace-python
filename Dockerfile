# 使用官方 Python 3.11 基础镜像
FROM python:3.11

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 设置默认运行命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8801", "--workers", "4"]
