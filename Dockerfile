# 使用官方 Python 3.11 基础镜像
FROM python:3.11

# 设置工作目录
WORKDIR /app

# 一次性安装所有必要的依赖，减少镜像层数
RUN apt update && apt install -y --no-install-recommends \
    cmake g++ make libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*  # 清理缓存，减小镜像大小

# 复制依赖文件并安装 Python 包
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 设置默认运行命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8809", "--workers", "4"]
