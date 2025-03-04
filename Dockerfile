# 使用官方 Python 3.11 基础镜像
FROM python:3.11

# 设置工作目录
WORKDIR /app

COPY requirements.txt .
# 安装依赖工具（cmake、g++、make）
RUN apt update && apt install -y cmake g++ make libgl1 libglib2.0-0 && RUN pip install --no-cache-dir -r requirements.txt
# 复制依赖文件并安装


# 复制应用代码
COPY . .

# 设置默认运行命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8809", "--workers", "4"]
