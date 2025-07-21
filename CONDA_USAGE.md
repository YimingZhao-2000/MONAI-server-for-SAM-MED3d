# Python Interpreter Usage

## 方法1：通过环境变量指定Python解释器路径

```bash
# 设置Python解释器路径（conda环境）
export PYTHON_PATH="/path/to/conda/envs/sam_med3d_env/bin/python"

# 启动服务器
./start_server.sh
```

## 方法2：直接在命令行中设置

```bash
# 一次性设置并启动
PYTHON_PATH="/path/to/conda/envs/sam_med3d_env/bin/python" ./start_server.sh
```

## 方法3：修改脚本中的默认值

编辑 `start_server.sh` 文件，修改第8行：
```bash
PYTHON_PATH=${PYTHON_PATH:-"/path/to/your/python"}  # 设置默认Python路径
```

## 方法4：使用默认Python

如果不指定 `PYTHON_PATH` 环境变量，脚本会使用系统默认的Python：
```bash
./start_server.sh
```

## 常见Python路径示例

### Conda环境
```bash
# 激活环境后查看路径
conda activate sam_med3d_env
which python
# 输出类似：/home/user/anaconda3/envs/sam_med3d_env/bin/python

# 使用该路径
export PYTHON_PATH="/home/user/anaconda3/envs/sam_med3d_env/bin/python"
```

### 虚拟环境
```bash
# 虚拟环境路径
export PYTHON_PATH="/path/to/venv/bin/python"
```

### 系统Python
```bash
# 使用特定版本的Python
export PYTHON_PATH="/usr/bin/python3.8"
```

## 注意事项

1. **路径正确性**：确保指定的Python路径存在且可执行
2. **依赖包已安装**：确保该Python环境中已安装所有必需的包（见 `requirements.txt`）
3. **权限问题**：确保脚本有执行权限：`chmod +x start_server.sh`

## 常见问题

- 如果提示 "Python interpreter not found"，请检查路径是否正确
- 如果提示模块找不到，请确保该Python环境中已安装monailabel等依赖包
- 可以通过以下命令查看当前使用的Python路径：
  ```bash
  which python
  ``` 