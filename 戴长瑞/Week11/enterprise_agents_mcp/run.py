#!/usr/bin/env python3
import subprocess
import sys
import time
import os
from pathlib import Path

# 将项目根目录加入环境变量
root_dir = Path(__file__).parent
os.environ["PYTHONPATH"] = str(root_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, str(root_dir))

# 确保日志目录存在
Path("logs").mkdir(exist_ok=True)

def start_mcp_server(script_path: str, port: int):
    proc = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    time.sleep(2)
    print(f"✅ MCP 服务器 {script_path} 启动在端口 {port}")
    return proc

def start_streamlit():
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run",
        "frontend/streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ]
    subprocess.run(streamlit_cmd)

if __name__ == "__main__":
    print("启动企业级 MCP 多 Agent 系统...")
    sentiment_proc = start_mcp_server("mcp_servers/sentiment_server.py", 8901)
    ner_proc = start_mcp_server("mcp_servers/ner_server.py", 8902)
    try:
        start_streamlit()
    except KeyboardInterrupt:
        print("\n正在关闭服务...")
    finally:
        sentiment_proc.terminate()
        ner_proc.terminate()
        print("服务已关闭")