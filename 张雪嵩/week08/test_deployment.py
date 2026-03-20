import sys
import os

os.chdir(r"d:\AI学习\第8周：智能体Agent基础(1)\代码\04-government-advanced-rag")
sys.path.insert(0, os.getcwd())

print("=" * 60)
print("Government Advanced RAG 系统部署测试")
print("=" * 60)

print("\n【步骤1】测试数据库连接...")
try:
    from db_api import Session, KnowledgeDatabase, Base, engine
    print("  [OK] 数据库模块导入成功")
    print("  [OK] 数据库文件: rag.db")
except Exception as e:
    print(f"  [FAIL] 数据库导入失败: {e}")

print("\n【步骤2】测试 Elasticsearch 连接...")
try:
    import httpx
    response = httpx.get("http://localhost:9200", timeout=5)
    if response.status_code == 200:
        print("  [OK] Elasticsearch 连接成功")
        es_info = response.json()
        print(f"  [OK] ES 版本: {es_info.get('version', {}).get('number', 'unknown')}")
    else:
        print(f"  [FAIL] ES 返回状态码: {response.status_code}")
except Exception as e:
    print(f"  [FAIL] Elasticsearch 未启动")
    print("  >>> 请先启动 Elasticsearch:")
    print("  >>>   cd D:\\Tools\\elasticsearch-8.11.0\\bin")
    print("  >>>   $env:JAVA_HOME=$null; .\\elasticsearch.bat")
    print("  >>> 或者使用 Docker: docker run -d -p 9200:9200 -e \"discovery.type=single-node\" elasticsearch:8.11.0")

print("\n【步骤3】测试 Embedding 模型...")
try:
    from sentence_transformers import SentenceTransformer
    print("  [OK] sentence_transformers 导入成功")
    print("  [INFO] 正在加载 bge-small-zh-v1.5 模型...")
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    test_embedding = model.encode("测试文本")
    print(f"  [OK] 模型加载成功，向量维度: {len(test_embedding)}")
except Exception as e:
    print(f"  [FAIL] Embedding 模型加载失败: {e}")

print("\n【步骤4】测试 LLM API 连接...")
try:
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    from openai import OpenAI
    client = OpenAI(
        api_key=config["rag"]["llm_api_key"],
        base_url=config["rag"]["llm_base"]
    )
    
    response = client.chat.completions.create(
        model=config["rag"]["llm_model"],
        messages=[{"role": "user", "content": "你好"}],
        max_tokens=10
    )
    print(f"  [OK] LLM API 连接成功")
    print(f"  [OK] 模型: {config['rag']['llm_model']}")
    print(f"  [OK] 响应: {response.choices[0].message.content}")
except Exception as e:
    print(f"  [FAIL] LLM API 连接失败: {e}")

print("\n【步骤5】测试 FastAPI 应用导入...")
try:
    from main import app
    print("  [OK] FastAPI 应用导入成功")
except Exception as e:
    print(f"  [FAIL] FastAPI 应用导入失败: {e}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
print("\n【启动服务命令】")
print("  cd d:\\AI学习\\第8周：智能体Agent基础(1)\\代码\\04-government-advanced-rag")
print("  D:\\ProgramData\\miniconda3\\envs\\py312\\python.exe -m uvicorn main:app --host 0.0.0.0 --port 6010")
