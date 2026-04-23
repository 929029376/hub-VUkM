import sqlite3
import json
from typing import List, Dict, Any, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import re
from pathlib import Path
from openai import OpenAI 
import traceback

#sk-c327102ab1ef4e50991493e949b21d57"
client = OpenAI(
    api_key="sk-c327102ab1ef4e50991493e949b21d57",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 连接到Chinook数据库
conn = sqlite3.connect('chinook.db')

# 创建一个游标对象
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cursor.fetchall()]

class SQLiteAgent:
    # 初始化
    def __init__(self):
        self.retry_time = 3
        
        self._planer_prompt = """你是一个专业的SQL生成专家，基于以下数据库表信息：{#tables#}，将用户的问题转换为数据库查询sql，现在请规划下面的问题可以从哪些表中获取答案：{#task#} """
        self._sql_gen_prompt = """你是一个数据库管理员，擅长将用户问题的问题：{#task#}，转化为为数据库查询sql

实现步骤参考：{#thought#}
  需要先检查本地数据库中的表，是否有用户想要的结果，没有的话直接返回“没有找到想要的数据”，否则就根据用户问题去生成可执行的sql，输出格式如下
  ```python
```
"""
        self._sql_reflection_prompt = """上述sql执行存在错误，错误信息为{#error#}，请在原有sql基础上进行改进，先写出新sql。

```python
{#sql#}
```
"""
        self._summary_prompt = """你是数据库执行专家，将用户的提问{#task#} 和 得到的sql{#sql#}执行结果:  简单汇总为自然语言回答。"""
        

    def action(self, question) -> Any:
        # 类似rag过程，找一下历史最接近的提问和代码
        
        # 思维链 -》 思考完成特定的步骤 需要 做什么
        init_thought = self.llm(
            messages=[    
                {"role": "user", "content": self._planer_prompt.replace("{#tables#}", json.dumps(tables, indent=2, ensure_ascii=False) ).replace("{#task#}", question)} 
            ],
        )
        
         
        # 生成代码 -》 借助思考过程生成代码
        messages=[    
            {"role": "user", "content": self._sql_gen_prompt.replace("{#task#}", question).replace("{#thought#}", init_thought)} 
        ]
        init_sql = self.llm(
            messages=messages, model="qwen-plus"
        )
        
        messages.append({
            "role": "system", "content": init_sql
        })
        init_sql = self.extract_code_from_llm(init_sql) # 先抽取代码
        for retry_idx in range(self.retry_time):
            if init_sql == "":
                messages.append({
                    "role": "user", "content": "the output do not contain any pythnon code using ```python ```, please generate."
                })
                init_sql = self.llm(messages=messages, model="qwen-plus")
                init_sql = self.extract_code_from_llm(init_sql)
            else: # 如果之前抽取了代码
                execute_issucess, execute_result, code, msg = self.execute(init_sql)
                if execute_issucess:
                    # messages = [{
                    #     "role": "user", "content": self._summary_prompt.replace("{#task#}", question).replace("{#result#}", execute_result)
                    # }]
                    # final_answer = self.llm(messages=messages, model="glm-4")
                    # return final_answer
                    
                    # 记忆成功提问和代码
                    return execute_result
                    

                messages.append({
                    "role": "user", "content": self._code_reflection_prompt.replace("{#error#}", msg).replace("{#code#}", init_sql)
                })
                init_sql = self.llm(messages=messages, model="qwen-plus")
                init_sql = self.extract_code_from_llm(init_sql)

        print("生成失败")
        return None

        # 调用大模型
    def llm(self, messages, model="qwen-plus", top_p=0.7, temperature=0.9):
        print(json.dumps(messages, indent=4, ensure_ascii=False))
        try:
            completion = client.chat.completions.create(
                model=model,  
                messages=messages,
                top_p=top_p,
                temperature=temperature
            ) 
            return completion.choices[0].message.content
        except Exception as e:
            print(f'调用大模型失败:{e}')
            return None

    
         # 从大模型回答抽取代码，markdown 抽取 代码
    def extract_code_from_llm(self, text) -> str:
        pattern = '```python\n(.*?)```'
        try:
            matches = re.findall(pattern, text, re.DOTALL)
            return matches[0]
        except:
            print(traceback.format_exc())
            return ""

        # 执行代码
    def execute(self, code) -> Union[bool, str, str, str]:
        # 超时机制
        try:
            print('---')
            print(code)
            print('---')
            cursor.execute(code) # 执行代码， 或 生成代码写为文件单独执行 或 在容器里面执行
            result = cursor.fetchall()[0]
            print(result)
            return True, result, code, ""
        except Exception as e:
            error_message = traceback.format_exc()
            return False, "", code, error_message
