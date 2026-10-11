"""LLM 客户端封装：统一读取 .env 配置，提供稳定的 invoke 接口。

设计要点：
- 复用 .env 中的 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 等配置
- 支持系统提示词（默认从 .env 读取，也可在调用时覆盖）
- 超时 / 异常兜底：失败时返回空字符串，不抛异常中断流程
"""
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# 确保 .env 已加载
load_dotenv()


class LLMClient:
    """LLM 统一调用入口。

    用法：
        client = LLMClient()
        answer = client.invoke("你好")

        # 覆盖系统提示词
        answer = client.invoke("你好", system_prompt="你是翻译助手")
    """

    def __init__(self, system_prompt: str | None = None):
        self.llm = ChatOpenAI(
            model_name=os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V3"),
            openai_api_base=os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/"),
            openai_api_key=SecretStr(os.getenv("LLM_API_KEY", "")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            request_timeout=int(os.getenv("LLM_TIMEOUT", "60")),
        )
        self.system_prompt = system_prompt or os.getenv(
            "LLM_SYSTEM_PROMPT", "你是高校教务政策知识库问答助手，回答简洁准确。"
        )

    def invoke(self, prompt: str, system_prompt: str | None = None) -> str:
        """调用 LLM，返回文本。失败时返回空字符串。"""
        messages = []
        sys_p = system_prompt if system_prompt is not None else self.system_prompt
        if sys_p:
            messages.append(SystemMessage(content=sys_p))
        messages.append(HumanMessage(content=prompt))

        try:
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            print(f"  LLM 调用失败: {e}")
            return ""
