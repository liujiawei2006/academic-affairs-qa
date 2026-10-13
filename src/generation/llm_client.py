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

        # 获取 Token 消耗统计
        stats = client.get_token_stats()
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
        # Token 消耗追踪
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_tokens = 0
        self._call_count = 0

    def invoke(self, prompt: str, system_prompt: str | None = None) -> str:
        """调用 LLM，返回文本。失败时返回空字符串。"""
        messages = []
        sys_p = system_prompt if system_prompt is not None else self.system_prompt
        if sys_p:
            messages.append(SystemMessage(content=sys_p))
        messages.append(HumanMessage(content=prompt))

        try:
            response = self.llm.invoke(messages)
            # 提取 Token 消耗（langchain 返回的 response_metadata 或 usage_metadata）
            self._extract_token_usage(response)
            return response.content.strip()
        except Exception as e:
            print(f"  LLM 调用失败: {e}")
            return ""

    def _extract_token_usage(self, response) -> dict:
        """从 LLM 响应中提取 Token 消耗并累计。"""
        usage = {}
        # 优先从 usage_metadata 获取（langchain 新版本）
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
        # 备选：从 response_metadata 获取
        elif hasattr(response, "response_metadata") and response.response_metadata:
            meta = response.response_metadata
            usage = meta.get("usage", meta.get("token_usage", {}))

        prompt_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
        total = usage.get("total_tokens", 0) or (prompt_tokens + completion_tokens)

        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens
        self._total_tokens += total
        self._call_count += 1

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total,
        }

    def get_token_stats(self) -> dict:
        """返回本次生命周期内的 Token 消耗统计。"""
        return {
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_tokens,
            "call_count": self._call_count,
        }

    def reset_token_stats(self):
        """重置 Token 统计（用于按请求粒度统计）。"""
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_tokens = 0
        self._call_count = 0
