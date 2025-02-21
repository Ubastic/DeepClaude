"""Claude API 客户端"""
import json
import random
from typing import AsyncGenerator,Optional,List
from app.utils.logger import logger
from .base_client import BaseClient
import aiohttp
class TokenManager:
    def __init__(self, token_file: str):
        """初始化 Token 管理器
        
        Args:
            token_file: token 配置文件路径
        """
        self.token_file = token_file
        self.tokens: List[dict] = []
        self.current_token_index = 0
        self.load_tokens()
        
    def load_tokens(self):
        """从文件加载 tokens"""
        try:
            with open(self.token_file, 'r', encoding='utf-8') as f:
                self.tokens = json.load(f)
                logger.info(f"成功加载 {len(self.tokens)} 个 token")
        except Exception as e:
            logger.error(f"加载 token 文件失败: {e}")
            self.tokens = []
            
    def get_next_token(self) -> Optional[str]:
        """获取下一个可用的 token
        
        Returns:
            str | None: 下一个 token，如果没有可用 token 则返回 None
        """
        if not self.tokens:
            return None
            
        # 记录初始索引
        start_index = self.current_token_index
        
        while True:
            token_info = self.tokens[self.current_token_index]
            # 如果 token 未被标记为已用完，则返回
            if not token_info.get("exhausted", False):
                logger.info(f"使用 token {self.current_token_index + 1}/{len(self.tokens)}")
                return token_info["token"]
                
            # 移动到下一个 token
            self.current_token_index = (self.current_token_index + 1) % len(self.tokens)
            
            # 如果已经检查了所有 token，退出循环
            if self.current_token_index == start_index:
                logger.error("所有 token 已用完")
                return None
                
    def mark_token_exhausted(self, token: str):
        """标记某个 token 已用完
        
        Args:
            token: 要标记的 token
        """
        for i, token_info in enumerate(self.tokens):
            if token_info["token"] == token:
                token_info["exhausted"] = True
                logger.warning(f"Token {i + 1}/{len(self.tokens)} 已标记为用完")
                # 移动到下一个 token
                self.current_token_index = (i + 1) % len(self.tokens)
                break

    def reset_exhausted_status(self):
        """重置所有 token 的使用状态"""
        for token_info in self.tokens:
            token_info["exhausted"] = False
        logger.info("已重置所有 token 的使用状态")

class ClaudeClient(BaseClient):
    def __init__(self, api_key: str,token_file:str, api_url: str = "https://api.anthropic.com/v1/messages", provider: str = "anthropic"):
        """初始化 Claude 客户端
        
        Args:
            
            api_key: Claude API密钥
            api_url: Claude API地址
            is_openrouter: 是否使用 OpenRouter API
        """
        self.token_manager = TokenManager(token_file)
        initial_token = self.token_manager.get_next_token()
        if not initial_token and api_key:
            raise ValueError("没有可用的 token")
        if api_key:
            super().__init__(api_key, api_url)
        else:
            super().__init__(initial_token, api_url)
        self.provider = provider
    
    async def _make_request(self, headers: dict, data: dict) -> AsyncGenerator[bytes, None]:
        """发送请求并处理响应，支持 token 轮换

        Args:
            headers: 请求头
            data: 请求数据

        Yields:
            bytes: 原始响应数据
        """
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.api_url, headers=headers, json=data) as response:
                        if response.status != 200:
                            error_text = await response.text()

                            # 检查是否是配额不足错误
                            try:
                                error_data = json.loads(error_text)
                                error_code = error_data.get("error", {}).get("code")
                                if error_code == "insufficient_user_quota":
                                    # 标记当前 token 已用完
                                    self.token_manager.mark_token_exhausted(self.api_key)

                                    # 获取下一个 token
                                    next_token = self.token_manager.get_next_token()
                                    if not next_token:
                                        logger.error("没有更多可用的 token")
                                        return

                                    # 更新 token 并重试
                                    self.api_key = next_token
                                    headers = self._update_headers(headers, next_token)
                                    continue
                            except json.JSONDecodeError:
                                pass

                            logger.error(f"API 请求失败: {error_text}")
                            return

                        async for chunk in response.content.iter_any():
                            yield chunk

                # 如果成功完成，跳出循环
                break

            except Exception as e:
                logger.error(f"请求 API 时发生错误: {e}")
                return

    def _update_headers(self, headers: dict, new_token: str) -> dict:
        """根据 provider 更新请求头中的 token

        Args:
            headers: 原始请求头
            new_token: 新的 token

        Returns:
            dict: 更新后的请求头
        """
        if self.provider == "anthropic":
            headers["x-api-key"] = new_token
        else:  # openrouter 或 oneapi
            headers["Authorization"] = f"Bearer {new_token}"
        return headers

    async def stream_chat(self, messages: list, model: str = "claude-3-5-sonnet-20241022") -> AsyncGenerator[tuple[str, str], None]:
        """流式对话
        
        Args:
            messages: 消息列表
            model: 模型名称。如果是 OpenRouter，会自动转换为 'anthropic/claude-3.5-sonnet' 格式
            
        Yields:
            tuple[str, str]: (内容类型, 内容)
                内容类型: "answer"
                内容: 实际的文本内容
        """

        if self.provider == "openrouter":
            logger.info("使用 OpenRouter API 作为 Claude 3.5 Sonnet 供应商 ")
            # 转换模型名称为 OpenRouter 格式
            model = "anthropic/claude-3.5-sonnet"
                
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/ErlichLiu/DeepClaude",  # OpenRouter 需要
                "X-Title": "DeepClaude"  # OpenRouter 需要
            }
            
            data = {
                "model": model,  # OpenRouter 使用 anthropic/claude-3.5-sonnet 格式
                "messages": messages,
                "stream": True
            }
        elif self.provider == "oneapi":
            logger.info("使用 OneAPI API 作为 Claude 3.5 Sonnet 供应商 ")
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": model,
                "messages": messages,
                "stream": True
            }
        elif self.provider == "anthropic":
            logger.info("使用 Anthropic API 作为 Claude 3.5 Sonnet 供应商 ")
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                "accept": "text/event-stream",
            }
            
            data = {
                "model": model,
                "messages": messages,
                "max_tokens": 8192,
                "stream": True
            }
        else:
            raise ValueError(f"不支持的Claude Provider: {self.provider}")
        
        async for chunk in self._make_request(headers, data):
            chunk_str = chunk.decode('utf-8')
            #logger.info(f"claude_client{chunk_str}")
            if not chunk_str.strip():
                continue
                
            for line in chunk_str.split('\n'):
                if line.startswith('data: '):
                    json_str = line[6:]  # 去掉 'data: ' 前缀
                    if json_str.strip() == '[DONE]':
                        return
                        
                    try:
                        data = json.loads(json_str)
                        if self.provider in ("openrouter", "oneapi"):
                            # OpenRouter/OneApi 格式
                            content = data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                            if content:
                                yield "answer", content
                        elif self.provider == "anthropic":
                            # Anthropic 格式
                            if data.get('type') == 'content_block_delta':
                                content = data.get('delta', {}).get('text', '')
                                if content:
                                    yield "answer", content
                        else:
                            raise ValueError(f"不支持的Claude Provider: {self.provider}")
                    except json.JSONDecodeError:
                        continue
