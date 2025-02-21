#encoding=utf8
"""DeepClaude 服务，用于协调 DeepSeek 和 Claude API 的调用"""
import json
import time
import asyncio
from typing import AsyncGenerator
from app.utils.logger import logger
from app.clients import DeepSeekClient, ClaudeClient


class DeepClaude:
    """处理 DeepSeek 和 Claude API 的流式输出衔接"""
    
    def __init__(self, deepseek_api_key: str,token_file: str, claude_api_key: str, 
                 deepseek_api_url: str = "https://api.deepseek.com/v1/chat/completions", 
                 claude_api_url: str = "https://api.anthropic.com/v1/messages",
                 claude_provider: str = "anthropic"):
        """初始化 API 客户端
        
        Args:
            deepseek_api_key: DeepSeek API密钥
            claude_api_key: Claude API密钥
        """
        self.deepseek_client = DeepSeekClient(deepseek_api_key, deepseek_api_url)
        self.claude_client = ClaudeClient(claude_api_key,token_file, claude_api_url, claude_provider)
    
    async def chat_completions(self, messages: list,
                             deepseek_model: str = "deepseek-reasoner",
                             claude_model: str = "claude-3-5-sonnet-20241022") -> dict:
        """处理完整的非流式输出过程

        Args:
            messages: 初始消息列表
            deepseek_model: DeepSeek 模型名称
            claude_model: Claude 模型名称

        Returns:
            dict: 完整的响应数据，格式如下：
            {
                "id": "chatcmpl-xxx",
                "object": "chat.completion",
                "created": timestamp,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": x,
                    "completion_tokens": y,
                    "total_tokens": z
                }
            }
        """
        try:
            # 生成唯一的会话ID和时间戳
            chat_id = f"chatcmpl-{hex(int(time.time() * 1000))[2:]}"
            created_time = int(time.time())

            # 收集 DeepSeek 的推理内容
            reasoning_content = []
            final_content = ""

            # 处理 DeepSeek 输出
            logger.info(f"开始处理 DeepSeek 请求，使用模型：{deepseek_model}")
            async for content_type, content in self.deepseek_client.stream_chat(messages, deepseek_model):
                if content_type == "reasoning":
                    reasoning_content.append(content)
                elif content_type == "content":
                    final_content = content
                    break
            full_reasoning = "".join(reasoning_content)
            logger.info(f"DeepSeek 推理完成，收集到的推理内容长度：{len(full_reasoning)}")

            if not full_reasoning:
                raise ValueError("未能获取到有效的推理内容")

            # 构造 Claude 的输入消息
            claude_messages = messages.copy()
            claude_messages.append({
                "role": "assistant",
                "content": f"Here's my reasoning process:\n{full_reasoning}\n\nBased on this reasoning, I will now provide my response:"
            })

            # 处理可能存在的 system role 消息
            claude_messages = [message for message in claude_messages if message.get("role", "") != "system"]

            # 收集 Claude 的输出内容
            logger.info(f"开始处理 Claude 请求，使用模型：{claude_model}")
            claude_content = []
            async for content_type, content in self.claude_client.stream_chat(claude_messages, claude_model):
                #logger.info(f"deepclaude{content_type} {content}")
                if content_type == "answer":
                    claude_content.append(content)

            final_content = "<think>"+full_reasoning+"</think>"+"".join(claude_content)

            # 构造最终响应
            return {
                "id": chat_id,
                "object": "chat.completion",
                "created": created_time,
                "model": claude_model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": final_content
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len("".join(str(m.get("content", "")) for m in messages)) // 4,  # 粗略估算
                    "completion_tokens": len(final_content) // 4,  # 粗略估算
                    "total_tokens": (len("".join(str(m.get("content", "")) for m in messages)) + len(final_content)) // 4  # 粗略估算
                }
            }

        except Exception as e:
            logger.error(f"处理请求时发生错误: {e}")
            raise

    async def chat_completions_with_stream(self, messages: list, 
                                         deepseek_model: str = "deepseek-reasoner",
                                         claude_model: str = "claude-3-5-sonnet-20241022") -> AsyncGenerator[bytes, None]:
        """处理完整的流式输出过程
        
        Args:
            messages: 初始消息列表
            deepseek_model: DeepSeek 模型名称
            claude_model: Claude 模型名称
            
        Yields:
            字节流数据，格式如下：
            {
                "id": "chatcmpl-xxx",
                "object": "chat.completion.chunk",
                "created": timestamp,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "reasoning_content": reasoning_content,
                        "content": content
                    }
                }],
                "usage":{"prompt_tokens":6,"completion_tokens":0,"total_tokens":6}
            }
        """
        # 生成唯一的会话ID和时间戳
        chat_id = f"chatcmpl-{hex(int(time.time() * 1000))[2:]}"
        created_time = int(time.time())

        # 创建队列，用于收集输出数据
        output_queue = asyncio.Queue()
        # 队列，用于传递 DeepSeek 推理内容给 Claude
        claude_queue = asyncio.Queue()
        
        # 用于存储 DeepSeek 的推理累积内容
        reasoning_content = []
        
        async def process_deepseek():
            logger.info(f"开始处理 DeepSeek 流，使用模型：{deepseek_model}")
            try:
                async for content_type, content in self.deepseek_client.stream_chat(messages, deepseek_model):
                    if content_type == "reasoning":
                        reasoning_content.append(content)
                        response = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": deepseek_model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "reasoning_content": content,
                                    "content": ""
                                }
                            }],
                            "usage":{"prompt_tokens":6,"completion_tokens":0,"total_tokens":6}
                        }
                        await output_queue.put(f"data: {json.dumps(response)}\n\n".encode('utf-8'))
                    elif content_type == "content":
                        # 当收到 content 类型时，将完整的推理内容发送到 claude_queue，并结束 DeepSeek 流处理
                        logger.info(f"DeepSeek 推理完成，收集到的推理内容长度：{len(''.join(reasoning_content))}")
                        await claude_queue.put("".join(reasoning_content))
                        break
            except Exception as e:
                logger.error(f"处理 DeepSeek 流时发生错误: {e}")
                await claude_queue.put("")
            # 用 None 标记 DeepSeek 任务结束
            logger.info("DeepSeek 任务处理完成，标记结束")
            await output_queue.put(None)
        
        async def process_claude():
            try:
                logger.info("等待获取 DeepSeek 的推理内容...")
                reasoning = await claude_queue.get()
                logger.debug(f"获取到推理内容，内容长度：{len(reasoning) if reasoning else 0}")
                if not reasoning:
                    logger.error("未能获取到有效的推理内容")
                    # 标记 Claude 任务结束
                    await output_queue.put(None)
                    return
                # 构造 Claude 的输入消息
                claude_messages = messages.copy()
                claude_messages.append({
                    "role": "assistant",
                    "content": f"Here's my reasoning process:\n{reasoning}\n\nBased on this reasoning, I will now provide my response:"
                })
                # 处理可能 messages 内存在 role = system 的情况，如果有，则去掉当前这一条的消息对象
                claude_messages = [message for message in claude_messages if message.get("role", "") != "system"]

                async for content_type, content in self.claude_client.stream_chat(claude_messages, claude_model):
                    if content_type == "answer":
                        response = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": claude_model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": content
                                }
                            }],
                            "usage":{"prompt_tokens":6,"completion_tokens":0,"total_tokens":6}
                        }
                        await output_queue.put(f"data: {json.dumps(response)}\n\n".encode('utf-8'))
            except Exception as e:
                logger.error(f"处理 Claude 流时发生错误: {e}")
            # 用 None 标记 Claude 任务结束
            await output_queue.put(None)
        
        # 创建并发任务
        deepseek_task = asyncio.create_task(process_deepseek())
        claude_task = asyncio.create_task(process_claude())
        
        # 等待两个任务完成，通过计数判断
        finished_tasks = 0
        while finished_tasks < 2:
            item = await output_queue.get()
            if item is None:
                finished_tasks += 1
            else:
                yield item
        
        # 发送结束标记
        yield b'data: [DONE]\n\n'
