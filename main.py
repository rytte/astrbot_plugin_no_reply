"""Allow the model to end a turn without sending a reply."""

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star

TOOL_NAME = "no_reply"


class NoReplyPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        expected_keys = {"inject_prompt", "prompt"}
        missing_keys = expected_keys - config.keys()
        unknown_keys = config.keys() - expected_keys
        if missing_keys:
            raise ValueError(f"自主沉默配置缺少字段：{', '.join(sorted(missing_keys))}")
        if unknown_keys:
            raise ValueError(
                f"自主沉默配置包含未知字段：{', '.join(sorted(unknown_keys))}"
            )
        if not isinstance(config["inject_prompt"], bool):
            raise ValueError("自主沉默配置 inject_prompt 必须为布尔值。")
        if not isinstance(config["prompt"], str):
            raise ValueError("自主沉默配置 prompt 必须为字符串。")

        self.inject_prompt = config["inject_prompt"]
        self.prompt = config["prompt"].strip()
        if self.inject_prompt and not self.prompt:
            raise ValueError("启用自主沉默提示词注入时，prompt 不能为空。")

    @filter.llm_tool(name=TOOL_NAME)
    async def no_reply(self, event: AstrMessageEvent) -> None:
        """选择本轮不回复，结束当前模型回复流程，不发送任何消息。

        决定保持沉默时仅调用本工具，不要先输出文字、解释原因或输出占位内容，
        也不要同时调用其他工具。需要正常回复时不要调用本工具。
        本工具无需参数，不会永久静音，也不会影响后续消息。
        """
        event.clear_result()

    @filter.on_llm_request()
    async def inject_guidance(
        self, event: AstrMessageEvent, request: ProviderRequest
    ) -> None:
        if not self.inject_prompt or request.func_tool is None:
            return
        tool = request.func_tool.get_tool(TOOL_NAME)
        if tool is None or not tool.active or self.prompt in request.system_prompt:
            return
        request.system_prompt = "\n\n".join(
            part for part in (request.system_prompt, self.prompt) if part
        )
