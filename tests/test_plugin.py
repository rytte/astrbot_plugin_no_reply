"""Validate configuration, prompt injection, and native tool-loop completion."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml
from astrbot.api.event import MessageChain
from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.astr_agent_hooks import MainAgentHooks
from astrbot.core.astr_agent_run_util import run_agent
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from astrbot.core.provider.provider import Provider
from astrbot.core.provider.register import llm_tools


def make_tool(plugin, *, active=True):
    return FunctionTool(
        name="no_reply",
        description="End the turn without a reply.",
        parameters={"type": "object", "properties": {}},
        handler=plugin.no_reply,
        active=active,
    )


def test_metadata_and_schema_match_plugin(env):
    plugin_dir = Path(__file__).resolve().parents[1]
    schema = json.loads((plugin_dir / "_conf_schema.json").read_text(encoding="utf-8"))
    metadata = yaml.safe_load(
        (plugin_dir / "metadata.yaml").read_text(encoding="utf-8")
    )
    assert set(schema) == {"inject_prompt", "prompt"}
    assert schema["inject_prompt"]["default"] is False
    assert "no_reply" in schema["prompt"]["default"]
    assert schema["prompt"]["condition"] == {"inject_prompt": True}
    assert metadata["name"] == "astrbot_plugin_no_reply"
    assert metadata["display_name"] == "自主沉默"


def test_registered_tool_has_no_parameters(env):
    tool = llm_tools.get_func(env.module.TOOL_NAME)
    assert tool is not None
    assert tool.parameters["properties"] == {}
    assert "本轮不回复" in tool.description


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"inject_prompt": "false"}, "inject_prompt 必须为布尔值"),
        ({"inject_prompt": 1}, "inject_prompt 必须为布尔值"),
        ({"prompt": None}, "prompt 必须为字符串"),
        ({"inject_prompt": True, "prompt": " \n "}, "prompt 不能为空"),
        ({"unknown": True}, "未知字段：unknown"),
    ],
)
def test_invalid_configuration_fails_fast(env, changes, error):
    with pytest.raises(ValueError, match=error):
        env.module.NoReplyPlugin(env.context, env.config | changes)


@pytest.mark.parametrize("missing_key", ["inject_prompt", "prompt"])
def test_missing_configuration_fails_fast(env, missing_key):
    config = dict(env.config)
    del config[missing_key]
    with pytest.raises(ValueError, match=f"缺少字段：{missing_key}"):
        env.module.NoReplyPlugin(env.context, config)


async def test_no_reply_clears_pending_result_without_stopping_event(env):
    env.event.should_call_llm(True)
    env.event.set_result(env.event.plain_result("不应发送的结果"))
    result = await env.plugin.no_reply(env.event)
    assert result is None
    assert env.event.get_result() is None
    assert not env.event.is_stopped()
    assert env.event.call_llm
    env.event.send.assert_not_awaited()


async def test_prompt_injection_is_disabled_by_default(env):
    request = ProviderRequest(
        system_prompt="原有人格", func_tool=ToolSet([make_tool(env.plugin)])
    )
    await env.plugin.inject_guidance(env.event, request)
    assert request.system_prompt == "原有人格"


@pytest.mark.parametrize("original_prompt", ["", "原有人格"])
async def test_prompt_is_appended_once(env, original_prompt):
    plugin = env.module.NoReplyPlugin(
        env.context, {"inject_prompt": True, "prompt": "  自定义引导  "}
    )
    request = ProviderRequest(
        system_prompt=original_prompt, func_tool=ToolSet([make_tool(plugin)])
    )
    await plugin.inject_guidance(env.event, request)
    await plugin.inject_guidance(env.event, request)
    expected = f"{original_prompt}\n\n自定义引导" if original_prompt else "自定义引导"
    assert request.system_prompt == expected
    assert not env.event.is_stopped()


@pytest.mark.parametrize("tool_state", ["none", "empty", "other", "inactive"])
async def test_prompt_is_not_injected_without_available_tool(env, tool_state):
    plugin = env.module.NoReplyPlugin(env.context, env.config | {"inject_prompt": True})
    tools = None
    if tool_state != "none":
        tools = ToolSet()
    if tool_state == "inactive":
        tools.add_tool(make_tool(plugin, active=False))
    if tool_state == "other":
        tool = make_tool(plugin)
        tool.name = "other_tool"
        tools.add_tool(tool)
    request = ProviderRequest(system_prompt="原有人格", func_tool=tools)
    await plugin.inject_guidance(env.event, request)
    assert request.system_prompt == "原有人格"


class TestProvider(Provider):
    __test__ = False

    def __init__(self, *, silent=True):
        super().__init__({}, {})
        self.calls = 0
        self.silent = silent

    def get_current_key(self):
        return "test-key"

    def set_key(self, key):
        pass

    async def get_models(self):
        return ["test-model"]

    async def text_chat(self, **kwargs):
        self.calls += 1
        if self.silent:
            return LLMResponse(
                role="assistant",
                completion_text="",
                tools_call_name=["no_reply"],
                tools_call_args=[{}],
                tools_call_ids=["call-no-reply"],
            )
        return LLMResponse(role="assistant", completion_text="正常回复")

    async def text_chat_stream(self, **kwargs):
        response = await self.text_chat(**kwargs)
        if response.completion_text:
            yield LLMResponse(
                role="assistant",
                completion_text=response.completion_text,
                is_chunk=True,
            )
        yield response


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("use_main_hooks", [False, True])
async def test_native_runner_ends_silently_and_next_turn_works(
    env, monkeypatch, streaming, use_main_hooks
):
    monkeypatch.setattr(
        "astrbot.core.astr_agent_hooks.call_event_hook", AsyncMock(return_value=False)
    )
    provider = TestProvider()
    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=provider,
        request=ProviderRequest(
            prompt=env.event.message_str,
            func_tool=ToolSet([make_tool(env.plugin)]),
        ),
        run_context=ContextWrapper(
            AstrAgentContext(context=env.context, event=env.event)
        ),
        tool_executor=FunctionToolExecutor(),
        agent_hooks=MainAgentHooks() if use_main_hooks else BaseAgentRunHooks(),
        streaming=streaming,
    )
    env.event.set_result(env.event.plain_result("待发送内容"))
    replies = [
        chain async for chain in run_agent(runner, max_step=1, show_tool_use=False)
    ]
    assert replies == []
    assert provider.calls == 1
    assert runner.done()
    assert not runner.was_aborted()
    assert runner.get_final_llm_resp() is None
    assert env.event.get_result() is None
    assert not env.event.is_stopped()
    env.event.send.assert_not_awaited()

    provider.silent = False
    await runner.reset(
        provider=provider,
        request=ProviderRequest(
            prompt="请正常回复", func_tool=ToolSet([make_tool(env.plugin)])
        ),
        run_context=ContextWrapper(
            AstrAgentContext(context=env.context, event=env.event)
        ),
        tool_executor=FunctionToolExecutor(),
        agent_hooks=MainAgentHooks() if use_main_hooks else BaseAgentRunHooks(),
        streaming=streaming,
    )
    replies = [chain async for chain in run_agent(runner, show_tool_use=False)]
    assert any(
        isinstance(chain, MessageChain) and chain.get_plain_text() == "正常回复"
        for chain in replies
    )
    assert provider.calls == 2
    assert not env.event.is_stopped()
