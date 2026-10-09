"""Load the plugin against the adjacent AstrBot source tree."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
ASTRBOT_SOURCE = PLUGIN_DIR.parent / "AstrBot"
if not (ASTRBOT_SOURCE / "astrbot").is_dir():
    raise RuntimeError("测试需要相邻的 ../AstrBot 源码目录。")
sys.path.insert(0, str(ASTRBOT_SOURCE))
os.environ["ASTRBOT_ROOT"] = str(PLUGIN_DIR / ".pytest_cache" / "astrbot")

from astrbot.api import sp
from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import Plain
from astrbot.core.platform.astrbot_message import (
    AstrBotMessage,
    MessageMember,
)
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata
from astrbot.core.provider.register import llm_tools
from astrbot.core.star.context import Context

spec = importlib.util.spec_from_file_location("no_reply_plugin", PLUGIN_DIR / "main.py")
plugin_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin_module
spec.loader.exec_module(plugin_module)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(sp, "global_get", AsyncMock(return_value={}))
    message = AstrBotMessage()
    message.type = MessageType.FRIEND_MESSAGE
    message.self_id = "bot"
    message.session_id = "session"
    message.message_id = "message"
    message.sender = MessageMember(user_id="user", nickname="User")
    message.message_str = "这条消息无需回应"
    message.message = [Plain(message.message_str)]
    event = AstrMessageEvent(
        message.message_str,
        message,
        PlatformMetadata("test", "Test", "test"),
        "session",
    )
    event.send = AsyncMock()
    context = Context.__new__(Context)
    context.provider_manager = SimpleNamespace(llm_tools=llm_tools)
    schema = json.loads((PLUGIN_DIR / "_conf_schema.json").read_text(encoding="utf-8"))
    config = {key: field["default"] for key, field in schema.items()}
    return SimpleNamespace(
        event=event,
        context=context,
        config=config,
        plugin=plugin_module.NoReplyPlugin(context, config),
        module=plugin_module,
    )
