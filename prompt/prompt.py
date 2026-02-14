"""
Prompt Factory: 用於不同 Guardian LLM / LLM 模型共用的 system prompt
以及不同任務的 user prompt template 生成。

設計採用 Factory 模式：
- SystemPromptProvider: 依 model_id 提供共用或模型專屬的 system prompt
- UserPromptTemplate: 依 task 提供可填入變數的 user prompt 模板
- PromptFactory: 統一入口，組裝 system + user prompts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


# =============================================================================
# 共用 System Prompt 內容
# =============================================================================

GUARDIAN_SYSTEM_PROMPT_BASE = """你是一個專業的內容安全守衛（Guardian）模型。
你的任務是分析使用者提供的文字內容，識別並分類其中的敏感資訊與風險。

請嚴格遵循以下原則：
1. 準確識別 PII（personally identifiable information，個人可識別資訊）
2. 判斷內容是否包含不當、有害或違規資訊
3. 依照指令要求的格式輸出分類結果
4. 保持中立、客觀，僅根據內容本身進行判斷

輸出時請確保格式清晰、可被程式解析。"""


# =============================================================================
# Protocol / 抽象介面
# =============================================================================


class SystemPromptProvider(Protocol):
    """提供 system prompt 的介面。"""

    def get(self, model_id: str) -> str:
        """依 model_id 回傳對應的 system prompt。"""
        ...


class UserPromptTemplate(Protocol):
    """提供 user prompt 模板的介面。"""

    def format(self, **kwargs: Any) -> str:
        """以 kwargs 填入模板並回傳完整 user prompt。"""
        ...


# =============================================================================
# System Prompt 實作
# =============================================================================


class BaseSystemPromptProvider:
    """共用 system prompt，所有模型預設使用同一套。"""

    def __init__(self, base_prompt: str | None = None) -> None:
        self._base = base_prompt or GUARDIAN_SYSTEM_PROMPT_BASE
        self._model_overrides: dict[str, str] = {}

    def register_model_override(self, model_id: str, prompt: str) -> None:
        """註冊特定模型的 system prompt 覆寫。"""
        self._model_overrides[model_id] = prompt

    def get(self, model_id: str) -> str:
        """依 model_id 回傳 system prompt，若有註冊覆寫則用覆寫。"""
        return self._model_overrides.get(model_id, self._base)


# =============================================================================
# User Prompt 模板實作
# =============================================================================


@dataclass
class TaskTemplate:
    """單一任務的 user prompt 模板。"""

    task_id: str
    template: str
    required_keys: tuple[str, ...]

    def format(self, **kwargs: Any) -> str:
        missing = set(self.required_keys) - set(kwargs.keys())
        if missing:
            raise ValueError(
                f"Task '{self.task_id}' 缺少必要參數: {sorted(missing)}"
            )
        return self.template.format(**kwargs)


class UserPromptRegistry:
    """管理多個任務的 user prompt 模板。"""

    def __init__(self) -> None:
        self._templates: dict[str, TaskTemplate] = {}

    def register(self, template: TaskTemplate) -> None:
        """註冊任務模板。"""
        self._templates[template.task_id] = template

    def get(self, task_id: str) -> TaskTemplate | None:
        """取得任務模板。"""
        return self._templates.get(task_id)

    def format(self, task_id: str, **kwargs: Any) -> str:
        """取得並填入指定任務的 user prompt。"""
        tpl = self._templates.get(task_id)
        if tpl is None:
            raise ValueError(
                f"未知任務 '{task_id}'，已註冊: {list(self._templates.keys())}"
            )
        return tpl.format(**kwargs)


# =============================================================================
# 預設任務模板
# =============================================================================

PII_DETECTION_TEMPLATE = TaskTemplate(
    task_id="pii_detection",
    template="""請分析以下文字，識別其中的個人可識別資訊（PII），並依照指定格式輸出結果。

## 輸入文字
{content}

## 輸出要求
- 列出所有偵測到的 PII 類型及對應內容片段
- 若無 PII，請明確回覆「未偵測到 PII」
- 格式：以 JSON 或清楚結構化方式輸出""",
    required_keys=("content",),
)

PII_BINARY_CLASSIFICATION_TEMPLATE = TaskTemplate(
    task_id="pii_binary",
    template="""請判斷以下文字是否包含個人可識別資訊（PII）。

## 輸入文字
{content}

## 輸出要求
請只回覆「是」或「否」，表示是否含有 PII。""",
    required_keys=("content",),
)

CONTENT_SAFETY_TEMPLATE = TaskTemplate(
    task_id="content_safety",
    template="""請評估以下內容的安全性。

## 輸入內容
{content}

## 輸出要求
- 判斷是否為安全內容
- 若有不當之處，請說明類別與理由
- 格式：以 JSON 或清楚結構化方式輸出""",
    required_keys=("content",),
)

STREAM_SAFETY_TEMPLATE = TaskTemplate(
    task_id="stream_safety",
    template="""請即時判斷以下輸入片段是否安全、是否應繼續輸出。

## 輸入片段
{content}

## 輸出要求
- 回覆「safe」或「unsafe」
- 若為 unsafe，簡短說明原因""",
    required_keys=("content",),
)


# =============================================================================
# Prompt Factory（統一入口）
# =============================================================================


class PromptFactory:
    """
    Factory：依 model_id 與 task 組裝 system prompt 與 user prompt。
    """

    def __init__(
        self,
        system_provider: SystemPromptProvider | None = None,
        user_registry: UserPromptRegistry | None = None,
    ) -> None:
        self._system = system_provider or BaseSystemPromptProvider()
        self._user = user_registry or self._default_user_registry()

    @staticmethod
    def _default_user_registry() -> UserPromptRegistry:
        """建立預設的任務模板註冊表。"""
        registry = UserPromptRegistry()
        for tpl in [
            PII_DETECTION_TEMPLATE,
            PII_BINARY_CLASSIFICATION_TEMPLATE,
            CONTENT_SAFETY_TEMPLATE,
            STREAM_SAFETY_TEMPLATE,
        ]:
            registry.register(tpl)
        return registry

    def get_system_prompt(self, model_id: str) -> str:
        """取得指定模型的 system prompt。"""
        return self._system.get(model_id)

    def get_user_prompt(self, task_id: str, **kwargs: Any) -> str:
        """取得並填入指定任務的 user prompt。"""
        return self._user.format(task_id, **kwargs)

    def build(
        self,
        model_id: str,
        task_id: str,
        **user_kwargs: Any,
    ) -> dict[str, str]:
        """
        組裝完整 prompt 字典。

        Returns:
            {"system": str, "user": str}
        """
        return {
            "system": self.get_system_prompt(model_id),
            "user": self.get_user_prompt(task_id, **user_kwargs),
        }

    def register_task_template(self, template: TaskTemplate) -> None:
        """註冊新的任務模板。"""
        self._user.register(template)

    def register_model_system_prompt(self, model_id: str, prompt: str) -> None:
        """註冊特定模型的 system prompt（需為 BaseSystemPromptProvider）。"""
        if isinstance(self._system, BaseSystemPromptProvider):
            self._system.register_model_override(model_id, prompt)
        else:
            raise TypeError(
                "當前 SystemPromptProvider 不支援 register_model_override"
            )


# =============================================================================
# 便捷函式
# =============================================================================

_default_factory: PromptFactory | None = None


def get_factory() -> PromptFactory:
    """取得預設的 PromptFactory 單例。"""
    global _default_factory
    if _default_factory is None:
        _default_factory = PromptFactory()
    return _default_factory


def build_prompts(
    model_id: str,
    task_id: str,
    **user_kwargs: Any,
) -> dict[str, str]:
    """
    便捷函式：直接組裝 prompt。

    Example:
        prompts = build_prompts(
            model_id="Qwen/Qwen3Guard-Stream-8B",
            task_id="pii_detection",
            content="王小明，身份證 A123456789，住台北市...",
        )
    """
    return get_factory().build(model_id, task_id, **user_kwargs)
