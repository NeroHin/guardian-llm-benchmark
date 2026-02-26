from prompt import build_prompts


def test_pii_binary_prompt_requires_structured_json() -> None:
    prompts = build_prompts(
        model_id="granite4:1b-h",
        task_id="pii_binary",
        content="王小明身分證字號是A123456789。",
    )

    user_prompt = prompts["user"]
    assert "只輸出一個可解析的 JSON 物件" in user_prompt
    assert '"contains_pii"' in user_prompt
    assert '"label"' in user_prompt
    assert '"confidence"' in user_prompt
    assert '"reason"' in user_prompt
