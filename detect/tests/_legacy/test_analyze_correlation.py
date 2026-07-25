from scripts.analyze_correlation import MESSAGE, OBJECTIVE, TASK_KIND, describe_task, main

def test_describe_task_mentions_kind_and_objective() -> None:
    text = describe_task()
    assert TASK_KIND in text
    assert OBJECTIVE in text

def test_main_returns_message() -> None:
    assert main() == MESSAGE