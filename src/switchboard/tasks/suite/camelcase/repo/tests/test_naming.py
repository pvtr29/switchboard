from naming import snake_to_camel


def test_basic() -> None:
    assert snake_to_camel("user_id") == "userId"


def test_three_parts() -> None:
    assert snake_to_camel("max_retry_count") == "maxRetryCount"
