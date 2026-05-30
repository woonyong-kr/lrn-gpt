# -*- coding: utf-8 -*-
"""작은 입력 검증 helper."""

from typing import NoReturn


def require(condition: bool, message: object, error_type: type[Exception] = ValueError) -> None:
    """condition이 False이면 지정한 예외를 발생시킵니다."""
    if not condition:
        raise error_type(message)


def fail(message: object, error_type: type[Exception] = ValueError) -> NoReturn:
    """분기 끝에서 명시적으로 예외를 발생시킵니다."""
    raise error_type(message)
