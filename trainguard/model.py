"""Shared domain models for train-guard."""

from __future__ import annotations

from enum import Enum


class Action(str, Enum):
    FULL = "full"
    GENTLE = "gentle"
    STOP = "stop"
