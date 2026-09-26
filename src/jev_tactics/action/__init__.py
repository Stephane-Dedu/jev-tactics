"""Couche d'action : du plan aux gestes.

La geometrie etant deja resolue par la calibration, le clic est trivial ; le sujet de
conception est le TIMING (delais log-normaux, trajectoires courbes). L'interface
`InputBackend` permet de tout tester sans toucher la souris (`DryRunBackend`).
"""

from jev_tactics.action.actions import END_TURN_KEY, Executor
from jev_tactics.action.mouse import (
    DirectInputBackend,
    DryRunBackend,
    InputBackend,
    bezier_path,
    move_along_curve,
)
from jev_tactics.action.timing import Delay, DelaySampler

__all__ = [
    "END_TURN_KEY", "Delay", "DelaySampler", "DirectInputBackend", "DryRunBackend",
    "Executor", "InputBackend", "bezier_path", "move_along_curve",
]
