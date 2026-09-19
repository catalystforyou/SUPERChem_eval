"""Argument types shared by answer-generation entry points; no API imports."""
import argparse


def parse_bool(value):
    """Accept explicit true/false and the historical empty text-only argument."""
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in ('true', '1', 'yes'):
        return True
    if lowered in ('false', '0', 'no', ''):
        return False
    raise argparse.ArgumentTypeError('Expected true or false')
