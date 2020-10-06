# -*- coding: utf-8 -*-
""" Simply defines a function :func:`with_proba` that is used everywhere.
"""
from random import random


def with_proba(epsilon):
    # True with proba epsilon
    return random() < epsilon  