"""I3.5.1 Governor Causal Protocol & Artifact Closure Repair.

This package implements the factorial 2x2 experiment design:
  - Factor S: cognitive-state visibility (BLIND, AWARE)
  - Factor G: governor availability (OFF, ON)

Four conditions:
  - BLIND_NO_GOVERNOR  (C00)
  - BLIND_GOVERNOR     (C01)
  - AWARE_NO_GOVERNOR  (C10)
  - AWARE_GOVERNOR     (C11)

The primary scientific question is whether the governor reduces
decision gap beyond what is explained by additional observable
information.
"""
