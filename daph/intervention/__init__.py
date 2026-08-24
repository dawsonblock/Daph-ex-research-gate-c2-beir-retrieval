"""DAPH intervention package — state checkpointing and causal action data collection.

This package provides:
  - checkpoint.py: Snapshot a state before action selection
  - restore.py: Reconstruct a state from a checkpoint
  - schedule.py: Frozen intervention schedules
  - force_action.py: Force a specific action from a checkpoint
  - receipts.py: Provenance receipts for interventions
"""
