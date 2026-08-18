"""Tests for I3.5.1 treatment definitions and factorial conditions."""
import pytest
from hrm_adaptive_memory.executive.i3_5_1.conditions import (
    ConditionID, ExperimentalCondition, ObservationMode, GovernorMode,
    FROZEN_CONDITIONS, CONDITION_BY_ID, get_condition, all_condition_ids,
)


class TestFactorialConditions:
    def test_four_conditions_exist(self):
        assert len(FROZEN_CONDITIONS) == 4

    def test_all_condition_ids(self):
        ids = all_condition_ids()
        assert len(ids) == 4
        assert ConditionID.BLIND_NO_GOVERNOR in ids
        assert ConditionID.BLIND_GOVERNOR in ids
        assert ConditionID.AWARE_NO_GOVERNOR in ids
        assert ConditionID.AWARE_GOVERNOR in ids

    def test_blind_no_governor(self):
        c = get_condition(ConditionID.BLIND_NO_GOVERNOR)
        assert c.observation_mode == ObservationMode.BLIND
        assert c.governor_enabled is False

    def test_blind_governor(self):
        c = get_condition(ConditionID.BLIND_GOVERNOR)
        assert c.observation_mode == ObservationMode.BLIND
        assert c.governor_enabled is True

    def test_aware_no_governor(self):
        c = get_condition(ConditionID.AWARE_NO_GOVERNOR)
        assert c.observation_mode == ObservationMode.AWARE
        assert c.governor_enabled is False

    def test_aware_governor(self):
        c = get_condition(ConditionID.AWARE_GOVERNOR)
        assert c.observation_mode == ObservationMode.AWARE
        assert c.governor_enabled is True

    def test_conditions_are_frozen(self):
        """Conditions must be immutable."""
        import dataclasses
        for c in FROZEN_CONDITIONS:
            assert dataclasses.is_dataclass(c)
            with pytest.raises((FrozenInstanceError, Exception)):
                c.governor_enabled = True  # type: ignore

    def test_condition_as_dict(self):
        c = get_condition(ConditionID.AWARE_GOVERNOR)
        d = c.as_dict()
        assert d["condition_id"] == "AWARE_GOVERNOR"
        assert d["observation_mode"] == "AWARE"
        assert d["governor_enabled"] is True

    def test_two_factors_independent(self):
        """S and G factors are independent — all 4 combinations exist."""
        combos = {(c.observation_mode, c.governor_enabled) for c in FROZEN_CONDITIONS}
        assert len(combos) == 4
        assert (ObservationMode.BLIND, False) in combos
        assert (ObservationMode.BLIND, True) in combos
        assert (ObservationMode.AWARE, False) in combos
        assert (ObservationMode.AWARE, True) in combos


try:
    from dataclasses import FrozenInstanceError
except ImportError:
    FrozenInstanceError = AttributeError
