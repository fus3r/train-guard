"""Pure power and thermal policy state machine."""

from __future__ import annotations

if __package__:
    from .config import PolicyConfig
    from .model import (
        Action,
        DecisionReason,
        Observation,
        PolicyDecision,
        PowerSource,
    )
else:  # Keep ``python trainguard/cli.py`` working from a checkout.
    from config import PolicyConfig
    from model import (
        Action,
        DecisionReason,
        Observation,
        PolicyDecision,
        PowerSource,
    )


class PolicyEngine:
    """Turn one observation into one decision.

    The engine owns the thermal hysteresis flag, so each supervisor carries its
    own cooldown state instead of sharing a process-wide one. It reads no
    sensor and touches no process, which makes every branch reachable from an
    observation built in memory.
    """

    def __init__(self, cooling: bool = False):
        self.cooling = cooling

    def _decide(self, action: Action, reason: DecisionReason) -> PolicyDecision:
        return PolicyDecision(action=action, reason=reason, cooling=self.cooling)

    def decide(self, config: PolicyConfig, observation: Observation) -> PolicyDecision:
        """Evaluate the rules in a fixed order and return the first match.

        A measurement this host did not provide skips only the rules that
        need it. The single exception is an active cooldown below, which no
        absence of evidence may end.
        """

        temperature = observation.temperature_c
        percent = observation.percent

        # An active cooldown outranks everything: without a reading we cannot
        # prove the pack has cooled, so the job stays paused.
        if self.cooling:
            if temperature is None or temperature > config.temp_resume_c:
                return self._decide(Action.STOP, DecisionReason.THERMAL_COOLDOWN)
            self.cooling = False

        if temperature is not None and temperature >= config.temp_pause_c:
            self.cooling = True
            return self._decide(Action.STOP, DecisionReason.THERMAL_LIMIT)

        if observation.source is PowerSource.BATTERY:
            if not config.run_on_battery:
                return self._decide(Action.STOP, DecisionReason.BATTERY_DISABLED)
            if percent is not None and percent <= config.battery_floor_pct:
                return self._decide(Action.STOP, DecisionReason.BATTERY_FLOOR)
            return self._decide(Action(config.battery_band), DecisionReason.BATTERY_POLICY)

        # On AC, the charging rule is checked before the generic warm rule: it
        # is the more specific one, so it names the reason when both match.
        if (
            observation.charging
            and percent is not None
            and percent < config.charge_cool_until_pct
            and temperature is not None
            and temperature >= config.temp_charge_gentle_c
        ):
            return self._decide(Action.GENTLE, DecisionReason.WARM_CHARGING)

        # A host with no battery climbs the same thermal ladder as one on AC;
        # only the reason it reports at the bottom of the ladder differs. A
        # separate scale would make the documented rule order untrue for it.
        if temperature is not None and temperature >= config.temp_gentle_c:
            return self._decide(Action.GENTLE, DecisionReason.WARM_AC)

        if observation.source is PowerSource.NO_BATTERY:
            return self._decide(Action(config.ac_band), DecisionReason.NO_BATTERY)

        return self._decide(Action(config.ac_band), DecisionReason.AC_POLICY)
