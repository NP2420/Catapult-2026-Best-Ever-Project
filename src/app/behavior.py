from __future__ import annotations

from dataclasses import dataclass, field

from .models import BreakState, DROWSY_SCORE_THRESHOLD, TIRED_SCORE_THRESHOLD, clamp_score


@dataclass(slots=True)
class BehaviorDecision:
    should_switch_song: bool = False
    should_refresh_queue: bool = False
    break_state: BreakState = field(default_factory=BreakState)


class ProductivityEngine:
    def __init__(
        self,
        short_term_threshold_seconds: int = 15,
        long_term_threshold_seconds: int = 90,
        minimum_break_seconds: int = 300,
    ) -> None:
        self.short_term_threshold_seconds = short_term_threshold_seconds
        self.long_term_threshold_seconds = long_term_threshold_seconds
        self.minimum_break_seconds = minimum_break_seconds

        self._fatigue_seconds = 0.0
        self._switch_armed = True
        self._break_state = BreakState()

    @property
    def fatigue_seconds(self) -> float:
        return self._fatigue_seconds

    @property
    def break_state(self) -> BreakState:
        return self._break_state

    def tick(self, ema_score: float, elapsed_seconds: float) -> BehaviorDecision:
        score = clamp_score(ema_score)

        if self._break_state.active:
            self._break_state.seconds_remaining = max(0, self._break_state.seconds_remaining - int(elapsed_seconds))
            self._break_state.can_resume = self._break_state.seconds_remaining == 0
            return BehaviorDecision(break_state=self._break_state)

        if score >= TIRED_SCORE_THRESHOLD:
            self._fatigue_seconds += elapsed_seconds
        elif score >= DROWSY_SCORE_THRESHOLD:
            score_span = TIRED_SCORE_THRESHOLD - DROWSY_SCORE_THRESHOLD
            scaled_increase = 0.4 + (0.6 * ((score - DROWSY_SCORE_THRESHOLD) / max(score_span, 1e-6)))
            self._fatigue_seconds += elapsed_seconds * scaled_increase
        else:
            recovery_scale = 1.0 + ((DROWSY_SCORE_THRESHOLD - score) / max(DROWSY_SCORE_THRESHOLD, 1e-6))
            self._fatigue_seconds = max(0.0, self._fatigue_seconds - (elapsed_seconds * recovery_scale))
            self._switch_armed = True

        should_switch = False
        if self._fatigue_seconds >= self.short_term_threshold_seconds and self._switch_armed:
            should_switch = True
            self._switch_armed = False

        if self._fatigue_seconds >= self.long_term_threshold_seconds:
            self._break_state = BreakState(active=True, seconds_remaining=self.minimum_break_seconds, can_resume=False)
            self._fatigue_seconds = 0.0
            self._switch_armed = True

        return BehaviorDecision(
            should_switch_song=should_switch,
            should_refresh_queue=should_switch,
            break_state=self._break_state,
        )

    def resume(self) -> None:
        if self._break_state.can_resume:
            self._break_state = BreakState(active=False, seconds_remaining=0, can_resume=False)
            self._fatigue_seconds = 0.0
            self._switch_armed = True
