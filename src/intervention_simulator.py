"""
Intervention Simulator - What-if projections for risk reduction.
Simple, transparent model that estimates how targeted interventions
could change addiction risk over time.
"""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass(frozen=True)
class Intervention:
    name: str
    weekly_effect: float
    description: str


DEFAULT_INTERVENTIONS: Dict[str, Intervention] = {
    "Screen-Time Limit": Intervention(
        name="Screen-Time Limit",
        weekly_effect=0.035,
        description="Set daily caps and enforce with OS-level controls."
    ),
    "Notification Detox": Intervention(
        name="Notification Detox",
        weekly_effect=0.02,
        description="Disable non-essential notifications for 4 weeks."
    ),
    "Sleep Hygiene": Intervention(
        name="Sleep Hygiene",
        weekly_effect=0.025,
        description="No phone 60 minutes before bed, consistent sleep window."
    ),
    "App Blocking": Intervention(
        name="App Blocking",
        weekly_effect=0.03,
        description="Block top addictive apps during focus hours."
    ),
    "Mindfulness Practice": Intervention(
        name="Mindfulness Practice",
        weekly_effect=0.015,
        description="Short daily mindfulness sessions to reduce stress-driven use."
    )
}


def simulate_interventions(
    baseline_risk: float,
    selected: List[str],
    weeks: int = 8
) -> Dict[str, List[float]]:
    """
    Simulate addiction risk over time under chosen interventions.
    The model uses additive weekly reductions with diminishing returns.
    """
    baseline = float(np.clip(baseline_risk, 0.0, 1.0))
    effects = [DEFAULT_INTERVENTIONS[name].weekly_effect for name in selected]
    total_effect = sum(effects)

    trajectory = [baseline]
    for week in range(1, weeks + 1):
        decay = 1.0 / (1.0 + 0.15 * week)
        new_score = trajectory[-1] * (1.0 - total_effect * decay)
        trajectory.append(float(np.clip(new_score, 0.0, 1.0)))

    return {
        "weeks": list(range(0, weeks + 1)),
        "risk": trajectory
    }
