"""Runnable demo: build the default scenario, run it, print a report.

    python -m pd_throughput_simulator_v5.demo
"""

from __future__ import annotations

from .metrics import build_report, format_report
from .scenario import default_demo_scenario
from .simulator import PDSimulator


def main() -> None:
    scenario = default_demo_scenario()
    sim = PDSimulator(scenario)
    states = sim.run()
    report = build_report(states, scenario.metrics)
    print(format_report(report))


if __name__ == "__main__":
    main()
