"""Tool-Integrated Reasoning extension for the RLOO Countdown trainer.

This package adds:
- A pool of callable tools (relevant + irrelevant) that the policy can invoke.
- A failure database that collects failed Countdown rollouts.
- A DSPy-based analyzer that recommends which tools should be enabled.
- A failure-mode-aware reward that bonuses correct tool selection.
- An SFT warm-start over tool-using trajectories.
- An RLOO trainer subclass that wires everything together.
"""
