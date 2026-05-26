# Heuristic Notes — _baseline

This is the canonical starting point for new experiments. Policy is a plain
greedy food-chaser: pick the safe action closest (manhattan) to food.

Do not run training rounds against this experiment directly.
Fork it instead: `python train.py --exp <new-name>`
