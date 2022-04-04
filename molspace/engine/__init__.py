"""
The engine is the show runner, it houses our training loops,
analysis and visualization pipelines, and all the other tooling
which makes our workflow keep running.

It's responsible for all debugging and analysis end, with Tensorboard
and WandB connections, 2-way interactions attempts, etc.
"""

import molspace.engine.trainer
import molspace.engine.constrastive
