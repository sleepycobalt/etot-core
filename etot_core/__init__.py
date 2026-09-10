"""
etot_core — shared machinery for ETOT agentic loops.

    llm.py      thin Claude client: one call, structured JSON out, usage tracked
    logger.py   per-run directory with every prompt, response, and metric saved
    loop.py     generic plan -> act -> check -> revise -> stop controller
    config.py   load YAML config with model roles and critic rules

Tools (synth, critique, ...) import from here and never talk to the API
directly, so every run is logged the same way and every eval is comparable.
"""
