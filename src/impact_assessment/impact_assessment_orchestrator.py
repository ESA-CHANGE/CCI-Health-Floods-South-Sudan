# -*- coding: utf-8 -*-

__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"

"""
CHANGE: impact assessment orchestrator
===========================================================================

Orchestrator entrypoint for the modular impact assessment workflow.
"""

from impact_model.pipeline import run_all

if __name__ == "__main__":
    run_all(parts_dir="../data/model_output/by_facility")
