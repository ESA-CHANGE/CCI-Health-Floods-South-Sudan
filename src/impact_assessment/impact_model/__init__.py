# -*- coding: utf-8 -*-
__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"

"""
Modular impact assessment package.

This package organizes the impact assessment workflow into independent modules:
- configuration
- vulnerability engine
- processing/aggregation
- analysis
- plotting
- sensitivity
- pipeline orchestration
"""

from impact_model.config import *
from impact_model.engine import *
from impact_model.processing import *
from impact_model.analysis import *
from impact_model.plotting import *
from impact_model.sensitivity import *
from impact_model.calibration import *
from impact_model.pipeline import *
