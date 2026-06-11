# -*- coding: utf-8 -*-


__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"

"""
BayFloodGEN Runtime utilities.

This script contains functions for detecting the runtime environment (e.g., GPU availability)
and selecting the appropriate sampler based on user preference and available libraries.
"""

from model_code.baygen_config import PREFERRED_SAMPLER


# ─────────────────────────────────────────────
# DETECTION OF GPU / JAX
# ─────────────────────────────────────────────
def detect_jax_devices():
    r"""It detects if JAX is available and if it can access GPU devices.
    This is useful to decide whether to use JAX-based samplers (numpyro,
    blackjax) or fallback to PyMC's CPU sampler.
    """

    try:
        import jax
        devices = jax.devices()
        gpu_devs = [d for d in devices if d.platform == "gpu"]
        if gpu_devs:
            print(f"  JAX: {len(gpu_devs)} GPU(s) detected: {gpu_devs}")
        else:
            print(f"  JAX: no GPU, using {devices[0].platform.upper()}")
        return bool(gpu_devs)
    except ImportError:
        print("  JAX not available.")
        return False


def select_sampler():
    r"""It returns the best available sampler based on the presence of JAX
    and user preference.
    """

    if PREFERRED_SAMPLER in ("numpyro", "blackjax"):
        try:
            if PREFERRED_SAMPLER == "numpyro":
                import numpyro  # noqa
            else:
                import blackjax  # noqa
            print(f"  Sampler seleccionado: {PREFERRED_SAMPLER} (JAX backend)")
            return PREFERRED_SAMPLER
        except ImportError:
            print(f"  {PREFERRED_SAMPLER} not installed. Fallback to standard PyMC.")
    print("  Sampler selected: pymc (CPU)")
    return "pymc"
