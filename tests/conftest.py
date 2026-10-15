import logging

logging.getLogger("skfem").setLevel(logging.WARNING)
try:
    import bempp_cl.api as _b
    _b.set_logging_level("warning")
except ImportError:
    pass
