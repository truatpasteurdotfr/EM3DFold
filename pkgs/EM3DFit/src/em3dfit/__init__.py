"""em3dfit: EM3D fitting utilities."""

from em3dfit.log_utils import install_exception_flush, install_package_print

__all__ = ["__version__"]

__version__ = "0.1.0"

install_package_print(
    target_prefixes=("em3dfit",),
    helper_modules=("em3dfit.log_utils", "em3dfit.utils"),
)
install_exception_flush()
