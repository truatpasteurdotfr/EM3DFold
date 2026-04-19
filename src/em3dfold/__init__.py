from em3dfold.utils.log_utils import install_exception_flush, install_package_print

__version__ = "1.1.0"

install_package_print(
    target_prefixes=("em3dfold",),
    excluded_prefixes=("em3dfold.rinalmo", "em3dfold.bin.src"),
    helper_modules=("em3dfold.utils.log_utils", "em3dfold.utils.misc_utils"),
)
install_exception_flush()

