"""
Automated Protein, DNA and RNA structure modeling from cryo-EM maps
Tao Li et al.
"""


def main():
    import time
    import platform
    import importlib
    import em3dfold

    # Check platform
    if platform.system() != "Linux":
        print("# WARN Your system is -> {}".format(platform.system()))
        print("# WARN This program supports Linux systems (tested on CentOS 7)")
        print("# WARN This program will still run, but at any time it will crash")
        time.sleep(1)

    import argparse
    import warnings
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{getattr(em3dfold, '__version__')}",
    )

    # Suppress some warnings
    warnings.filterwarnings("ignore")

    modules = {
        "build": "em3dfold.build",
        "assemble": "em3dfold.pipeline.assemble",
        "eval": "em3dfold.pipeline.eval",
        "pred": "em3dfold.pipeline.pred",
        "qscore": "em3dfold.pipeline.get_qscore",
    }

    subparsers = parser.add_subparsers(title="Choose a module",)
    subparsers.required = "True"

    for key, module_path in modules.items():
        module = None
        module_error = None
        try:
            module = importlib.import_module(module_path)
        except Exception as exc:
            module_error = exc

        module_parser = subparsers.add_parser(
            key,
            description=(module.__doc__ if module is not None else f"Module {module_path} is currently unavailable."),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        if module is not None:
            if hasattr(module, "add_args"):
                module.add_args(module_parser)
            elif hasattr(module, "build_arg_parser"):
                temp_parser = module.build_arg_parser()
                for action in temp_parser._actions:
                    if action.dest == "help":
                        continue
                    kwargs = {
                        "default": action.default,
                        "help": action.help,
                        "required": action.required,
                    }
                    if action.metavar is not None:
                        kwargs["metavar"] = action.metavar
                    if action.choices is not None:
                        kwargs["choices"] = action.choices
                    if action.nargs is not None:
                        kwargs["nargs"] = action.nargs
                    if action.const is not None:
                        kwargs["const"] = action.const
                    if getattr(action, "type", None) is not None:
                        kwargs["type"] = action.type
                    if action.option_strings:
                        if action.__class__.__name__ in {"_StoreTrueAction", "_StoreFalseAction"}:
                            kwargs["action"] = "store_true" if action.__class__.__name__ == "_StoreTrueAction" else "store_false"
                            kwargs.pop("type", None)
                            kwargs.pop("const", None)
                            kwargs.pop("nargs", None)
                        module_parser.add_argument(*action.option_strings, **kwargs)
                    else:
                        module_parser.add_argument(action.dest, **kwargs)
            else:
                raise AttributeError(
                    "Module {} must define add_args(parser) or build_arg_parser().".format(
                        module_path
                    )
                )
            module_parser.set_defaults(func=module.main)
        else:
            def _raise_unavailable(_args, module_path=module_path, module_error=module_error):
                raise RuntimeError(
                    "Failed to import module {}: {}".format(module_path, module_error)
                )

            module_parser.set_defaults(func=_raise_unavailable)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()

