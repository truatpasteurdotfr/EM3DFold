"""Compatibility wrapper for complex build entrypoint."""

from em3dfold.build import add_args, main


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)
