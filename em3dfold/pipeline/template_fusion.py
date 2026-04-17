"""Compatibility wrapper around the formal EM3DFold assemble pipeline."""

import argparse

from em3dfold.pipeline.assemble import add_args, main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the EM3DFold protein-template + initial-structure fusion pipeline.",
    )
    add_args(parser)
    main(parser.parse_args())

