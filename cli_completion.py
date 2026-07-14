#!/usr/bin/env python3
"""Optional argcomplete integration shared by command-line scripts."""

import argparse


def parse_args_with_completion(parser: argparse.ArgumentParser) -> argparse.Namespace:
    try:
        import argcomplete
    except ImportError:
        pass
    else:
        argcomplete.autocomplete(parser)

    return parser.parse_args()
