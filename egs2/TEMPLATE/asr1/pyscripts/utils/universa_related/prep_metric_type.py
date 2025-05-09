#!/usr/bin/env python3

import argparse
import logging

from espnet2.fileio.metric_scp import MetricReader
from espnet2.fileio.read_text import read_2columns_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Prepare metric ID")
    parser.add_argument("metric_scp", type=str, help="metric.scp information")
    parser.add_argument("metric2type", type=str, help="output metric2type")
    parser.add_argument(
        "--reading_size",
        type=int,
        default=-1,
        help="reading size (for efficient loading)",
    )
    args = parser.parse_args()

    metric_reader = MetricReader(args.metric_scp)
    reading_size = args.reading_size if args.reading_size > 0 else len(metric_reader)
    metric2type = dict()
    with open(args.metric2type, "w") as f:
        row_num = 0
        for key, metric in metric_reader.items():
            for k, v in metric.items():
                # NOTE(jiatong): specifically related to VERSA setup
                if k == "key" or "hyp_text" in k:
                    continue
                if k in metric2type.keys():
                    continue
                if type(v) == str:
                    metric2type[k] = "categorical"
                elif type(v) == int or type(v) == float:
                    metric2type[k] = "numerical"
                else:
                    logging.warning(f"Unknown metric type for {k}: {v}")
                    continue
            row_num += 1
            if row_num > reading_size:
                print(f"Reading size reached {reading_size}, stop reading.")

        for k, v in metric2type.items():
            f.write(f"{k} {v}\n")
    logging.info(f"metric2type: {args.metric2type}")
