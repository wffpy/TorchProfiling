import argparse
import pathlib

from ..percision.compare_persion import compare
from ..perf.analysis import (
    AtenOpAnalyzer,
    DistAnalyzer,
    gen_module_compare_table_str,
    gen_module_compare_tables,
)
from .cut_log import extract_section


def parse_args():
    """
    Parse the input arguments

    Returns:
        void
    """

    arg_parser = argparse.ArgumentParser(
        description="Module Logging command line tools.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # arg_parser.add_argument("--c", type=pathlib.Path, help="path to XPU log file")
    arg_parser.add_argument("--csv", action="store_true", help="write tables to csv files")

    arg_parser.add_argument("--all", action="store_true", help="generate all tables")

    arg_parser.add_argument("--detail", action="store_true", help="generate detail table")

    arg_parser.add_argument("--summary", action="store_true", help="generate summary table")

    arg_parser.add_argument("--total", action="store_false", help="generate total table")

    arg_parser.add_argument("--path", type=pathlib.Path, help="path to XPU log file")

    arg_parser.add_argument("--compare", action="store_true", help="generate summary table")

    arg_parser.add_argument("--lhs_path", type=pathlib.Path, help="path to log file")

    arg_parser.add_argument("--rhs_path", type=pathlib.Path, help="path to log file")

    arg_parser.add_argument("--cut_log", action="store_true", help="split log")

    arg_parser.add_argument("--dist", action="store_true", help="analysis distributed ops")

    arg_parser.add_argument(
        "--begin",
        action="store",
        type=str,
        default="iteration        2",
        help="path to log file",
    )

    arg_parser.add_argument(
        "--end",
        action="store",
        type=str,
        default="iteration        3",
        help="path to log file",
    )
    arg_parser.add_argument("--percision", action="store_true", help="generate summary table")

    return arg_parser.parse_args()


def write_table(table, table_name=None, csv=False):
    """
    Function:
    write table to csv file or print to stdout
    """
    if csv and table_name:
        with open("/tmp/{}.csv".format(table_name), "w") as f:
            f.write(table.get_string())
            f.close()
    else:
        print(table)


def parse_log():
    """
    Parse the input arguments and run the analysis.
    Returns:
        void
    """
    args = parse_args()
    if args.cut_log:
        print("args.begin: {}".format(args.begin))
        print("args.end: {}".format(args.end))
        extract_section(args.path, args.begin, args.end)
    elif args.dist:
        analyzer = DistAnalyzer(args.path)
        analyzer.analysis()
        if args.all:
            s_table = analyzer.gen_summary_table()
            d_table = analyzer.gen_detail_table()
            write_table(s_table, "summary", args.csv)
            write_table(d_table, "detail", args.csv)
        else:
            if args.summary:
                s_table = analyzer.gen_summary_table()
                write_table(s_table, "summary", args.csv)
                # write_table(s_table, "summary", args.csv)
            if args.detail:
                d_table = analyzer.gen_detail_table()
                write_table(d_table, "detail", args.csv)

    elif args.percision:
        table = compare(args.lhs_path, args.rhs_path)
        print(table)

    elif not args.compare:
        analyzer = AtenOpAnalyzer(args.path)
        analyzer.analysis()
        if args.all:
            s_table = analyzer.gen_summary_table()
            d_table = analyzer.gen_detail_table()
            t_table = analyzer.gen_total_time_table()
            write_table(s_table, "summary", args.csv)
            write_table(d_table, "detail", args.csv)
            write_table(t_table, "total", args.csv)
        else:
            if args.total:
                t_table = analyzer.gen_total_time_table()
                write_table(t_table, "total", args.csv)
            if args.summary:
                s_table = analyzer.gen_summary_table()
                write_table(s_table, "summary", args.csv)
            if args.detail:
                d_table = analyzer.gen_detail_table()
                write_table(d_table, "detail", args.csv)

    elif args.compare and args.lhs_path and args.rhs_path:
        analyzer1 = AtenOpAnalyzer(args.lhs_path)
        analyzer2 = AtenOpAnalyzer(args.rhs_path)
        if args.csv:
            table_str = gen_module_compare_table_str(analyzer1, analyzer2)
            with open("/tmp/compare.csv", "w") as f:
                f.write(table_str)
                f.close()
        else:
            tables = gen_module_compare_tables(analyzer1, analyzer2)
            for table in tables:
                print(table)
