"""Command-line entry point for the Buying the Dip analysis pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from buying_the_dip import (
    AnalysisPipeline,
    HtmlReportBuilder,
    IndexDataLoader,
    load_config,
)
from buying_the_dip.config import AppConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run multi-index DCA versus Buying the Dip analyses and build HTML reports."
        )
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def configure_logging(log_level: str) -> None:
    """Configure application logging."""

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def resolve_relative_paths(config: AppConfig, config_path: Path) -> None:
    """Resolve relative paths from the configuration file directory."""

    config_directory = config_path.resolve().parent

    if not Path(config.project.output_directory).is_absolute():
        config.project.output_directory = str(
            (config_directory / config.project.output_directory).resolve()
        )

    if not Path(config.report.output_directory).is_absolute():
        config.report.output_directory = str(
            (config_directory / config.report.output_directory).resolve()
        )

    for source in config.indices:
        if source.path and not Path(source.path).is_absolute():
            source.path = str((config_directory / source.path).resolve())
        if source.cache_path and not Path(source.cache_path).is_absolute():
            source.cache_path = str((config_directory / source.cache_path).resolve())


def main() -> None:
    """Run the full analysis pipeline."""

    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    resolve_relative_paths(config, config_path)
    configure_logging(config.project.log_level)

    try:
        data_loader = IndexDataLoader(config)
        loaded_sources = data_loader.load_many(config.indices)

        pipeline = AnalysisPipeline(config)
        results = pipeline.run(loaded_sources)

        report_builder = HtmlReportBuilder(config)
        report_path = report_builder.build(results)
    except Exception as error:
        logging.getLogger(__name__).error("%s", error)
        raise SystemExit(f"Execution failed: {error}") from None

    print(f"Analysis completed for {len(results)} index source(s).")
    print(f"Main HTML report: {report_path}")
    print(f"Index HTML reports directory: {report_path.parent}")
    print(f"Raw outputs directory: {config.output_directory}")


if __name__ == "__main__":
    main()
