# The script used to preprocess single column plain dataset
import re
import math
import numpy as np
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool
from argparse import ArgumentParser
from tiktoken import get_encoding
from sortedcontainers import SortedDict
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import write_html
from pyarrow.parquet import ParquetWriter
from pyarrow import Table, string, schema
from pandas import DataFrame, read_parquet
from hellolm.utils.logger import logger, setup_logger
from hellolm.utils.tools import generate_run_id, ensure_directory


# multiple processes unit
def prepare_unit(
    chunk_data,
    tokenizer_name: str,
    column_name: str,
    threshold: int,
    chunk_start: int,
    df_idx: int,
    # additional tasks
    removes: str,
):
    try:
        tokenizer = get_encoding(tokenizer_name)
    except Exception as e:
        logger.exception(e)
        raise RuntimeError(f"No valid tiktoken encoding matches {tokenizer_name}")

    result: list[(int, int, int, str)] = []

    for idx, text in enumerate(chunk_data[column_name]):
        if not isinstance(text, str) or not text.strip():
            continue

        # removes some known unused (noise) text pattern in particular dataset (like wikitext)
        if removes:
            text = re.sub(removes, "", text)

        # add positional tokens
        text += "<|endoftext|>"

        token_count = len(tokenizer.encode(text, allowed_special={"<|endoftext|>"}))

        if token_count >= threshold:
            result.append((df_idx, chunk_start + idx, token_count, text))

    return result


# multiple processes unit
def combine_texts(bin_item):
    return "\n".join(bin_item[1])


def plot_distribution(payload: dict[str, list], run_id: str):
    # build plot titles
    titles = []
    for prefix, _ in payload.items():
        titles.append(f"Distribution of <{prefix}> group")

    fig = make_subplots(rows=math.ceil(len(payload) / 2), cols=2, subplot_titles=titles)
    for idx, (prefix, data) in enumerate(payload.items()):
        # Calculate statistics
        mean = np.mean(data)
        median = np.median(data)
        min_val = np.min(data)
        max_val = np.max(data)
        std_dev = np.std(data)

        # Add histogram
        fig.add_trace(
            go.Histogram(
                name=prefix,
                x=data,
                nbinsx=50,
                showlegend=True,
            ),
            row=(idx // 2) + 1,
            col=(idx % 2) + 1,
        )

        # Add statistics as annotations
        stats_text = (
            f"Mean: {mean:.2f}<br>"
            f"Median: {median:.2f}<br>"
            f"Min: {min_val}<br>"
            f"Max: {max_val}<br>"
            f"Std Dev: {std_dev:.2f}"
        )
        fig.add_annotation(
            text=stats_text,
            showarrow=False,
            align="left",
            xref="x domain",
            yref="y domain",
            x=0.95,
            y=0.95,
            font=dict(size=10),
            row=(idx // 2) + 1,
            col=(idx % 2) + 1,
        )

    fig.update_layout(
        title="Token Distribution Histograms",
        template="plotly_white",
        showlegend=True,
    )

    # Set x and y axis titles for each subplot
    for i in range(1, len(payload) + 1):
        fig.update_xaxes(
            title_text="Token Count", row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1
        )
        fig.update_yaxes(
            title_text="Frequency", row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1
        )

    # resolve path and save
    viz_path = "plots/preprocessed"
    plots_path = (
        ensure_directory(viz_path) / f"token_distribution_histogram_{run_id}.html"
    )
    write_html(fig, plots_path)
    logger.success(f"Plots saved to {plots_path}")


@logger.catch
def _main(
    inputs: str,
    prefix: str,
    output_path: str,
    removes: str,
    run_id: str,
    tokenizer_name: str,
    column_name: str,
    threshold: int,
    target_length: int,
    num_processes: int = None,
):
    # debug insight
    logger.debug(f"Target inputs path: {inputs}")
    logger.debug(f"Prefix to match: {prefix}")
    logger.debug(f"Output path: {output_path}")
    logger.debug(f"Used tokenizer: {tokenizer_name}")
    logger.debug(f"Threshold to filter out raw data: {threshold}")
    logger.debug(f"Target length: {target_length}")
    logger.debug(f"Using processes: {num_processes}")

    # parse arguments
    if isinstance(inputs, str):
        inputs = [inputs]

    # resolve
    resolved_path: list[Path] = []
    for item in map(Path, inputs):
        item = item.absolute()
        if item.is_dir():
            resolved_path.extend(
                file for file in item.rglob(f"{prefix}*") if file.is_file()
            )
        elif item.is_file() and item.name.startswith(prefix):
            resolved_path.append(item)

    # validate resolved_path
    if len(resolved_path) == 0:
        raise FileNotFoundError("Nothing input file(s) resolved")

    # load parquet files
    logger.info("Loading datasets")
    loaded_data: list[DataFrame] = []
    loaded_count = 0
    for index, entries in enumerate(resolved_path):
        logger.trace(f"Loading {index}/{len(resolved_path)} file: {entries}")
        try:
            loaded_data.append(read_parquet(path=entries))
            loaded_count += 1
            logger.trace(f"Loaded {index}/{len(resolved_path)} file")
        except Exception as e:
            logger.exception(e)
            raise ValueError(f"File {entries} is not a valid parquet file")
    logger.success(f"Datasets loading successful, loaded {loaded_count} file(s)")

    # process in batches
    batch_size = 10000
    text_metadata = []
    all_texts = {}

    logger.info(f"Counting tokens and filter texts using {num_processes} processes")
    count_start_time = datetime.now()
    with Pool(processes=num_processes) as pool:
        for df_idx, df in enumerate(loaded_data):
            logger.trace(f"Counting dataframe {df_idx + 1}/{loaded_count}")
            if column_name not in df.columns:
                logger.warning(
                    f"Dataframe {df_idx + 1} missing {column_name} column, skipping"
                )
                continue

            chunks = []

            for chunk_start in range(0, len(df), batch_size):
                chunk = df.iloc[chunk_start : chunk_start + batch_size]
                chunks.append(
                    (
                        chunk,
                        tokenizer_name,
                        column_name,
                        threshold,
                        chunk_start,
                        df_idx,
                        removes,
                    )
                )

            results = pool.starmap(prepare_unit, chunks)

            for chunk_results in results:
                for df_idx, row_idx, token_count, text in chunk_results:
                    text_metadata.append((df_idx, row_idx, token_count))
                    all_texts[(df_idx, row_idx)] = text

            logger.success(
                f"Dataframe {df_idx + 1}/{loaded_count} successfully counted"
            )

    # resolve text_metadata
    if len(text_metadata) > 0:
        logger.success(
            f"Found {len(text_metadata)} text segments with at least {threshold} tokens "
            + f"in {(datetime.now() - count_start_time).total_seconds()} sec"
        )
    else:
        logger.warning("No any desired text found in provided dataset, stop processing")
        return

    # sort by token count
    text_metadata.sort(key=lambda x: x[2], reverse=True)

    # bin packing
    logger.info(f"Applying bin packing to target length of {target_length}")
    bins: list[tuple[int, list[str]]] = []
    remaining_space = SortedDict()
    metadata_len = len(text_metadata)
    pack_start_time = datetime.now()
    for item_idx, (df_idx, row_idx, token_count) in enumerate(text_metadata):
        # access original text
        text = all_texts[(df_idx, row_idx)]

        # find best bin
        best_bin_idx = -1
        pos = remaining_space.bisect_left(token_count)

        if pos < len(remaining_space):
            best_space = remaining_space.keys()[pos]
            best_bin_idx = remaining_space[best_space][0]

            remaining_space[best_space].pop(0)
            if not remaining_space[best_space]:
                del remaining_space[best_space]

            # add to existing bin
            bins[best_bin_idx][0] += token_count
            bins[best_bin_idx][1].append(text)

            new_remaining = target_length - bins[best_bin_idx][0]
            if new_remaining > 0:
                if new_remaining not in remaining_space:
                    remaining_space[new_remaining] = []
                remaining_space[new_remaining].append(best_bin_idx)
        else:
            # create a new bin
            new_bin_idx = len(bins)
            bins.append([token_count, [text]])

            new_remaining = target_length - token_count
            if new_remaining > 0:
                if new_remaining not in remaining_space:
                    remaining_space[new_remaining] = []
                remaining_space[new_remaining].append(new_bin_idx)

        # insight
        if (item_idx + 1) % 50000 == 0 or item_idx == metadata_len - 1:
            logger.trace(f"Applied to {item_idx + 1}/{metadata_len} rows")

    logger.complete()
    logger.success(
        f"Created {len(bins)} grouped text segments "
        + f"{metadata_len / (datetime.now() - pack_start_time).total_seconds()} rows/sec"
    )

    # filter bins that are below 95% of target length
    min_tokens_threshold = int(target_length * 0.95)
    original_bin_count = len(bins)
    bins = [bin for bin in bins if bin[0] >= min_tokens_threshold]
    filtered_bin_count = original_bin_count - len(bins)
    logger.info(
        f"Filtered out {filtered_bin_count} bins below {min_tokens_threshold} tokens (95% of target length)"
    )
    logger.success(f"Remaining {len(bins)} bins after filtering")

    # plot
    logger.info("Plotting grouped token numbers distribution")
    plot_distribution(
        {
            "raw": [item[2] for item in text_metadata],
            "processed": [item[0] for item in bins],
        },
        run_id=run_id,
    )

    # combine texts
    logger.info(f"Combining texts in {num_processes} processes")
    with Pool(processes=num_processes) as pool:
        final_texts = pool.map(combine_texts, bins)

    # output path resolve
    output_path = ensure_directory(output_path)
    output_path = output_path / f"{prefix}_processed.parquet"
    logger.complete()
    logger.info(f"Preprocessed data will write to {output_path}")

    # write to memory by batches
    batch_size = 1000

    # create writer
    writer = ParquetWriter(where=output_path, schema=schema([(column_name, string())]))
    for idx in range(0, len(final_texts), batch_size):
        batch = final_texts[idx : idx + batch_size]
        batch_df = DataFrame({"text": batch})

        writer.write_table(Table.from_pandas(batch_df))
    logger.success(f"Successfully wrote processed data to {output_path}")


if __name__ == "__main__":
    # arguments settings
    parser = ArgumentParser(
        description="utils to preprocess one column plain text dataset"
    )

    # file option
    file_options = parser.add_argument_group("file options")
    file_options.add_argument(
        "--input", "-i", type=str, help="input file or path to input files"
    )
    file_options.add_argument(
        "--prefix", "-p", type=str, help="match files by prefix (as filter)"
    )
    file_options.add_argument(
        "--output", "-o", type=str, help="target path to store processed files"
    )

    # process options
    process_options = parser.add_argument_group("process options")
    process_options.add_argument(
        "--tokenizer",
        "-k",
        type=str,
        help="specify a valid tiktoken encoding name",
        default="cl100k_base",
    )
    process_options.add_argument(
        "--removes",
        "-e",
        type=str,
        help="regex pattern to exclude some words from the data",
    )
    process_options.add_argument(
        "--column",
        "-c",
        type=str,
        help="specify the column name to read data",
        default="text",
    )
    process_options.add_argument(
        "--threshold",
        "-t",
        type=int,
        help="threshold tokens number to filter out dataset rows",
        default=20,
    )
    process_options.add_argument(
        "--target-length",
        "-l",
        type=int,
        help="target length to group dataset",
        default=1024,
    )
    process_options.add_argument(
        "--processes", "-j", type=int, help="processes used to count tokens", default=1
    )

    # resolve
    args = parser.parse_args()

    # setup logger
    run_id = generate_run_id()
    setup_logger(
        run_id=run_id,
        logger_config={"log_to_file": True, "log_path": Path("logs/preprocess")},
    )

    logger.info("Starting preprocess")

    _main(
        inputs=args.input,
        prefix=args.prefix,
        output_path=args.output,
        run_id=run_id,
        tokenizer_name=args.tokenizer,
        removes=args.removes,
        column_name=args.column,
        threshold=args.threshold,
        target_length=args.target_length,
        num_processes=args.processes,
    )
