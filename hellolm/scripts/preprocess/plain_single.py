# The script used to preprocess single column plain dataset
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool
from argparse import ArgumentParser
from tiktoken import get_encoding
from sortedcontainers import SortedDict
from pyarrow.parquet import ParquetWriter
from pyarrow import Table, string, schema
from pandas import DataFrame, read_parquet
from hellolm.utils.logger import logger, setup_logger
from hellolm.utils.tools import generate_run_id, ensure_directory


def count_tokens_in_chunk(
    chunk_data,
    tokenizer_name: str,
    column_name: str,
    threshold: int,
    chunk_start: int,
    df_idx: int,
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

        token_count = len(tokenizer.encode(text))

        if token_count >= threshold:
            result.append((df_idx, chunk_start + idx, token_count, text))

    return result


def combine_texts(bin_item):
    return "\n".join(bin_item[1])


@logger.catch
def _main(
    inputs: str,
    prefix: str,
    output_path: str,
    tokenizer_name: str,
    column_name: str,
    threshold: int,
    target_length: int,
    num_processes: int = None,
):
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
                    (chunk, tokenizer_name, column_name, threshold, chunk_start, df_idx)
                )

            results = pool.starmap(count_tokens_in_chunk, chunks)

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
            best_space = remaining_space.keys()[best_bin_idx]
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
        tokenizer_name=args.tokenizer,
        column_name=args.column,
        threshold=args.threshold,
        target_length=args.target_length,
        num_processes=args.processes,
    )
