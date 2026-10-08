"""Read live run progress without repeatedly decoding old inverse records."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import zlib

NAMES = ("ar27_log", "ar27_sqrt", "ar27_linear", "ar27_quadratic")


class Stream:
    def __init__(self, records=False):
        self.offset = self.lines = 0
        self.decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        self.tail = b""
        self.records = [] if records else None

    def update(self, path):
        if not path.exists():
            return dict(bytes=0, complete_lines=0, exists=False)
        with path.open("rb") as stream:
            stream.seek(self.offset)
            while chunk := stream.read(1024*1024):
                self.offset += len(chunk)
                data = self.tail + self.decoder.decompress(chunk)
                complete = data.split(b"\n")
                self.tail = complete.pop()
                self.lines += len(complete)
                if self.records is not None:
                    for line in complete:
                        value = json.loads(line)
                        self.records.append(dict(value["record"],
                                                 state_rows=len(value.get("states", []))))
        result = dict(bytes=self.offset, complete_lines=self.lines, exists=True,
                      gzip_closed=self.decoder.eof, partial_line_bytes=len(self.tail))
        if self.records is not None:
            result["records"] = self.records
        return result


def load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def process_tree(pid):
    processes = {}
    ticks, page = os.sysconf("SC_CLK_TCK"), os.sysconf("SC_PAGE_SIZE")
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry/"stat").read_text()
            fields = raw[raw.rfind(")")+2:].split()
            processes[int(entry.name)] = dict(
                ppid=int(fields[1]), state=fields[0], start_ticks=int(fields[19]),
                cpu_seconds=(int(fields[11])+int(fields[12]))/ticks,
                rss_MiB=int(fields[21])*page/2**20)
        except (OSError, ValueError, IndexError):
            continue
    chosen = {pid} if pid in processes else set()
    while True:
        expanded = chosen | {p for p, value in processes.items() if value["ppid"] in chosen}
        if expanded == chosen:
            break
        chosen = expanded
    return {str(p): processes[p] for p in sorted(chosen)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--interval", type=float, default=60)
    args = parser.parse_args()
    if not 1 <= args.interval <= 60:
        raise ValueError("Monitor interval must be between 1 and 60 seconds")
    output = Path(__file__).resolve().parent
    readers = {name: (Stream(True), Stream()) for name in NAMES}
    while True:
        processes = process_tree(args.pid)
        snapshot = dict(utc=datetime.now(timezone.utc).isoformat(),
                        runner_pid=args.pid, runner_alive=str(args.pid) in processes,
                        processes=processes, numerical_results_verified=False, laws={})
        for name, (calibration, inverse) in readers.items():
            folder = output/"fits"/"full_calibration"/name
            model = load(folder/"model.json")
            completion = load(folder/"completion.json")
            snapshot["laws"][name] = dict(
                calibration=calibration.update(folder/"calibration_candidates.jsonl.gz"),
                calibration_certified=None if model is None else model["calibration"].get("converged"),
                selected_start=None if model is None else model["calibration"].get("selected_start"),
                inverse=inverse.update(folder/"inverse_candidates.jsonl.gz"),
                completion=completion,
                failed_checkpoint_exists=(folder/"failed_checkpoint.json").exists(),
                exception=(folder/"exception.txt").read_text()
                          if (folder/"exception.txt").exists() else None)
        snapshot["completion"] = load(output/"completion.json")
        temporary = output/"progress.json.tmp"
        temporary.write_text(json.dumps(snapshot, indent=2, allow_nan=False)+"\n")
        temporary.replace(output/"progress.json")
        print(json.dumps(dict(utc=snapshot["utc"], runner_alive=snapshot["runner_alive"],
                              laws={name: dict(calibration_certified=value["calibration_certified"],
                                               inverse_records=value["inverse"]["complete_lines"])
                                    for name, value in snapshot["laws"].items()})), flush=True)
        if not snapshot["runner_alive"] or snapshot["completion"] is not None:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
