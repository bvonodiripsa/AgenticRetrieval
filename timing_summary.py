import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT_DIR / "out"
LATEST_LOG = OUT_DIR / "timing_5q_latest.log"
TABLE_PATH = OUT_DIR / "timing_5q_compare_table.tsv"

TIMING_RE = re.compile(r"\[TIMING\]\s+(.*?):\s+\+([0-9]*\.?[0-9]+)s")
VEC_DONE_RE = re.compile(r"vector query – done \((\d+) results(?:,\s*([^)]+))?\)")
MAT_DONE_RE = re.compile(r"vector materialize x\d+ \(([^)]+)\) – done")


def run_timed_rag() -> str:
    command = [sys.executable, "rag_divdet.py", "--max-questions", "5", "--timing"]
    result = subprocess.run(
        command,
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        raise RuntimeError(f"Timed run failed with exit code {result.returncode}\n{output}")
    return output


def parse_timings(text: str):
    data = {
        "retrieve_total": [],
        "fulltext_done": [],
        "vector_done": [],
        "mat_done": [],
        "vector_by_container": {},
        "mat_by_container": {},
        "llm_subq": [],
        "llm_synth": [],
        "llm_prelim": [],
        "llm_regen1": [],
        "llm_gap": [],
        "pipeline_total": [],
    }

    for line in text.splitlines():
        match = TIMING_RE.search(line)
        if not match:
            continue

        label = match.group(1).strip()
        value = float(match.group(2))

        if label.startswith("retrieve – TOTAL"):
            data["retrieve_total"].append(value)
        elif label.startswith("fulltext query – done"):
            data["fulltext_done"].append(value)
        elif label.startswith("vector query – done"):
            vm = VEC_DONE_RE.search(label)
            if vm:
                data["vector_done"].append(value)
                container = (vm.group(2) or "unknown").strip()
                data["vector_by_container"].setdefault(container, []).append(value)
        elif label.startswith("vector materialize"):
            data["mat_done"].append(value)
            mm = MAT_DONE_RE.search(label)
            if mm:
                container = (mm.group(1) or "unknown").strip()
                data["mat_by_container"].setdefault(container, []).append(value)
        elif label.startswith("LLM sub-Q answer – done"):
            data["llm_subq"].append(value)
        elif label.startswith("LLM synthesis – done"):
            data["llm_synth"].append(value)
        elif label.startswith("LLM preliminary – done"):
            data["llm_prelim"].append(value)
        elif label.startswith("LLM regenerate rnd 1 – done"):
            data["llm_regen1"].append(value)
        elif label.startswith("LLM gap-decompose – done"):
            data["llm_gap"].append(value)
        elif label.startswith("pipeline.run – TOTAL"):
            data["pipeline_total"].append(value)

    err_lines = [line for line in text.splitlines() if line.startswith("Error:")]
    badrequest = text.count("BadRequestError on")
    max_retry = text.count("Max retries exceeded")

    data["_meta"] = {
        "errors": len(err_lines),
        "badrequest": badrequest,
        "max_retry": max_retry,
    }
    return data


def mean(values):
    return sum(values) / len(values) if values else None


def first_wave_mean(values, n=5):
    return mean(values[:n]) if values else None


def contention_range(values, n=5):
    tail = values[n:] if len(values) > n else []
    if not tail:
        return None
    return min(tail), max(tail)


def full_range(values):
    if not values:
        return None
    return min(values), max(values)


def fmt_single(value):
    return "NA" if value is None else f"{value:.2f}s"


def fmt_range(rng):
    return "NA" if rng is None else f"{rng[0]:.2f}–{rng[1]:.2f}s"


def pct_change(prev, curr):
    if prev is None or curr is None or prev == 0:
        return "NA"
    return f"{((curr - prev) / prev) * 100:+.1f}%"


def range_width(rng):
    if rng is None:
        return None
    return rng[1] - rng[0]


def pct_change_range(prev_rng, curr_rng):
    prev_w = range_width(prev_rng)
    curr_w = range_width(curr_rng)
    if prev_w is None or curr_w is None or prev_w == 0:
        return "NA"
    return f"{((curr_w - prev_w) / prev_w) * 100:+.1f}%"


def build_metric_values(data):
    values = {}
    values["retrieve() TOTAL – single call"] = fmt_single(first_wave_mean(data["retrieve_total"], 5))
    values["retrieve() TOTAL – under sub-Q contention"] = fmt_range(contention_range(data["retrieve_total"], 5))
    values["Fulltext query – single"] = fmt_single(first_wave_mean(data["fulltext_done"], 5))
    values["Fulltext query – under contention (sub-Q parallel)"] = fmt_range(contention_range(data["fulltext_done"], 5))
    values["Vector query – single (all sources)"] = fmt_single(first_wave_mean(data["vector_done"], 5))
    values["Vector query – under contention (all sources)"] = fmt_range(contention_range(data["vector_done"], 5))
    values["Vector materialize/read stage (all sources)"] = fmt_range(full_range(data["mat_done"]))

    for container in sorted(data["vector_by_container"]):
        vals = data["vector_by_container"][container]
        values[f"Vector query – single ({container})"] = fmt_single(first_wave_mean(vals, 5))
        values[f"Vector query – under contention ({container})"] = fmt_range(contention_range(vals, 5))

    for container in sorted(data["mat_by_container"]):
        vals = data["mat_by_container"][container]
        values[f"Vector materialize/read stage ({container})"] = fmt_range(full_range(vals))

    values["LLM sub-Q answer"] = fmt_range(full_range(data["llm_subq"]))
    values["LLM synthesis"] = fmt_single(mean(data["llm_synth"]))
    values["LLM preliminary"] = fmt_single(mean(data["llm_prelim"]))
    values["LLM regenerate rnd 1"] = fmt_single(mean(data["llm_regen1"]))
    values["LLM gap-decompose"] = fmt_range(full_range(data["llm_gap"]))
    values["pipeline.run TOTAL"] = fmt_single(mean(data["pipeline_total"]))
    return values


def parse_single_seconds(value):
    if not value or value == "NA":
        return None
    return float(value.rstrip("s"))


def parse_range_seconds(value):
    if not value or value == "NA":
        return None
    left, right = value.rstrip("s").split("–", 1)
    return float(left), float(right)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    previous_log_text = None
    if LATEST_LOG.exists():
        previous_log_text = LATEST_LOG.read_text(encoding="utf-8", errors="ignore")

    current_log_text = run_timed_rag()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (OUT_DIR / f"timing_5q_rerun_{timestamp}.log").write_text(current_log_text, encoding="utf-8")
    LATEST_LOG.write_text(current_log_text, encoding="utf-8")

    current_data = parse_timings(current_log_text)
    current_values = build_metric_values(current_data)
    ordered_components = list(current_values.keys())

    lines = []
    if previous_log_text is None:
        lines.append("Component\tThis run")
        for component in ordered_components:
            lines.append(f"{component}\t{current_values[component]}")
        lines.append("")
        lines.append(
            f"_meta this_badrequest_errors={current_data['_meta']['badrequest']} this_max_retry_exceeded={current_data['_meta']['max_retry']}"
        )
    else:
        previous_data = parse_timings(previous_log_text)
        previous_values = build_metric_values(previous_data)

        lines.append("Component\tPrev run\tThis run\tChange")
        for component in ordered_components:
            prev_value = previous_values.get(component, "NA")
            curr_value = current_values.get(component, "NA")
            if "–" in prev_value and "–" in curr_value:
                change = pct_change_range(parse_range_seconds(prev_value), parse_range_seconds(curr_value))
            else:
                change = pct_change(parse_single_seconds(prev_value), parse_single_seconds(curr_value))
            lines.append(f"{component}\t{prev_value}\t{curr_value}\t{change}")

        lines.append("")
        lines.append(
            f"_meta prev_errors={previous_data['_meta']['errors']} new_badrequest_errors={current_data['_meta']['badrequest']} new_max_retry_exceeded={current_data['_meta']['max_retry']}"
        )

    TABLE_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
