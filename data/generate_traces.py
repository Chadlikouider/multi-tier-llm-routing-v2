import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Constants
YEAR = 2024
START_OF_YEAR = datetime(YEAR, 1, 1)
END_OF_YEAR = datetime(YEAR + 1, 1, 1)
HOURLY_INDEX = pd.date_range(start=START_OF_YEAR, end=END_OF_YEAR, freq="h", inclusive="left")
HOURS_PER_YEAR = len(HOURLY_INDEX)
BASE_REQUESTS = 1_000_000
MEAN_REQUESTS = 1_000_000
STD_REQUESTS = int(MEAN_REQUESTS / 3)
POISSON_LAMBDA = 1_000_000
SEED = 69

_OUTPUT_DIR = Path(__file__).resolve().parent / "request_traces"
_RNG = np.random.default_rng(SEED)


def _format_trace(values: np.ndarray) -> pd.DataFrame:
    """Return a trace DataFrame with hourly timestamps and integer counts."""
    counts = np.clip(np.rint(values), 0, None).astype(int)
    return pd.DataFrame({"timestamp": HOURLY_INDEX, "requests": counts})


def generate_static_trace(base_requests: int = BASE_REQUESTS) -> pd.DataFrame:
    """Constant number of requests per hour."""
    values = np.full(HOURS_PER_YEAR, base_requests)
    return _format_trace(values)


def generate_normal_trace(
    mean_requests: int = MEAN_REQUESTS,
    std_requests: int = STD_REQUESTS,
) -> pd.DataFrame:
    """Requests sampled from a normal distribution."""
    values = _RNG.normal(mean_requests, std_requests, size=HOURS_PER_YEAR)
    return _format_trace(values)


def generate_poisson_trace(lam: int = POISSON_LAMBDA) -> pd.DataFrame:
    """Requests sampled from a Poisson distribution."""
    values = _RNG.poisson(lam=lam, size=HOURS_PER_YEAR)
    return _format_trace(values)


def generate_synthetic_trace(
    base_requests: int = BASE_REQUESTS,
    daily_amplitude: int = 250,
    weekly_amplitude: int = 400,
    annual_amplitude: int = 600,
    noise_scale: float = STD_REQUESTS / 4,
) -> pd.DataFrame:
    """Sinusoidal pattern with noise, simulating realistic load."""
    hours = np.arange(HOURS_PER_YEAR)
    daily_cycle = np.sin(2 * np.pi * hours / 24)
    weekly_cycle = np.sin(2 * np.pi * hours / (24 * 7))
    annual_cycle = np.sin(2 * np.pi * hours / HOURS_PER_YEAR)
    noise = _RNG.normal(0, noise_scale, size=HOURS_PER_YEAR)

    values = (
        base_requests
        + daily_amplitude * daily_cycle
        + weekly_amplitude * weekly_cycle
        + annual_amplitude * annual_cycle
        + noise
    )
    return _format_trace(values)


def save_traces(output_dir: Path = _OUTPUT_DIR) -> None:
    """Generate and save all traces to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_generators = {
        "static": generate_static_trace,
        "normal": generate_normal_trace,
        "poisson": generate_poisson_trace,
        "synthetic": generate_synthetic_trace,
    }

    for name, generator in trace_generators.items():
        trace = generator()
        output_path = output_dir / f"{name}_requests_{YEAR}.csv"
        trace.to_csv(output_path, index=False)


if __name__ == "__main__":
    save_traces()