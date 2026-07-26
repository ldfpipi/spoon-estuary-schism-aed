from __future__ import annotations

import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 只保存图片，不使用PyCharm内嵌交互窗口

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from netCDF4 import Dataset


# ============================================================
# 1. USER SETTINGS
# ============================================================

TUFLOW_BASE = Path(
    r"D:\SCHISM_W9\TUFLOWFV_Tutorial_M05"
    r"\Tutorial_Module_05_Estuary_Module"
    r"\Complete_Model\TUFLOWFV"
)

TUFLOW_NC = (
    TUFLOW_BASE
    / "results"
    / "HYD_002_W.nc"
)

SELECTED_POINTS_CSV = (
    TUFLOW_BASE
    / "comparison_three_locations"
    / "01_selected_three_locations.csv"
)

SCHISM_CASE = Path(
    r"C:\Users\uqdliu17\Documents"
    r"\schism_tutorial_m05_benchmark"
)

SCHISM_HGRID = SCHISM_CASE / "hgrid.gr3"
SCHISM_OUTPUTS = SCHISM_CASE / "outputs"

OUTPUT_DIR = (
    TUFLOW_BASE
    / "comparison_three_locations"
    / "final_timeseries_comparison"
)

FIGURE_DIR = OUTPUT_DIR / "figures"
CSV_DIR = OUTPUT_DIR / "csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)


# Comparison interval, in model local time: AEST
COMPARISON_START = pd.Timestamp("2011-05-01 01:00:00")
COMPARISON_END = pd.Timestamp("2011-05-07 00:00:00")

# SCHISM time values are seconds from this local model start.
SCHISM_LOCAL_START = pd.Timestamp("2011-05-01 00:00:00")

# TUFLOW ResTime is hours from this origin.
TUFLOW_TIME_ORIGIN = pd.Timestamp("1990-01-01 00:00:00")

# Common physical vertical positions
SURFACE_OFFSET_M = 0.5
BOTTOM_OFFSET_M = 0.5

# The common instantaneous reference surface is defined as:
# 0.5 * (TUFLOW water level + SCHISM water level)
REFERENCE_SURFACE_MODE = "mean_model_surface"

EXPECTED_COMMON_RECORDS = 144


# ============================================================
# 2. VARIABLE DEFINITIONS
# ============================================================

VARIABLE_CONFIG = {
    "Vx": {
        "tuflow_name": "V_x",
        "schism_prefix": "horizontalVelX",
        "schism_name": "horizontalVelX",
        "display_name": "X-direction velocity",
        "unit": "m s$^{-1}$",
        "file_order": 2,
    },
    "Vy": {
        "tuflow_name": "V_y",
        "schism_prefix": "horizontalVelY",
        "schism_name": "horizontalVelY",
        "display_name": "Y-direction velocity",
        "unit": "m s$^{-1}$",
        "file_order": 3,
    },
    "Vz": {
        # Keep the raw model variable names unchanged.
        "tuflow_name": "W",
        "schism_prefix": "verticalVelocity",
        "schism_name": "verticalVelocity",
        "display_name": "Vertical velocity, Vz",
        "unit": "m s$^{-1}$",
        "file_order": 4,
    },
    "Temperature": {
        "tuflow_name": "TEMP",
        "schism_prefix": "temperature",
        "schism_name": "temperature",
        "display_name": "Temperature",
        "unit": "°C",
        "file_order": 5,
    },
    "Salinity": {
        "tuflow_name": "SAL",
        "schism_prefix": "salinity",
        "schism_name": "salinity",
        "display_name": "Salinity",
        "unit": "psu",
        "file_order": 6,
    },
}

LEVEL_NAMES = [
    "Surface",
    "Mid-depth",
    "Near-bottom",
]


# ============================================================
# 2.1 LOCATION DEFINITIONS AND DISPLAY NAMES
# ============================================================

LOCATION_RENAME_MAP = {
    "mouth": "Lower",
    "lower": "Lower",
    "lower estuary": "Lower",
    "middle": "Middle",
    "middle estuary": "Middle",
    "upstream": "Upper",
    "upper": "Upper",
    "upper estuary": "Upper",
}

LOCATION_ORDER = {
    "Lower": 0,
    "Middle": 1,
    "Upper": 2,
}

EXPECTED_LOCATIONS = {
    "Lower": {
        "cell_number_one_based": 1359,
        "element_id_2dm": 1359,
    },
    "Middle": {
        "cell_number_one_based": 681,
        "element_id_2dm": 681,
    },
    "Upper": {
        "cell_number_one_based": 113,
        "element_id_2dm": 113,
    },
}


# ============================================================
# 3. GENERAL UTILITIES
# ============================================================

def to_nan(values) -> np.ndarray:
    """
    Convert a NetCDF masked array to a normal float NumPy array.
    Very large fill values are converted to NaN.
    """

    if np.ma.isMaskedArray(values):
        array = np.asarray(
            values.filled(np.nan),
            dtype=float,
        )
    else:
        array = np.asarray(
            values,
            dtype=float,
        )

    array[np.abs(array) > 1.0e30] = np.nan

    return array


def safe_filename(text: str) -> str:
    return re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        text,
    ).strip("_")


def normalise_location_name(value: object) -> str:
    """Convert old and new location names to Lower/Middle/Upper."""

    text = str(value).strip()
    normalised = LOCATION_RENAME_MAP.get(text.lower())

    if normalised is None:
        raise ValueError(
            "Unrecognised location name in selected-point CSV: "
            f"{text!r}"
        )

    return normalised


def backup_and_clear_previous_outputs() -> Path | None:
    """
    Back up prior generated figures, CSV files and the summary report,
    then remove them so obsolete Mouth/Upstream/W files cannot remain
    mixed with the new Lower/Upper/Vz outputs.
    """

    existing_figures = sorted(FIGURE_DIR.glob("*.png"))
    existing_csvs = sorted(CSV_DIR.glob("*.csv"))
    existing_summary = OUTPUT_DIR / "04_run_summary.txt"

    existing_files = [*existing_figures, *existing_csvs]

    if existing_summary.exists():
        existing_files.append(existing_summary)

    if not existing_files:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_directory = (
        OUTPUT_DIR
        / "backup_before_lower_middle_upper_vz"
        / timestamp
    )

    backup_figure_directory = backup_directory / "figures"
    backup_csv_directory = backup_directory / "csv"

    backup_figure_directory.mkdir(parents=True, exist_ok=True)
    backup_csv_directory.mkdir(parents=True, exist_ok=True)

    for path in existing_figures:
        shutil.copy2(path, backup_figure_directory / path.name)

    for path in existing_csvs:
        shutil.copy2(path, backup_csv_directory / path.name)

    if existing_summary.exists():
        shutil.copy2(existing_summary, backup_directory / existing_summary.name)

    for path in existing_figures:
        path.unlink()

    for path in existing_csvs:
        path.unlink()

    if existing_summary.exists():
        existing_summary.unlink()

    return backup_directory


def find_stack_numbers(
    folder: Path,
    prefix: str,
) -> list[int]:
    """
    Robust enumeration for mapped/network drives.

    Example:
        out2d_1.nc
        out2d_2.nc
        ...
        out2d_15.nc
    """

    pattern = re.compile(
        rf"^{re.escape(prefix)}_(\d+)\.nc$",
        flags=re.IGNORECASE,
    )

    numbers: list[int] = []

    with os.scandir(folder) as entries:
        for entry in entries:

            if not entry.is_file():
                continue

            match = pattern.fullmatch(entry.name)

            if match:
                numbers.append(
                    int(match.group(1))
                )

    return sorted(set(numbers))


# ============================================================
# 4. READ SCHISM HGRID
# ============================================================

def read_schism_hgrid(path: Path):
    """
    Read:
      - SCHISM node order and coordinates
      - node depth
      - element connectivity
    """

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:

        title = file.readline().strip()

        header = file.readline().split()

        if len(header) < 2:
            raise RuntimeError(
                f"Invalid hgrid header:\n{path}"
            )

        number_elements = int(header[0])
        number_nodes = int(header[1])

        nodes: dict[int, dict] = {}
        node_ids_in_file_order: list[int] = []

        for node_index in range(number_nodes):

            parts = file.readline().split()

            if len(parts) < 4:
                raise RuntimeError(
                    "Unexpected end of SCHISM node section."
                )

            node_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            depth = float(parts[3])

            nodes[node_id] = {
                "index": node_index,
                "x": x,
                "y": y,
                "depth": depth,
            }

            node_ids_in_file_order.append(node_id)

        elements: dict[int, list[int]] = {}

        for _ in range(number_elements):

            parts = file.readline().split()

            if len(parts) < 4:
                raise RuntimeError(
                    "Unexpected end of SCHISM element section."
                )

            element_id = int(parts[0])
            number_vertices = int(parts[1])

            node_ids = [
                int(value)
                for value in parts[
                    2:2 + number_vertices
                ]
            ]

            elements[element_id] = node_ids

    return (
        title,
        nodes,
        node_ids_in_file_order,
        elements,
    )


# ============================================================
# 5. HORIZONTAL INTERPOLATION WEIGHTS
# ============================================================

def triangle_barycentric_weights(
    point: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:
    """
    Return barycentric weights for one triangle.
    vertices shape: (3, 2)
    """

    x, y = point

    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]

    denominator = (
        (y2 - y3) * (x1 - x3)
        + (x3 - x2) * (y1 - y3)
    )

    if abs(denominator) < 1.0e-20:
        raise RuntimeError(
            "Degenerate triangular element encountered."
        )

    w1 = (
        (y2 - y3) * (x - x3)
        + (x3 - x2) * (y - y3)
    ) / denominator

    w2 = (
        (y3 - y1) * (x - x3)
        + (x1 - x3) * (y - y3)
    ) / denominator

    w3 = 1.0 - w1 - w2

    return np.asarray(
        [w1, w2, w3],
        dtype=float,
    )


def quad_shape_functions(
    xi: float,
    eta: float,
) -> np.ndarray:
    """
    Four-node bilinear quadrilateral shape functions.

    Expected node ordering is the cyclic element ordering
    stored in the SCHISM/TUFLOW mesh.
    """

    return 0.25 * np.asarray(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ],
        dtype=float,
    )


def quad_shape_derivatives(
    xi: float,
    eta: float,
):
    """
    Derivatives of the four bilinear shape functions.
    """

    d_n_d_xi = 0.25 * np.asarray(
        [
            -(1.0 - eta),
            +(1.0 - eta),
            +(1.0 + eta),
            -(1.0 + eta),
        ],
        dtype=float,
    )

    d_n_d_eta = 0.25 * np.asarray(
        [
            -(1.0 - xi),
            -(1.0 + xi),
            +(1.0 + xi),
            +(1.0 - xi),
        ],
        dtype=float,
    )

    return d_n_d_xi, d_n_d_eta


def bilinear_quad_weights(
    point: np.ndarray,
    vertices: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    Invert the bilinear quadrilateral mapping using Newton iteration.

    Returns:
        weights, xi, eta
    """

    xi = 0.0
    eta = 0.0

    tolerance = 1.0e-13

    for _ in range(50):

        weights = quad_shape_functions(
            xi,
            eta,
        )

        mapped_point = (
            weights[:, None] * vertices
        ).sum(axis=0)

        residual = mapped_point - point

        if np.linalg.norm(residual) < tolerance:
            break

        d_n_d_xi, d_n_d_eta = (
            quad_shape_derivatives(
                xi,
                eta,
            )
        )

        dx_d_xi = np.sum(
            d_n_d_xi * vertices[:, 0]
        )

        dy_d_xi = np.sum(
            d_n_d_xi * vertices[:, 1]
        )

        dx_d_eta = np.sum(
            d_n_d_eta * vertices[:, 0]
        )

        dy_d_eta = np.sum(
            d_n_d_eta * vertices[:, 1]
        )

        jacobian = np.asarray(
            [
                [dx_d_xi, dx_d_eta],
                [dy_d_xi, dy_d_eta],
            ],
            dtype=float,
        )

        if abs(np.linalg.det(jacobian)) < 1.0e-20:
            raise RuntimeError(
                "Degenerate quadrilateral Jacobian."
            )

        correction = np.linalg.solve(
            jacobian,
            residual,
        )

        xi -= correction[0]
        eta -= correction[1]

    weights = quad_shape_functions(
        xi,
        eta,
    )

    return weights, xi, eta


def determine_horizontal_weights(
    point: np.ndarray,
    vertices: np.ndarray,
) -> tuple[np.ndarray, str]:
    """
    Determine interpolation weights for a triangle or quadrilateral.

    For quadrilaterals:
      1. Try four-node bilinear interpolation.
      2. Fall back to a triangular split if necessary.
    """

    number_vertices = len(vertices)

    if number_vertices == 3:

        weights = triangle_barycentric_weights(
            point,
            vertices,
        )

        return weights, "triangle barycentric"

    if number_vertices != 4:
        raise ValueError(
            f"Unsupported element with "
            f"{number_vertices} vertices."
        )

    try:
        weights, xi, eta = bilinear_quad_weights(
            point,
            vertices,
        )

        if (
            abs(xi) <= 1.0001
            and abs(eta) <= 1.0001
            and np.all(weights >= -1.0e-6)
        ):
            weights[np.abs(weights) < 1.0e-12] = 0.0

            weights = weights / np.sum(weights)

            method = (
                "quadrilateral bilinear "
                f"(xi={xi:.6f}, eta={eta:.6f})"
            )

            return weights, method

    except Exception:
        pass

    # Fallback: split the quad into two triangles.
    triangle_candidates = [
        [0, 1, 2],
        [0, 2, 3],
    ]

    best_result = None
    best_minimum_weight = -np.inf

    for triangle_indices in triangle_candidates:

        triangle_vertices = vertices[
            triangle_indices
        ]

        triangle_weights = (
            triangle_barycentric_weights(
                point,
                triangle_vertices,
            )
        )

        minimum_weight = float(
            np.min(triangle_weights)
        )

        if minimum_weight > best_minimum_weight:

            full_weights = np.zeros(
                4,
                dtype=float,
            )

            full_weights[
                triangle_indices
            ] = triangle_weights

            best_result = full_weights
            best_minimum_weight = minimum_weight

    if best_result is None:
        raise RuntimeError(
            "Could not calculate quadrilateral weights."
        )

    best_result[
        np.abs(best_result) < 1.0e-12
    ] = 0.0

    # Remove tiny negative round-off values.
    best_result[
        (best_result < 0.0)
        & (best_result > -1.0e-8)
    ] = 0.0

    if np.sum(best_result) == 0.0:
        raise RuntimeError(
            "Horizontal interpolation weights sum to zero."
        )

    best_result = best_result / np.sum(
        best_result
    )

    return best_result, "quadrilateral triangular fallback"


def weighted_finite_value(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, int]:
    """
    Weighted horizontal interpolation.

    NaN values are excluded and the remaining weights
    are renormalised.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (np.abs(weights) > 0.0)
    )

    number_valid = int(np.sum(valid))

    if number_valid == 0:
        return np.nan, 0

    valid_weights = weights[valid]

    weight_sum = np.sum(valid_weights)

    if abs(weight_sum) < 1.0e-14:
        return np.nan, number_valid

    value = np.sum(
        values[valid] * valid_weights
    ) / weight_sum

    return float(value), number_valid


# ============================================================
# 6. VERTICAL INTERPOLATION
# ============================================================

def vertical_interpolate(
    z_coordinates: np.ndarray,
    values: np.ndarray,
    target_z: float,
) -> float:
    """
    Linearly interpolate a vertical profile to one absolute z elevation.

    No extrapolation is applied.
    """

    z_coordinates = np.asarray(
        z_coordinates,
        dtype=float,
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    valid = (
        np.isfinite(z_coordinates)
        & np.isfinite(values)
    )

    z = z_coordinates[valid]
    v = values[valid]

    if len(z) < 2:
        return np.nan

    order = np.argsort(z)

    z = z[order]
    v = v[order]

    # Average exact duplicate z levels.
    unique_z, inverse = np.unique(
        z,
        return_inverse=True,
    )

    if len(unique_z) < 2:
        return np.nan

    if len(unique_z) != len(z):

        accumulated_values = np.zeros(
            len(unique_z),
            dtype=float,
        )

        accumulated_counts = np.zeros(
            len(unique_z),
            dtype=int,
        )

        for source_index, unique_index in enumerate(
            inverse
        ):
            accumulated_values[unique_index] += (
                v[source_index]
            )

            accumulated_counts[unique_index] += 1

        v = (
            accumulated_values
            / accumulated_counts
        )

        z = unique_z

    tolerance = 1.0e-8

    if target_z < z[0] - tolerance:
        return np.nan

    if target_z > z[-1] + tolerance:
        return np.nan

    target_z_clipped = float(
        np.clip(
            target_z,
            z[0],
            z[-1],
        )
    )

    return float(
        np.interp(
            target_z_clipped,
            z,
            v,
        )
    )


def interpolate_schism_point_value(
    node_z_profiles: np.ndarray,
    node_value_profiles: np.ndarray,
    horizontal_weights: np.ndarray,
    target_z: float,
) -> tuple[float, int]:
    """
    SCHISM interpolation order:

      1. Vertically interpolate at each element node.
      2. Horizontally interpolate the node values to the
         TUFLOW cell centroid.

    This handles different bottom indices among the nodes.
    """

    number_nodes = (
        node_z_profiles.shape[0]
    )

    node_values_at_target = np.full(
        number_nodes,
        np.nan,
        dtype=float,
    )

    for node_position in range(number_nodes):

        node_values_at_target[
            node_position
        ] = vertical_interpolate(
            node_z_profiles[node_position, :],
            node_value_profiles[node_position, :],
            target_z,
        )

    return weighted_finite_value(
        node_values_at_target,
        horizontal_weights,
    )


# ============================================================
# 7. LOAD SCHISM STACKS
# ============================================================

def load_schism_node_variable(
    outputs_folder: Path,
    prefix: str,
    variable_name: str,
    selected_node_indices: list[int],
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Load all SCHISM stacks for one variable.

    Only the required horizontal nodes are retained.
    """

    stack_numbers = find_stack_numbers(
        outputs_folder,
        prefix,
    )

    if not stack_numbers:
        raise FileNotFoundError(
            f"No SCHISM files found for prefix: {prefix}"
        )

    time_parts: list[pd.DatetimeIndex] = []
    data_parts: list[np.ndarray] = []

    for stack_number in stack_numbers:

        path = (
            outputs_folder
            / f"{prefix}_{stack_number}.nc"
        )

        with Dataset(path, "r") as ds:

            if "time" not in ds.variables:
                raise KeyError(
                    f"No time variable in:\n{path}"
                )

            if variable_name not in ds.variables:
                raise KeyError(
                    f"Variable {variable_name} not found in:\n"
                    f"{path}"
                )

            seconds = to_nan(
                ds.variables["time"][:]
            )

            current_times = (
                SCHISM_LOCAL_START
                + pd.to_timedelta(
                    seconds,
                    unit="s",
                )
            )

            raw_data = to_nan(
                ds.variables[variable_name][:]
            )

            if raw_data.ndim == 2:

                selected_data = raw_data[
                    :,
                    selected_node_indices,
                ]

            elif raw_data.ndim == 3:

                selected_data = raw_data[
                    :,
                    selected_node_indices,
                    :,
                ]

            else:
                raise RuntimeError(
                    f"Unexpected dimensions for "
                    f"{variable_name}: {raw_data.shape}"
                )

            time_parts.append(
                pd.DatetimeIndex(current_times)
            )

            data_parts.append(
                selected_data
            )

    times = pd.DatetimeIndex(
        np.concatenate(
            [
                part.to_numpy()
                for part in time_parts
            ]
        )
    )

    data = np.concatenate(
        data_parts,
        axis=0,
    )

    if times.duplicated().any():
        raise RuntimeError(
            f"Duplicate timestamps detected in "
            f"SCHISM variable {variable_name}."
        )

    return times, data


# ============================================================
# 8. METRICS
# ============================================================

def calculate_metrics(
    tuflow_values: np.ndarray,
    schism_values: np.ndarray,
) -> dict:
    """
    Difference convention:
        SCHISM - TUFLOW
    """

    tuflow_values = np.asarray(
        tuflow_values,
        dtype=float,
    )

    schism_values = np.asarray(
        schism_values,
        dtype=float,
    )

    valid = (
        np.isfinite(tuflow_values)
        & np.isfinite(schism_values)
    )

    number_valid = int(np.sum(valid))

    if number_valid == 0:
        return {
            "n": 0,
            "bias": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "maximum_absolute_error": np.nan,
            "correlation": np.nan,
        }

    tuf = tuflow_values[valid]
    sch = schism_values[valid]

    difference = sch - tuf

    bias = float(
        np.mean(difference)
    )

    mae = float(
        np.mean(
            np.abs(difference)
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                difference ** 2
            )
        )
    )

    maximum_absolute_error = float(
        np.max(
            np.abs(difference)
        )
    )

    if (
        number_valid >= 2
        and np.std(tuf) > 0.0
        and np.std(sch) > 0.0
    ):
        correlation = float(
            np.corrcoef(
                tuf,
                sch,
            )[0, 1]
        )
    else:
        correlation = np.nan

    return {
        "n": number_valid,
        "bias": bias,
        "mae": mae,
        "rmse": rmse,
        "maximum_absolute_error": (
            maximum_absolute_error
        ),
        "correlation": correlation,
    }


# ============================================================
# 9. CHECK INPUTS
# ============================================================

print("===== INPUT PATH CHECK =====")

input_paths = [
    ("TUFLOW NetCDF", TUFLOW_NC),
    ("Selected points", SELECTED_POINTS_CSV),
    ("SCHISM hgrid", SCHISM_HGRID),
    ("SCHISM outputs", SCHISM_OUTPUTS),
]

for label, path in input_paths:
    print(
        f"{label:18s}: "
        f"{path.exists()} | {path}"
    )

missing_paths = [
    path
    for _, path in input_paths
    if not path.exists()
]

if missing_paths:
    raise FileNotFoundError(
        "Missing required paths:\n"
        + "\n".join(
            str(path)
            for path in missing_paths
        )
    )

backup_directory = backup_and_clear_previous_outputs()

if backup_directory is not None:
    print("Previous generated outputs backed up to:")
    print(backup_directory)


# ============================================================
# 10. READ HGRID AND SELECTED LOCATIONS
# ============================================================

(
    hgrid_title,
    schism_nodes,
    node_ids_in_file_order,
    schism_elements,
) = read_schism_hgrid(
    SCHISM_HGRID
)

selected_points = pd.read_csv(
    SELECTED_POINTS_CSV
)

required_selected_columns = {
    "location",
    "cell_number_one_based",
    "element_id_2dm",
    "cell_x",
    "cell_y",
}

missing_selected_columns = (
    required_selected_columns
    - set(selected_points.columns)
)

if missing_selected_columns:
    raise KeyError(
        "Selected-point CSV is missing columns: "
        + ", ".join(sorted(missing_selected_columns))
    )

selected_points["location"] = (
    selected_points["location"]
    .apply(normalise_location_name)
)

selected_points["_location_order"] = (
    selected_points["location"]
    .map(LOCATION_ORDER)
)

selected_points = (
    selected_points
    .sort_values("_location_order")
    .drop(columns="_location_order")
    .reset_index(drop=True)
)

if selected_points["location"].duplicated().any():
    raise RuntimeError(
        "Duplicate Lower/Middle/Upper rows were found in the "
        "selected-point CSV."
    )

for location_name, expected in EXPECTED_LOCATIONS.items():
    matching = selected_points.loc[
        selected_points["location"] == location_name
    ]

    if len(matching) != 1:
        raise RuntimeError(
            f"Expected exactly one row for {location_name}, "
            f"but found {len(matching)}."
        )

    row = matching.iloc[0]
    actual_cell = int(row["cell_number_one_based"])
    actual_element = int(row["element_id_2dm"])

    if actual_cell != expected["cell_number_one_based"]:
        raise RuntimeError(
            f"{location_name} cell mismatch: "
            f"expected {expected['cell_number_one_based']}, "
            f"found {actual_cell}."
        )

    if actual_element != expected["element_id_2dm"]:
        raise RuntimeError(
            f"{location_name} element mismatch: "
            f"expected {expected['element_id_2dm']}, "
            f"found {actual_element}."
        )

print()
print("===== FIXED LOCATION CHECK =====")
for _, row in selected_points.iterrows():
    print(
        f"{row['location']:8s}: "
        f"Cell {int(row['cell_number_one_based'])}, "
        f"Element {int(row['element_id_2dm'])} — PASS"
    )

location_definitions: list[dict] = []
all_required_node_ids: set[int] = set()

print()
print("===== HORIZONTAL LOCATION DEFINITIONS =====")

for _, row in selected_points.iterrows():

    location_name = str(
        row["location"]
    )

    element_id = int(
        row["element_id_2dm"]
    )

    point_x = float(
        row["cell_x"]
    )

    point_y = float(
        row["cell_y"]
    )

    if element_id not in schism_elements:
        raise KeyError(
            f"Element {element_id} not found "
            f"for location {location_name}."
        )

    element_node_ids = (
        schism_elements[element_id]
    )

    vertices = np.asarray(
        [
            [
                schism_nodes[node_id]["x"],
                schism_nodes[node_id]["y"],
            ]
            for node_id in element_node_ids
        ],
        dtype=float,
    )

    point = np.asarray(
        [point_x, point_y],
        dtype=float,
    )

    horizontal_weights, weight_method = (
        determine_horizontal_weights(
            point,
            vertices,
        )
    )

    location_definition = {
        "location": location_name,
        "element_id": element_id,
        "cell_number_one_based": int(
            row["cell_number_one_based"]
        ),
        "x": point_x,
        "y": point_y,
        "node_ids": element_node_ids,
        "horizontal_weights": (
            horizontal_weights
        ),
        "weight_method": weight_method,
    }

    location_definitions.append(
        location_definition
    )

    all_required_node_ids.update(
        element_node_ids
    )

    print()
    print(f"Location : {location_name}")
    print(f"Element  : {element_id}")
    print(f"Nodes    : {element_node_ids}")
    print(
        "Weights  : "
        + str(
            np.round(
                horizontal_weights,
                8,
            ).tolist()
        )
    )
    print(f"Method   : {weight_method}")


# Convert SCHISM node IDs to NetCDF node indices.
required_node_ids = sorted(
    all_required_node_ids
)

required_node_indices = [
    schism_nodes[node_id]["index"]
    for node_id in required_node_ids
]

subset_position_by_node_id = {
    node_id: position
    for position, node_id in enumerate(
        required_node_ids
    )
}

for location_definition in location_definitions:

    location_definition[
        "subset_node_positions"
    ] = [
        subset_position_by_node_id[node_id]
        for node_id
        in location_definition["node_ids"]
    ]


# ============================================================
# 11. LOAD SCHISM RESULTS
# ============================================================

print()
print("===== LOADING SCHISM RESULTS =====")

schism_times, schism_elevation = (
    load_schism_node_variable(
        SCHISM_OUTPUTS,
        "out2d",
        "elevation",
        required_node_indices,
    )
)

print(
    f"Elevation: "
    f"{schism_elevation.shape}"
)

schism_times_z, schism_z = (
    load_schism_node_variable(
        SCHISM_OUTPUTS,
        "zCoordinates",
        "zCoordinates",
        required_node_indices,
    )
)

print(
    f"zCoordinates: "
    f"{schism_z.shape}"
)

if not schism_times_z.equals(
    schism_times
):
    raise RuntimeError(
        "SCHISM zCoordinates times do not match elevation."
    )

schism_variable_data: dict[str, np.ndarray] = {}

for variable_key, config in VARIABLE_CONFIG.items():

    variable_times, variable_data = (
        load_schism_node_variable(
            SCHISM_OUTPUTS,
            config["schism_prefix"],
            config["schism_name"],
            required_node_indices,
        )
    )

    if not variable_times.equals(
        schism_times
    ):
        raise RuntimeError(
            f"SCHISM time mismatch for {variable_key}."
        )

    schism_variable_data[
        variable_key
    ] = variable_data

    print(
        f"{variable_key:12s}: "
        f"{variable_data.shape}"
    )


# ============================================================
# 12. LOAD TUFLOW RESULTS
# ============================================================

print()
print("===== LOADING TUFLOW RESULTS =====")

tuflow_location_data: dict[str, dict] = {}

with Dataset(TUFLOW_NC, "r") as ds:

    required_tuflow_variables = [
        "ResTime",
        "NL",
        "idx3",
        "cell_Zb",
        "layerface_Z",
        "H",
        "D",
    ] + [
        config["tuflow_name"]
        for config in VARIABLE_CONFIG.values()
    ]

    missing_variables = [
        name
        for name in required_tuflow_variables
        if name not in ds.variables
    ]

    if missing_variables:
        raise KeyError(
            "TUFLOW variables missing: "
            + ", ".join(missing_variables)
        )

    tuflow_time_hours = to_nan(
        ds.variables["ResTime"][:]
    )

    tuflow_times = pd.DatetimeIndex(
        TUFLOW_TIME_ORIGIN
        + pd.to_timedelta(
            tuflow_time_hours,
            unit="h",
        )
    )

    number_layers = np.asarray(
        ds.variables["NL"][:],
        dtype=int,
    )

    index_2d_to_3d_one_based = np.asarray(
        ds.variables["idx3"][:],
        dtype=int,
    )

    bed_elevation = to_nan(
        ds.variables["cell_Zb"][:]
    )

    number_faces_per_cell = (
        number_layers + 1
    )

    face_start_zero_based = np.concatenate(
        (
            np.asarray([0], dtype=int),
            np.cumsum(
                number_faces_per_cell[:-1],
                dtype=int,
            ),
        )
    )

    for location_definition in location_definitions:

        location_name = (
            location_definition["location"]
        )

        cell_number = int(
            location_definition[
                "cell_number_one_based"
            ]
        )

        cell_index = (
            cell_number - 1
        )

        number_vertical_layers = int(
            number_layers[cell_index]
        )

        cell_3d_start = (
            int(
                index_2d_to_3d_one_based[
                    cell_index
                ]
            )
            - 1
        )

        cell_3d_stop = (
            cell_3d_start
            + number_vertical_layers
        )

        cell_face_start = int(
            face_start_zero_based[
                cell_index
            ]
        )

        cell_face_stop = (
            cell_face_start
            + number_vertical_layers
            + 1
        )

        location_data = {
            "cell_index": cell_index,
            "number_layers": (
                number_vertical_layers
            ),
            "bed_elevation": float(
                bed_elevation[cell_index]
            ),
            "water_level": to_nan(
                ds.variables["H"][
                    :,
                    cell_index,
                ]
            ),
            "water_depth": to_nan(
                ds.variables["D"][
                    :,
                    cell_index,
                ]
            ),
            "layer_faces": to_nan(
                ds.variables["layerface_Z"][
                    :,
                    cell_face_start:
                    cell_face_stop,
                ]
            ),
            "profiles": {},
        }

        for variable_key, config in (
            VARIABLE_CONFIG.items()
        ):

            location_data[
                "profiles"
            ][variable_key] = to_nan(
                ds.variables[
                    config["tuflow_name"]
                ][
                    :,
                    cell_3d_start:
                    cell_3d_stop,
                ]
            )

        tuflow_location_data[
            location_name
        ] = location_data

        print(
            f"{location_name:10s}: "
            f"cell={cell_number}, "
            f"NL={number_vertical_layers}, "
            f"bed={location_data['bed_elevation']:.6f} m"
        )


# ============================================================
# 13. ALIGN COMMON TIMES
# ============================================================

common_times = (
    tuflow_times
    .intersection(schism_times)
    .sort_values()
)

common_times = common_times[
    (common_times >= COMPARISON_START)
    & (common_times <= COMPARISON_END)
]

if len(common_times) != EXPECTED_COMMON_RECORDS:
    raise RuntimeError(
        "Unexpected number of common times:\n"
        f"Expected: {EXPECTED_COMMON_RECORDS}\n"
        f"Found   : {len(common_times)}\n"
        f"First   : "
        f"{common_times[0] if len(common_times) else 'None'}\n"
        f"Last    : "
        f"{common_times[-1] if len(common_times) else 'None'}"
    )

tuflow_time_indices = (
    tuflow_times.get_indexer(
        common_times
    )
)

schism_time_indices = (
    schism_times.get_indexer(
        common_times
    )
)

if np.any(tuflow_time_indices < 0):
    raise RuntimeError(
        "One or more common times are absent from TUFLOW."
    )

if np.any(schism_time_indices < 0):
    raise RuntimeError(
        "One or more common times are absent from SCHISM."
    )

print()
print("===== COMMON TIME PERIOD =====")
print(f"Records: {len(common_times)}")
print(f"First  : {common_times[0]}")
print(f"Last   : {common_times[-1]}")


# ============================================================
# 14. EXTRACT AND INTERPOLATE
# ============================================================

time_series_records: list[dict] = []
vertical_target_records: list[dict] = []

for location_definition in location_definitions:

    location_name = (
        location_definition["location"]
    )

    element_id = int(
        location_definition["element_id"]
    )

    cell_number = int(
        location_definition[
            "cell_number_one_based"
        ]
    )

    point_x = float(
        location_definition["x"]
    )

    point_y = float(
        location_definition["y"]
    )

    node_positions = (
        location_definition[
            "subset_node_positions"
        ]
    )

    horizontal_weights = np.asarray(
        location_definition[
            "horizontal_weights"
        ],
        dtype=float,
    )

    tuflow_data = (
        tuflow_location_data[
            location_name
        ]
    )

    bed_z = float(
        tuflow_data["bed_elevation"]
    )

    tuflow_water_level = (
        tuflow_data["water_level"][
            tuflow_time_indices
        ]
    )

    schism_water_level = np.full(
        len(common_times),
        np.nan,
        dtype=float,
    )

    schism_water_level_support = np.zeros(
        len(common_times),
        dtype=int,
    )

    for common_index, schism_index in enumerate(
        schism_time_indices
    ):

        interpolated_elevation, support = (
            weighted_finite_value(
                schism_elevation[
                    schism_index,
                    node_positions,
                ],
                horizontal_weights,
            )
        )

        schism_water_level[
            common_index
        ] = interpolated_elevation

        schism_water_level_support[
            common_index
        ] = support

    # The common physical z targets use the mean model water surface.
    reference_water_surface = (
        0.5
        * (
            tuflow_water_level
            + schism_water_level
        )
    )

    target_surface_z = (
        reference_water_surface
        - SURFACE_OFFSET_M
    )

    target_mid_z = (
        0.5
        * (
            reference_water_surface
            + bed_z
        )
    )

    target_bottom_z = np.full(
        len(common_times),
        bed_z + BOTTOM_OFFSET_M,
        dtype=float,
    )

    target_z_by_level = {
        "Surface": target_surface_z,
        "Mid-depth": target_mid_z,
        "Near-bottom": target_bottom_z,
    }

    # Save water-level records first.
    for time_position, timestamp in enumerate(
        common_times
    ):

        tuflow_value = float(
            tuflow_water_level[
                time_position
            ]
        )

        schism_value = float(
            schism_water_level[
                time_position
            ]
        )

        time_series_records.append(
            {
                "time_aest": timestamp,
                "location": location_name,
                "element_id": element_id,
                "tuflow_cell_number": cell_number,
                "x": point_x,
                "y": point_y,
                "bed_elevation_m": bed_z,
                "variable": "Water level",
                "level": "2D",
                "unit": "m",
                "target_z_m": np.nan,
                "tuflow": tuflow_value,
                "schism": schism_value,
                "schism_minus_tuflow": (
                    schism_value
                    - tuflow_value
                ),
                "schism_valid_nodes": int(
                    schism_water_level_support[
                        time_position
                    ]
                ),
            }
        )

        vertical_target_records.append(
            {
                "time_aest": timestamp,
                "location": location_name,
                "bed_elevation_m": bed_z,
                "tuflow_water_level_m": (
                    tuflow_value
                ),
                "schism_water_level_m": (
                    schism_value
                ),
                "reference_surface_m": float(
                    reference_water_surface[
                        time_position
                    ]
                ),
                "surface_target_z_m": float(
                    target_surface_z[
                        time_position
                    ]
                ),
                "mid_depth_target_z_m": float(
                    target_mid_z[
                        time_position
                    ]
                ),
                "near_bottom_target_z_m": float(
                    target_bottom_z[
                        time_position
                    ]
                ),
            }
        )

    # Extract all 3D variables.
    for time_position, timestamp in enumerate(
        common_times
    ):

        tuflow_index = int(
            tuflow_time_indices[
                time_position
            ]
        )

        schism_index = int(
            schism_time_indices[
                time_position
            ]
        )

        tuflow_layer_faces = (
            tuflow_data["layer_faces"][
                tuflow_index,
                :,
            ]
        )

        tuflow_layer_centres = (
            0.5
            * (
                tuflow_layer_faces[:-1]
                + tuflow_layer_faces[1:]
            )
        )

        schism_node_z_profiles = (
            schism_z[
                schism_index,
                node_positions,
                :,
            ]
        )

        for level_name in LEVEL_NAMES:

            target_z = float(
                target_z_by_level[
                    level_name
                ][time_position]
            )

            for variable_key, config in (
                VARIABLE_CONFIG.items()
            ):

                tuflow_profile = (
                    tuflow_data["profiles"][
                        variable_key
                    ][
                        tuflow_index,
                        :,
                    ]
                )

                tuflow_value = (
                    vertical_interpolate(
                        tuflow_layer_centres,
                        tuflow_profile,
                        target_z,
                    )
                )

                schism_node_value_profiles = (
                    schism_variable_data[
                        variable_key
                    ][
                        schism_index,
                        node_positions,
                        :,
                    ]
                )

                (
                    schism_value,
                    schism_support,
                ) = interpolate_schism_point_value(
                    schism_node_z_profiles,
                    schism_node_value_profiles,
                    horizontal_weights,
                    target_z,
                )

                if (
                    np.isfinite(tuflow_value)
                    and np.isfinite(schism_value)
                ):
                    difference = (
                        schism_value
                        - tuflow_value
                    )
                else:
                    difference = np.nan

                time_series_records.append(
                    {
                        "time_aest": timestamp,
                        "location": location_name,
                        "element_id": element_id,
                        "tuflow_cell_number": cell_number,
                        "x": point_x,
                        "y": point_y,
                        "bed_elevation_m": bed_z,
                        "variable": variable_key,
                        "level": level_name,
                        "unit": config["unit"],
                        "target_z_m": target_z,
                        "tuflow": tuflow_value,
                        "schism": schism_value,
                        "schism_minus_tuflow": (
                            difference
                        ),
                        "schism_valid_nodes": int(
                            schism_support
                        ),
                    }
                )


# ============================================================
# 15. BUILD DATAFRAMES AND METRICS
# ============================================================

comparison_df = pd.DataFrame(
    time_series_records
)

vertical_targets_df = pd.DataFrame(
    vertical_target_records
)

comparison_df = comparison_df.sort_values(
    [
        "location",
        "variable",
        "level",
        "time_aest",
    ]
).reset_index(drop=True)

vertical_targets_df = (
    vertical_targets_df.sort_values(
        [
            "location",
            "time_aest",
        ]
    ).reset_index(drop=True)
)

metrics_records: list[dict] = []

for (
    location_name,
    variable_name,
    level_name,
), group in comparison_df.groupby(
    [
        "location",
        "variable",
        "level",
    ],
    sort=False,
):

    metric_values = calculate_metrics(
        group["tuflow"].to_numpy(),
        group["schism"].to_numpy(),
    )

    minimum_support = int(
        group["schism_valid_nodes"].min()
    )

    maximum_support = int(
        group["schism_valid_nodes"].max()
    )

    metrics_records.append(
        {
            "location": location_name,
            "variable": variable_name,
            "level": level_name,
            "n": metric_values["n"],
            "bias_schism_minus_tuflow": (
                metric_values["bias"]
            ),
            "mae": metric_values["mae"],
            "rmse": metric_values["rmse"],
            "maximum_absolute_error": (
                metric_values[
                    "maximum_absolute_error"
                ]
            ),
            "correlation": (
                metric_values["correlation"]
            ),
            "minimum_schism_valid_nodes": (
                minimum_support
            ),
            "maximum_schism_valid_nodes": (
                maximum_support
            ),
        }
    )

metrics_df = pd.DataFrame(
    metrics_records
)


# ============================================================
# 16. SAVE CSV DATA
# ============================================================

comparison_csv = (
    CSV_DIR
    / "all_locations_timeseries_long.csv"
)

metrics_csv = (
    CSV_DIR
    / "all_locations_metrics.csv"
)

targets_csv = (
    CSV_DIR
    / "all_locations_vertical_targets.csv"
)

comparison_df.to_csv(
    comparison_csv,
    index=False,
    float_format="%.10f",
)

metrics_df.to_csv(
    metrics_csv,
    index=False,
    float_format="%.10f",
)

vertical_targets_df.to_csv(
    targets_csv,
    index=False,
    float_format="%.10f",
)

for location_name in selected_points[
    "location"
].astype(str):

    location_output = comparison_df[
        comparison_df["location"]
        == location_name
    ]

    location_output.to_csv(
        CSV_DIR
        / (
            safe_filename(location_name)
            + "_timeseries_long.csv"
        ),
        index=False,
        float_format="%.10f",
    )


# ============================================================
# 17. PLOTTING UTILITIES
# ============================================================

def configure_time_axis(ax) -> None:

    locator = mdates.AutoDateLocator(
        minticks=6,
        maxticks=10,
    )

    formatter = mdates.ConciseDateFormatter(
        locator
    )

    ax.xaxis.set_major_locator(
        locator
    )

    ax.xaxis.set_major_formatter(
        formatter
    )


def metric_title(
    location: str,
    variable: str,
    level: str,
) -> str:

    row = metrics_df[
        (metrics_df["location"] == location)
        & (metrics_df["variable"] == variable)
        & (metrics_df["level"] == level)
    ]

    if row.empty:
        return ""

    row = row.iloc[0]

    correlation = row["correlation"]

    if np.isfinite(correlation):
        correlation_text = (
            f"{correlation:.3f}"
        )
    else:
        correlation_text = "NaN"

    return (
        f"RMSE={row['rmse']:.4g}, "
        f"Bias={row['bias_schism_minus_tuflow']:.4g}, "
        f"R={correlation_text}, "
        f"N={int(row['n'])}"
    )


# ============================================================
# 18. PLOT EACH LOCATION
# ============================================================

figure_files: list[Path] = []

for location_name in selected_points[
    "location"
].astype(str):

    location_safe = safe_filename(
        location_name
    )

    # --------------------------------------------------------
    # 18.1 Vertical target-level audit figure
    # --------------------------------------------------------

    target_group = vertical_targets_df[
        vertical_targets_df["location"]
        == location_name
    ].sort_values("time_aest")

    fig, ax = plt.subplots(
        figsize=(12, 5.5)
    )

    ax.plot(
        target_group["time_aest"],
        target_group[
            "tuflow_water_level_m"
        ],
        label="TUFLOW water level",
        linewidth=1.2,
    )

    ax.plot(
        target_group["time_aest"],
        target_group[
            "schism_water_level_m"
        ],
        label="SCHISM water level",
        linewidth=1.2,
    )

    ax.plot(
        target_group["time_aest"],
        target_group[
            "surface_target_z_m"
        ],
        label="Surface target z",
        linewidth=1.0,
    )

    ax.plot(
        target_group["time_aest"],
        target_group[
            "mid_depth_target_z_m"
        ],
        label="Mid-depth target z",
        linewidth=1.0,
    )

    ax.plot(
        target_group["time_aest"],
        target_group[
            "near_bottom_target_z_m"
        ],
        label="Near-bottom target z",
        linewidth=1.0,
    )

    ax.axhline(
        target_group[
            "bed_elevation_m"
        ].iloc[0],
        linestyle="--",
        linewidth=1.0,
        label="Bed elevation",
    )

    ax.set_title(
        f"{location_name}: common physical vertical targets"
    )

    ax.set_ylabel("Elevation z (m)")
    ax.set_xlabel("Time (AEST)")
    ax.grid(True, alpha=0.3)

    configure_time_axis(ax)

    target_handles, target_labels = (
        ax.get_legend_handles_labels()
    )

    fig.legend(
        target_handles,
        target_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=True,
        framealpha=0.95,
        fontsize=9,
    )

    fig.tight_layout(
        rect=[0, 0.12, 1, 1]
    )

    target_figure = (
        FIGURE_DIR
        / (
            f"{location_safe}_00_"
            f"vertical_targets.png"
        )
    )

    fig.savefig(
        target_figure,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
    figure_files.append(target_figure)

    # --------------------------------------------------------
    # 18.2 Water level
    # --------------------------------------------------------

    water_group = comparison_df[
        (comparison_df["location"] == location_name)
        & (
            comparison_df["variable"]
            == "Water level"
        )
    ].sort_values("time_aest")

    fig, ax = plt.subplots(
        figsize=(12, 4.8)
    )

    ax.plot(
        water_group["time_aest"],
        water_group["tuflow"],
        label="TUFLOW FV",
        linewidth=1.3,
    )

    ax.plot(
        water_group["time_aest"],
        water_group["schism"],
        label="SCHISM",
        linewidth=1.3,
    )

    ax.set_title(
        f"{location_name}: Water level\n"
        + metric_title(
            location_name,
            "Water level",
            "2D",
        )
    )

    ax.set_ylabel("Water level (m)")
    ax.set_xlabel("Time (AEST)")
    ax.grid(True, alpha=0.3)

    configure_time_axis(ax)

    water_handles, water_labels = (
        ax.get_legend_handles_labels()
    )

    fig.legend(
        water_handles,
        water_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        fontsize=9.5,
    )

    fig.tight_layout(
        rect=[0, 0.10, 1, 1]
    )

    water_figure = (
        FIGURE_DIR
        / (
            f"{location_safe}_01_"
            f"water_level.png"
        )
    )

    fig.savefig(
        water_figure,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
    figure_files.append(water_figure)

    # --------------------------------------------------------
    # 18.3 Five 3D variables
    # --------------------------------------------------------

    for variable_key, config in (
        VARIABLE_CONFIG.items()
    ):

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(12, 10),
            sharex=True,
        )

        for axis, level_name in zip(
            axes,
            LEVEL_NAMES,
        ):

            group = comparison_df[
                (
                    comparison_df["location"]
                    == location_name
                )
                & (
                    comparison_df["variable"]
                    == variable_key
                )
                & (
                    comparison_df["level"]
                    == level_name
                )
            ].sort_values("time_aest")

            axis.plot(
                group["time_aest"],
                group["tuflow"],
                label="TUFLOW FV",
                linewidth=1.1,
            )

            axis.plot(
                group["time_aest"],
                group["schism"],
                label="SCHISM",
                linewidth=1.1,
            )

            if variable_key in {
                "Vx",
                "Vy",
                "Vz",
            }:
                axis.axhline(
                    0.0,
                    linewidth=0.6,
                )

            axis.set_title(
                f"{level_name}: "
                + metric_title(
                    location_name,
                    variable_key,
                    level_name,
                ),
                fontsize=10,
            )

            axis.set_ylabel(
                config["unit"]
            )

            axis.grid(
                True,
                alpha=0.3,
            )

        comparison_handles, comparison_labels = (
            axes[0].get_legend_handles_labels()
        )

        axes[-1].set_xlabel(
            "Time (AEST)"
        )

        configure_time_axis(
            axes[-1]
        )

        fig.suptitle(
            f"{location_name}: "
            f"{config['display_name']}",
            fontsize=14,
            y=0.985,
        )

        fig.legend(
            comparison_handles,
            comparison_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
            ncol=2,
            frameon=True,
            framealpha=0.95,
            fontsize=9.5,
        )

        fig.tight_layout(
            rect=[0, 0.055, 1, 0.955]
        )

        variable_figure = (
            FIGURE_DIR
            / (
                f"{location_safe}_"
                f"{config['file_order']:02d}_"
                f"{safe_filename(variable_key)}.png"
            )
        )

        fig.savefig(
            variable_figure,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)
        figure_files.append(
            variable_figure
        )


# ============================================================
# 19. WRITE SUMMARY REPORT
# ============================================================

summary_report = (
    OUTPUT_DIR
    / "04_run_summary.txt"
)

summary_lines: list[str] = []

summary_lines.append(
    "===== FINAL LOWER-MIDDLE-UPPER COMPARISON ====="
)

summary_lines.append(
    f"TUFLOW file: {TUFLOW_NC}"
)

summary_lines.append(
    f"SCHISM outputs: {SCHISM_OUTPUTS}"
)

summary_lines.append(
    f"SCHISM hgrid title: {hgrid_title}"
)

summary_lines.append("")
summary_lines.append(
    "Location names: Lower, Middle and Upper"
)
summary_lines.append(
    "Vertical velocity display name: Vz "
    "(raw TUFLOW W; raw SCHISM verticalVelocity)"
)
summary_lines.append(
    "Figure legends were placed below the plotting axes "
    "to avoid obscuring the time-series data."
)

summary_lines.append("")

summary_lines.append(
    f"Comparison start: {common_times[0]}"
)

summary_lines.append(
    f"Comparison end  : {common_times[-1]}"
)

summary_lines.append(
    f"Common records : {len(common_times)}"
)

summary_lines.append("")

summary_lines.append(
    "Vertical comparison method:"
)

summary_lines.append(
    "  Common reference water surface = "
    "0.5 * (TUFLOW water level + SCHISM water level)"
)

summary_lines.append(
    f"  Surface      = reference surface "
    f"- {SURFACE_OFFSET_M:.3f} m"
)

summary_lines.append(
    "  Mid-depth    = midpoint between "
    "reference surface and bed elevation"
)

summary_lines.append(
    f"  Near-bottom  = bed elevation "
    f"+ {BOTTOM_OFFSET_M:.3f} m"
)

summary_lines.append("")

summary_lines.append(
    "Difference convention:"
)

summary_lines.append(
    "  Difference = SCHISM - TUFLOW"
)

summary_lines.append("")

summary_lines.append(
    "Vertical velocity sign:"
)

summary_lines.append(
    "  Raw TUFLOW variable W and raw SCHISM variable "
    "verticalVelocity were displayed as Vz and compared "
    "without sign reversal."
)

summary_lines.append("")

summary_lines.append(
    "===== HORIZONTAL INTERPOLATION ====="
)

for location_definition in location_definitions:

    summary_lines.append("")

    summary_lines.append(
        f"Location: "
        f"{location_definition['location']}"
    )

    summary_lines.append(
        f"  Element ID: "
        f"{location_definition['element_id']}"
    )

    summary_lines.append(
        f"  Node IDs: "
        f"{location_definition['node_ids']}"
    )

    summary_lines.append(
        "  Weights: "
        + str(
            np.round(
                location_definition[
                    "horizontal_weights"
                ],
                10,
            ).tolist()
        )
    )

    summary_lines.append(
        f"  Method: "
        f"{location_definition['weight_method']}"
    )

summary_lines.append("")
summary_lines.append(
    "===== COMPARISON METRICS ====="
)
summary_lines.append("")

summary_lines.append(
    metrics_df.to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.8g}"
        ),
    )
)

summary_lines.append("")
summary_lines.append(
    "===== OUTPUT FILES ====="
)

summary_lines.append(
    f"Long time-series CSV: {comparison_csv}"
)

summary_lines.append(
    f"Metrics CSV: {metrics_csv}"
)

summary_lines.append(
    f"Vertical targets CSV: {targets_csv}"
)

summary_lines.append(
    f"Number of figures: {len(figure_files)}"
)

summary_report.write_text(
    "\n".join(summary_lines) + "\n",
    encoding="utf-8",
)


# ============================================================
# 20. FINAL TERMINAL OUTPUT
# ============================================================

print()
print("===== FINAL EXTRACTION COMPLETE =====")
print(f"Comparison records : {len(common_times)}")
print(f"Locations          : {len(location_definitions)}")
print(f"Figures created    : {len(figure_files)}")
print("Locations          : Lower / Middle / Upper")
print("Vertical velocity  : Vz")
print("Legends            : below axes; no data obstruction")
print()
print("Output directory:")
print(OUTPUT_DIR)
print()
print("Main CSV:")
print(comparison_csv)
print()
print("Metrics CSV:")
print(metrics_csv)
print()
print("Summary report:")
print(summary_report)