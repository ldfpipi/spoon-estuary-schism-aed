from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from netCDF4 import Dataset


# ============================================================
# 1. Paths and settings
# ============================================================

TUFLOW_BASE = Path(
    r"D:\SCHISM_W9\TUFLOWFV_Tutorial_M05"
    r"\Tutorial_Module_05_Estuary_Module"
    r"\Complete_Model\TUFLOWFV"
)

TUFLOW_GRID = (
    TUFLOW_BASE
    / "model"
    / "geo"
    / "hydraul_006.2dm"
)

TUFLOW_NC = (
    TUFLOW_BASE
    / "results"
    / "HYD_002_W.nc"
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
    / "same_coordinate_ocean_residual_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SECTION_CSV = (
    OUTPUT_DIR
    / "same_coordinate_section_timeseries.csv"
)

SEGMENT_CSV = (
    OUTPUT_DIR
    / "same_coordinate_segment_timeseries.csv"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "same_coordinate_summary.csv"
)

SEGMENT_SUMMARY_CSV = (
    OUTPUT_DIR
    / "same_coordinate_segment_summary.csv"
)

SEGMENT_REPORT = (
    OUTPUT_DIR
    / "same_coordinate_segment_definitions.txt"
)

SUMMARY_REPORT = (
    OUTPUT_DIR
    / "same_coordinate_ocean_residual_summary.txt"
)


TUFLOW_TIME_ORIGIN = pd.Timestamp(
    "1990-01-01 00:00:00"
)

SCHISM_MODEL_START = pd.Timestamp(
    "2011-05-01 00:00:00"
)

COMPARISON_START = pd.Timestamp(
    "2011-05-01 01:00:00"
)

COMPARISON_END = pd.Timestamp(
    "2011-05-07 00:00:00"
)

EXPECTED_RECORDS = 144


# Must follow the Ocean-boundary order.
OCEAN_NODE_IDS = [
    1402,
    1408,
    1413,
    1417,
    1419,
    1418,
    1415,
]


LEVEL_NAMES = [
    "Surface",
    "Mid-depth",
    "Near-bottom",
]


SURFACE_OFFSET_M = 0.5
BOTTOM_OFFSET_M = 0.5


# ============================================================
# 2. General utilities
# ============================================================

def to_nan(values) -> np.ndarray:
    """
    Convert NetCDF masked values to normal float arrays.
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


def find_stack_numbers(
    folder: Path,
    prefix: str,
) -> list[int]:
    """
    Robustly enumerate SCHISM stacks on a mapped drive.
    """

    pattern = re.compile(
        rf"^{re.escape(prefix)}_(\d+)\.nc$",
        flags=re.IGNORECASE,
    )

    stack_numbers = []

    with os.scandir(folder) as entries:
        for entry in entries:

            if not entry.is_file():
                continue

            match = pattern.fullmatch(
                entry.name
            )

            if match:
                stack_numbers.append(
                    int(match.group(1))
                )

    return sorted(set(stack_numbers))


def configure_time_axis(axis) -> None:

    locator = mdates.AutoDateLocator(
        minticks=6,
        maxticks=10,
    )

    axis.xaxis.set_major_locator(
        locator
    )

    axis.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(
            locator
        )
    )


def weighted_finite_value(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, int]:
    """
    Weighted interpolation excluding invalid values.
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

    number_valid = int(
        np.sum(valid)
    )

    if number_valid == 0:
        return np.nan, 0

    valid_weights = weights[valid]

    weight_sum = float(
        np.sum(valid_weights)
    )

    if abs(weight_sum) < 1.0e-14:
        return np.nan, number_valid

    result = np.sum(
        values[valid]
        * valid_weights
    ) / weight_sum

    return float(result), number_valid


def weighted_finite_mean(
    values: list[float],
    weights: list[float],
) -> float:
    """
    Length-weighted section mean.
    """

    values_array = np.asarray(
        values,
        dtype=float,
    )

    weights_array = np.asarray(
        weights,
        dtype=float,
    )

    valid = (
        np.isfinite(values_array)
        & np.isfinite(weights_array)
        & (weights_array > 0.0)
    )

    if not np.any(valid):
        return np.nan

    return float(
        np.sum(
            values_array[valid]
            * weights_array[valid]
        )
        / np.sum(weights_array[valid])
    )


def vertical_interpolate(
    z_coordinates: np.ndarray,
    values: np.ndarray,
    target_z: float,
) -> float:
    """
    Linear interpolation at an absolute physical elevation.

    No extrapolation is performed.
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

    unique_z, inverse = np.unique(
        z,
        return_inverse=True,
    )

    if len(unique_z) < 2:
        return np.nan

    if len(unique_z) != len(z):

        value_sum = np.zeros(
            len(unique_z),
            dtype=float,
        )

        value_count = np.zeros(
            len(unique_z),
            dtype=int,
        )

        for old_index, new_index in enumerate(
            inverse
        ):
            value_sum[new_index] += (
                v[old_index]
            )

            value_count[new_index] += 1

        z = unique_z
        v = value_sum / value_count

    tolerance = 1.0e-8

    if target_z < z[0] - tolerance:
        return np.nan

    if target_z > z[-1] + tolerance:
        return np.nan

    target_z = float(
        np.clip(
            target_z,
            z[0],
            z[-1],
        )
    )

    return float(
        np.interp(
            target_z,
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
    First interpolate vertically at each SCHISM node,
    then interpolate horizontally to the TUFLOW cell centre.
    """

    number_nodes = (
        node_z_profiles.shape[0]
    )

    node_values = np.full(
        number_nodes,
        np.nan,
        dtype=float,
    )

    for node_position in range(
        number_nodes
    ):

        node_values[
            node_position
        ] = vertical_interpolate(
            node_z_profiles[
                node_position,
                :,
            ],
            node_value_profiles[
                node_position,
                :,
            ],
            target_z,
        )

    return weighted_finite_value(
        node_values,
        horizontal_weights,
    )


# ============================================================
# 3. Horizontal interpolation functions
# ============================================================

def triangle_barycentric_weights(
    point: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:

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
            "Degenerate triangular element."
        )

    weight_1 = (
        (y2 - y3) * (x - x3)
        + (x3 - x2) * (y - y3)
    ) / denominator

    weight_2 = (
        (y3 - y1) * (x - x3)
        + (x1 - x3) * (y - y3)
    ) / denominator

    weight_3 = (
        1.0
        - weight_1
        - weight_2
    )

    return np.asarray(
        [
            weight_1,
            weight_2,
            weight_3,
        ],
        dtype=float,
    )


def quad_shape_functions(
    xi: float,
    eta: float,
) -> np.ndarray:

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
) -> tuple[np.ndarray, np.ndarray]:

    derivative_xi = 0.25 * np.asarray(
        [
            -(1.0 - eta),
            +(1.0 - eta),
            +(1.0 + eta),
            -(1.0 + eta),
        ],
        dtype=float,
    )

    derivative_eta = 0.25 * np.asarray(
        [
            -(1.0 - xi),
            -(1.0 + xi),
            +(1.0 + xi),
            +(1.0 - xi),
        ],
        dtype=float,
    )

    return derivative_xi, derivative_eta


def bilinear_quad_weights(
    point: np.ndarray,
    vertices: np.ndarray,
) -> tuple[np.ndarray, float, float]:

    xi = 0.0
    eta = 0.0

    for _ in range(50):

        weights = quad_shape_functions(
            xi,
            eta,
        )

        mapped_point = (
            weights[:, None]
            * vertices
        ).sum(axis=0)

        residual = (
            mapped_point
            - point
        )

        if (
            np.linalg.norm(residual)
            < 1.0e-13
        ):
            break

        (
            derivative_xi,
            derivative_eta,
        ) = quad_shape_derivatives(
            xi,
            eta,
        )

        jacobian = np.asarray(
            [
                [
                    np.sum(
                        derivative_xi
                        * vertices[:, 0]
                    ),
                    np.sum(
                        derivative_eta
                        * vertices[:, 0]
                    ),
                ],
                [
                    np.sum(
                        derivative_xi
                        * vertices[:, 1]
                    ),
                    np.sum(
                        derivative_eta
                        * vertices[:, 1]
                    ),
                ],
            ],
            dtype=float,
        )

        if (
            abs(np.linalg.det(jacobian))
            < 1.0e-20
        ):
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

    number_vertices = len(vertices)

    if number_vertices == 3:

        weights = triangle_barycentric_weights(
            point,
            vertices,
        )

        if np.min(weights) < -1.0e-6:
            raise RuntimeError(
                "Point appears outside triangular element."
            )

        weights[
            np.abs(weights) < 1.0e-12
        ] = 0.0

        weights /= np.sum(weights)

        return weights, "triangle barycentric"

    if number_vertices != 4:
        raise RuntimeError(
            f"Unsupported element with "
            f"{number_vertices} vertices."
        )

    try:

        weights, xi, eta = (
            bilinear_quad_weights(
                point,
                vertices,
            )
        )

        if (
            abs(xi) <= 1.0001
            and abs(eta) <= 1.0001
            and np.min(weights) >= -1.0e-6
        ):

            weights[
                np.abs(weights)
                < 1.0e-12
            ] = 0.0

            weights /= np.sum(weights)

            method = (
                "quadrilateral bilinear "
                f"(xi={xi:.8f}, "
                f"eta={eta:.8f})"
            )

            return weights, method

    except Exception:
        pass

    triangle_candidates = [
        [0, 1, 2],
        [0, 2, 3],
    ]

    best_weights = None
    best_minimum_weight = -np.inf

    for triangle_indices in (
        triangle_candidates
    ):

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

        if (
            minimum_weight
            > best_minimum_weight
        ):

            full_weights = np.zeros(
                4,
                dtype=float,
            )

            full_weights[
                triangle_indices
            ] = triangle_weights

            best_weights = full_weights
            best_minimum_weight = (
                minimum_weight
            )

    if best_weights is None:
        raise RuntimeError(
            "Could not determine quad weights."
        )

    if np.min(best_weights) < -1.0e-6:
        raise RuntimeError(
            "Point appears outside quadrilateral."
        )

    best_weights[
        np.abs(best_weights)
        < 1.0e-12
    ] = 0.0

    best_weights /= np.sum(
        best_weights
    )

    return (
        best_weights,
        "quadrilateral triangular fallback",
    )


# ============================================================
# 4. Read meshes
# ============================================================

def read_2dm(path: Path):

    nodes = {}
    elements = {}

    element_ids_in_order = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:

        for raw_line in file:

            parts = (
                raw_line
                .strip()
                .split()
            )

            if not parts:
                continue

            record_type = (
                parts[0].upper()
            )

            if record_type == "ND":

                node_id = int(parts[1])

                nodes[node_id] = {
                    "x": float(parts[2]),
                    "y": float(parts[3]),
                    "bed": float(parts[4]),
                }

            elif record_type == "E3T":

                element_id = int(parts[1])

                elements[element_id] = [
                    int(parts[2]),
                    int(parts[3]),
                    int(parts[4]),
                ]

                element_ids_in_order.append(
                    element_id
                )

            elif record_type == "E4Q":

                element_id = int(parts[1])

                elements[element_id] = [
                    int(parts[2]),
                    int(parts[3]),
                    int(parts[4]),
                    int(parts[5]),
                ]

                element_ids_in_order.append(
                    element_id
                )

    element_to_cell_index = {
        element_id: index
        for index, element_id
        in enumerate(
            element_ids_in_order
        )
    }

    return (
        nodes,
        elements,
        element_ids_in_order,
        element_to_cell_index,
    )


def read_hgrid(path: Path):

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:

        title = file.readline().strip()
        header = file.readline().split()

        number_elements = int(header[0])
        number_nodes = int(header[1])

        nodes = {}

        for node_index in range(
            number_nodes
        ):

            parts = file.readline().split()

            node_id = int(parts[0])
            depth_positive = float(parts[3])

            nodes[node_id] = {
                "index": node_index,
                "x": float(parts[1]),
                "y": float(parts[2]),
                "bed_z": -depth_positive,
            }

        elements = {}

        for _ in range(
            number_elements
        ):

            parts = file.readline().split()

            element_id = int(parts[0])
            number_vertices = int(parts[1])

            elements[element_id] = [
                int(value)
                for value in parts[
                    2:2 + number_vertices
                ]
            ]

    return title, nodes, elements


def element_contains_edge(
    element_node_ids: list[int],
    node_id_1: int,
    node_id_2: int,
) -> bool:

    number_nodes = len(
        element_node_ids
    )

    for node_position in range(
        number_nodes
    ):

        first_node = element_node_ids[
            node_position
        ]

        second_node = element_node_ids[
            (node_position + 1)
            % number_nodes
        ]

        if (
            first_node == node_id_1
            and second_node == node_id_2
        ):
            return True

        if (
            first_node == node_id_2
            and second_node == node_id_1
        ):
            return True

    return False


(
    tuflow_nodes,
    tuflow_elements,
    tuflow_element_order,
    element_to_cell_index,
) = read_2dm(
    TUFLOW_GRID
)

(
    hgrid_title,
    schism_nodes,
    schism_elements,
) = read_hgrid(
    SCHISM_HGRID
)


# ============================================================
# 5. Read TUFLOW cell coordinates and static metadata
# ============================================================

with Dataset(
    TUFLOW_NC,
    "r",
) as dataset:

    required_metadata = [
        "cell_X",
        "cell_Y",
        "cell_Zb",
        "NL",
        "idx3",
        "ResTime",
    ]

    missing_metadata = [
        name
        for name in required_metadata
        if name not in dataset.variables
    ]

    if missing_metadata:
        raise KeyError(
            "Missing TUFLOW metadata: "
            + ", ".join(missing_metadata)
        )

    cell_x = to_nan(
        dataset.variables["cell_X"][:]
    )

    cell_y = to_nan(
        dataset.variables["cell_Y"][:]
    )

    cell_bed_z = to_nan(
        dataset.variables["cell_Zb"][:]
    )

    number_layers = np.asarray(
        dataset.variables["NL"][:],
        dtype=int,
    )

    index_2d_to_3d_one_based = (
        np.asarray(
            dataset.variables["idx3"][:],
            dtype=int,
        )
    )

    tuflow_time_hours = to_nan(
        dataset.variables["ResTime"][:]
    )


tuflow_times = pd.DatetimeIndex(
    TUFLOW_TIME_ORIGIN
    + pd.to_timedelta(
        tuflow_time_hours,
        unit="h",
    )
)


# ============================================================
# 6. Build six exact same-coordinate segment definitions
# ============================================================

segments = []

all_required_node_ids = set()


for segment_index in range(
    len(OCEAN_NODE_IDS) - 1
):

    node_id_1 = OCEAN_NODE_IDS[
        segment_index
    ]

    node_id_2 = OCEAN_NODE_IDS[
        segment_index + 1
    ]

    adjacent_elements = []

    for element_id, node_ids in (
        tuflow_elements.items()
    ):

        if element_contains_edge(
            node_ids,
            node_id_1,
            node_id_2,
        ):
            adjacent_elements.append(
                element_id
            )

    if len(adjacent_elements) != 1:
        raise RuntimeError(
            "Expected exactly one cell beside "
            f"Ocean segment "
            f"{node_id_1}-{node_id_2}, "
            f"found {adjacent_elements}"
        )

    element_id = adjacent_elements[0]

    if element_id not in schism_elements:
        raise RuntimeError(
            f"Element {element_id} missing "
            "from SCHISM hgrid."
        )

    tuflow_element_nodes = (
        tuflow_elements[element_id]
    )

    schism_element_nodes = (
        schism_elements[element_id]
    )

    if (
        set(tuflow_element_nodes)
        != set(schism_element_nodes)
    ):
        raise RuntimeError(
            f"Connectivity mismatch for "
            f"element {element_id}."
        )

    cell_index = (
        element_to_cell_index[
            element_id
        ]
    )

    target_point = np.asarray(
        [
            cell_x[cell_index],
            cell_y[cell_index],
        ],
        dtype=float,
    )

    point_1 = np.asarray(
        [
            tuflow_nodes[
                node_id_1
            ]["x"],
            tuflow_nodes[
                node_id_1
            ]["y"],
        ],
        dtype=float,
    )

    point_2 = np.asarray(
        [
            tuflow_nodes[
                node_id_2
            ]["x"],
            tuflow_nodes[
                node_id_2
            ]["y"],
        ],
        dtype=float,
    )

    segment_midpoint = (
        0.5
        * (
            point_1
            + point_2
        )
    )

    tangent_degrees = (
        point_2 - point_1
    )

    mean_latitude_radians = np.deg2rad(
        segment_midpoint[1]
    )

    tangent_metres = np.asarray(
        [
            (
                tangent_degrees[0]
                * 111_320.0
                * np.cos(
                    mean_latitude_radians
                )
            ),
            (
                tangent_degrees[1]
                * 110_540.0
            ),
        ],
        dtype=float,
    )

    segment_length_m = float(
        np.linalg.norm(
            tangent_metres
        )
    )

    normal_candidate = np.asarray(
        [
            -tangent_metres[1],
            tangent_metres[0],
        ],
        dtype=float,
    )

    normal_candidate /= np.linalg.norm(
        normal_candidate
    )

    interior_vector_degrees = (
        target_point
        - segment_midpoint
    )

    interior_vector_metres = np.asarray(
        [
            (
                interior_vector_degrees[0]
                * 111_320.0
                * np.cos(
                    mean_latitude_radians
                )
            ),
            (
                interior_vector_degrees[1]
                * 110_540.0
            ),
        ],
        dtype=float,
    )

    if np.dot(
        normal_candidate,
        interior_vector_metres,
    ) < 0.0:

        normal_candidate *= -1.0

    schism_node_ids = (
        schism_element_nodes
    )

    schism_vertices = np.asarray(
        [
            [
                schism_nodes[node_id]["x"],
                schism_nodes[node_id]["y"],
            ]
            for node_id in schism_node_ids
        ],
        dtype=float,
    )

    (
        horizontal_weights,
        interpolation_method,
    ) = determine_horizontal_weights(
        target_point,
        schism_vertices,
    )

    schism_bed_values = np.asarray(
        [
            schism_nodes[node_id][
                "bed_z"
            ]
            for node_id in schism_node_ids
        ],
        dtype=float,
    )

    schism_point_bed_z, _ = (
        weighted_finite_value(
            schism_bed_values,
            horizontal_weights,
        )
    )

    tuflow_point_bed_z = float(
        cell_bed_z[cell_index]
    )

    common_bed_z = (
        0.5
        * (
            tuflow_point_bed_z
            + schism_point_bed_z
        )
    )

    all_required_node_ids.update(
        schism_node_ids
    )

    segments.append(
        {
            "segment_number": (
                segment_index + 1
            ),
            "boundary_node_1": node_id_1,
            "boundary_node_2": node_id_2,
            "element_id": element_id,
            "cell_index": cell_index,
            "cell_number_one_based": (
                cell_index + 1
            ),
            "cell_x": float(
                target_point[0]
            ),
            "cell_y": float(
                target_point[1]
            ),
            "length_m": (
                segment_length_m
            ),
            "normal_x": float(
                normal_candidate[0]
            ),
            "normal_y": float(
                normal_candidate[1]
            ),
            "schism_node_ids": (
                schism_node_ids
            ),
            "horizontal_weights": (
                horizontal_weights
            ),
            "interpolation_method": (
                interpolation_method
            ),
            "tuflow_bed_z": (
                tuflow_point_bed_z
            ),
            "schism_bed_z": (
                schism_point_bed_z
            ),
            "common_bed_z": (
                common_bed_z
            ),
        }
    )


required_node_ids = sorted(
    all_required_node_ids
)

required_global_node_indices = [
    schism_nodes[node_id]["index"]
    for node_id in required_node_ids
]

subset_position_by_node_id = {
    node_id: position
    for position, node_id in enumerate(
        required_node_ids
    )
}

for segment in segments:

    segment[
        "subset_node_positions"
    ] = [
        subset_position_by_node_id[
            node_id
        ]
        for node_id
        in segment["schism_node_ids"]
    ]


# ============================================================
# 7. Write segment-definition report
# ============================================================

segment_lines = [
    "===== SAME-COORDINATE OCEAN SEGMENTS =====",
    "",
    (
        "Positive normal velocity means movement "
        "from the Ocean boundary into the model domain."
    ),
    "",
]


for segment in segments:

    segment_lines.extend(
        [
            (
                f"Segment "
                f"{segment['segment_number']}"
            ),
            (
                f"  Boundary edge: "
                f"{segment['boundary_node_1']} -> "
                f"{segment['boundary_node_2']}"
            ),
            (
                f"  Element ID: "
                f"{segment['element_id']}"
            ),
            (
                f"  TUFLOW cell number: "
                f"{segment['cell_number_one_based']}"
            ),
            (
                f"  Exact coordinate: "
                f"({segment['cell_x']:.10f}, "
                f"{segment['cell_y']:.10f})"
            ),
            (
                f"  Segment length: "
                f"{segment['length_m']:.6f} m"
            ),
            (
                f"  Inward normal: "
                f"({segment['normal_x']:.10f}, "
                f"{segment['normal_y']:.10f})"
            ),
            (
                f"  SCHISM node IDs: "
                f"{segment['schism_node_ids']}"
            ),
            (
                "  SCHISM interpolation weights: "
                + str(
                    np.round(
                        segment[
                            "horizontal_weights"
                        ],
                        10,
                    ).tolist()
                )
            ),
            (
                f"  Interpolation method: "
                f"{segment['interpolation_method']}"
            ),
            (
                f"  TUFLOW bed z: "
                f"{segment['tuflow_bed_z']:.8f} m"
            ),
            (
                f"  SCHISM bed z: "
                f"{segment['schism_bed_z']:.8f} m"
            ),
            (
                f"  Common bed z: "
                f"{segment['common_bed_z']:.8f} m"
            ),
            (
                f"  Bed difference: "
                f"{segment['schism_bed_z'] - segment['tuflow_bed_z']:.8f} m"
            ),
            "",
        ]
    )


SEGMENT_REPORT.write_text(
    "\n".join(segment_lines) + "\n",
    encoding="utf-8",
)


# ============================================================
# 8. Load SCHISM selected-node outputs
# ============================================================

def load_schism_variable(
    prefix: str,
    variable_name: str,
) -> tuple[pd.DatetimeIndex, np.ndarray]:

    stack_numbers = find_stack_numbers(
        SCHISM_OUTPUTS,
        prefix,
    )

    expected_stack_numbers = list(
        range(1, 16)
    )

    if (
        stack_numbers
        != expected_stack_numbers
    ):
        raise RuntimeError(
            f"Unexpected stacks for {prefix}: "
            f"{stack_numbers}"
        )

    time_parts = []
    data_parts = []

    for stack_number in stack_numbers:

        file_path = (
            SCHISM_OUTPUTS
            / f"{prefix}_{stack_number}.nc"
        )

        with Dataset(
            file_path,
            "r",
        ) as dataset:

            seconds = to_nan(
                dataset.variables["time"][:]
            )

            current_times = (
                SCHISM_MODEL_START
                + pd.to_timedelta(
                    seconds,
                    unit="s",
                )
            )

            raw_values = to_nan(
                dataset.variables[
                    variable_name
                ][:]
            )

            if raw_values.ndim == 2:

                selected_values = (
                    raw_values[
                        :,
                        required_global_node_indices,
                    ]
                )

            elif raw_values.ndim == 3:

                selected_values = (
                    raw_values[
                        :,
                        required_global_node_indices,
                        :,
                    ]
                )

            else:

                raise RuntimeError(
                    f"Unexpected shape for "
                    f"{variable_name}: "
                    f"{raw_values.shape}"
                )

        time_parts.append(
            pd.DatetimeIndex(
                current_times
            )
        )

        data_parts.append(
            selected_values
        )

    times = pd.DatetimeIndex(
        np.concatenate(
            [
                time_part.to_numpy()
                for time_part in time_parts
            ]
        )
    )

    values = np.concatenate(
        data_parts,
        axis=0,
    )

    if times.duplicated().any():
        raise RuntimeError(
            f"Duplicate SCHISM times in {prefix}"
        )

    return times, values


print("===== LOADING SCHISM OUTPUTS =====")


schism_times, schism_elevation = (
    load_schism_variable(
        "out2d",
        "elevation",
    )
)

_, schism_z = load_schism_variable(
    "zCoordinates",
    "zCoordinates",
)

_, schism_velocity_x = (
    load_schism_variable(
        "horizontalVelX",
        "horizontalVelX",
    )
)

_, schism_velocity_y = (
    load_schism_variable(
        "horizontalVelY",
        "horizontalVelY",
    )
)

_, schism_salinity = (
    load_schism_variable(
        "salinity",
        "salinity",
    )
)

_, schism_temperature = (
    load_schism_variable(
        "temperature",
        "temperature",
    )
)


# ============================================================
# 9. Align exact common times
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

if len(common_times) != EXPECTED_RECORDS:
    raise RuntimeError(
        "Unexpected common record count:\n"
        f"Expected: {EXPECTED_RECORDS}\n"
        f"Found   : {len(common_times)}"
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

print()
print("===== COMMON TIME RANGE =====")
print(f"Records: {len(common_times)}")
print(f"First  : {common_times[0]}")
print(f"Last   : {common_times[-1]}")


# ============================================================
# 10. TUFLOW vertical indexing
# ============================================================

face_count_per_cell = (
    number_layers + 1
)

face_start_zero_based = np.concatenate(
    (
        np.asarray(
            [0],
            dtype=int,
        ),
        np.cumsum(
            face_count_per_cell[:-1],
            dtype=int,
        ),
    )
)


# ============================================================
# 11. Extract both models at exactly the same coordinates
# ============================================================

segment_records = []


with Dataset(
    TUFLOW_NC,
    "r",
) as dataset:

    required_variables = [
        "H",
        "layerface_Z",
        "V_x",
        "V_y",
        "SAL",
        "TEMP",
    ]

    missing_variables = [
        variable_name
        for variable_name
        in required_variables
        if variable_name
        not in dataset.variables
    ]

    if missing_variables:
        raise KeyError(
            "Missing TUFLOW variables: "
            + ", ".join(
                missing_variables
            )
        )

    for common_time_position, timestamp in enumerate(
        common_times
    ):

        tuflow_time_index = int(
            tuflow_time_indices[
                common_time_position
            ]
        )

        schism_time_index = int(
            schism_time_indices[
                common_time_position
            ]
        )

        for segment in segments:

            cell_index = int(
                segment["cell_index"]
            )

            number_vertical_layers = int(
                number_layers[
                    cell_index
                ]
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

            face_start = int(
                face_start_zero_based[
                    cell_index
                ]
            )

            face_stop = (
                face_start
                + number_vertical_layers
                + 1
            )

            tuflow_water_level = float(
                to_nan(
                    dataset.variables[
                        "H"
                    ][
                        tuflow_time_index,
                        cell_index,
                    ]
                )
            )

            node_positions = segment[
                "subset_node_positions"
            ]

            horizontal_weights = np.asarray(
                segment[
                    "horizontal_weights"
                ],
                dtype=float,
            )

            (
                schism_water_level,
                schism_elevation_support,
            ) = weighted_finite_value(
                schism_elevation[
                    schism_time_index,
                    node_positions,
                ],
                horizontal_weights,
            )

            common_surface_z = (
                0.5
                * (
                    tuflow_water_level
                    + schism_water_level
                )
            )

            common_bed_z = float(
                segment["common_bed_z"]
            )

            water_depth = (
                common_surface_z
                - common_bed_z
            )

            if (
                not np.isfinite(
                    water_depth
                )
                or water_depth
                <= (
                    SURFACE_OFFSET_M
                    + BOTTOM_OFFSET_M
                )
            ):
                continue

            target_z_by_level = {
                "Surface": (
                    common_surface_z
                    - SURFACE_OFFSET_M
                ),
                "Mid-depth": (
                    0.5
                    * (
                        common_surface_z
                        + common_bed_z
                    )
                ),
                "Near-bottom": (
                    common_bed_z
                    + BOTTOM_OFFSET_M
                ),
            }

            layer_faces = to_nan(
                dataset.variables[
                    "layerface_Z"
                ][
                    tuflow_time_index,
                    face_start:face_stop,
                ]
            )

            layer_centres = (
                0.5
                * (
                    layer_faces[:-1]
                    + layer_faces[1:]
                )
            )

            tuflow_profiles = {
                "velocity_x": to_nan(
                    dataset.variables[
                        "V_x"
                    ][
                        tuflow_time_index,
                        cell_3d_start:
                        cell_3d_stop,
                    ]
                ),
                "velocity_y": to_nan(
                    dataset.variables[
                        "V_y"
                    ][
                        tuflow_time_index,
                        cell_3d_start:
                        cell_3d_stop,
                    ]
                ),
                "salinity": to_nan(
                    dataset.variables[
                        "SAL"
                    ][
                        tuflow_time_index,
                        cell_3d_start:
                        cell_3d_stop,
                    ]
                ),
                "temperature": to_nan(
                    dataset.variables[
                        "TEMP"
                    ][
                        tuflow_time_index,
                        cell_3d_start:
                        cell_3d_stop,
                    ]
                ),
            }

            schism_node_z_profiles = (
                schism_z[
                    schism_time_index,
                    node_positions,
                    :,
                ]
            )

            schism_profiles = {
                "velocity_x": (
                    schism_velocity_x[
                        schism_time_index,
                        node_positions,
                        :,
                    ]
                ),
                "velocity_y": (
                    schism_velocity_y[
                        schism_time_index,
                        node_positions,
                        :,
                    ]
                ),
                "salinity": (
                    schism_salinity[
                        schism_time_index,
                        node_positions,
                        :,
                    ]
                ),
                "temperature": (
                    schism_temperature[
                        schism_time_index,
                        node_positions,
                        :,
                    ]
                ),
            }

            for level_name in LEVEL_NAMES:

                target_z = float(
                    target_z_by_level[
                        level_name
                    ]
                )

                # --------------------------------------------
                # TUFLOW at the exact TUFLOW cell centre
                # --------------------------------------------

                tuflow_velocity_x_value = (
                    vertical_interpolate(
                        layer_centres,
                        tuflow_profiles[
                            "velocity_x"
                        ],
                        target_z,
                    )
                )

                tuflow_velocity_y_value = (
                    vertical_interpolate(
                        layer_centres,
                        tuflow_profiles[
                            "velocity_y"
                        ],
                        target_z,
                    )
                )

                tuflow_salinity_value = (
                    vertical_interpolate(
                        layer_centres,
                        tuflow_profiles[
                            "salinity"
                        ],
                        target_z,
                    )
                )

                tuflow_temperature_value = (
                    vertical_interpolate(
                        layer_centres,
                        tuflow_profiles[
                            "temperature"
                        ],
                        target_z,
                    )
                )

                if (
                    np.isfinite(
                        tuflow_velocity_x_value
                    )
                    and np.isfinite(
                        tuflow_velocity_y_value
                    )
                ):

                    tuflow_normal_velocity = (
                        tuflow_velocity_x_value
                        * segment["normal_x"]
                        + tuflow_velocity_y_value
                        * segment["normal_y"]
                    )

                else:

                    tuflow_normal_velocity = np.nan

                # --------------------------------------------
                # SCHISM interpolated to same cell centre
                # --------------------------------------------

                (
                    schism_velocity_x_value,
                    support_x,
                ) = interpolate_schism_point_value(
                    schism_node_z_profiles,
                    schism_profiles[
                        "velocity_x"
                    ],
                    horizontal_weights,
                    target_z,
                )

                (
                    schism_velocity_y_value,
                    support_y,
                ) = interpolate_schism_point_value(
                    schism_node_z_profiles,
                    schism_profiles[
                        "velocity_y"
                    ],
                    horizontal_weights,
                    target_z,
                )

                (
                    schism_salinity_value,
                    support_salinity,
                ) = interpolate_schism_point_value(
                    schism_node_z_profiles,
                    schism_profiles[
                        "salinity"
                    ],
                    horizontal_weights,
                    target_z,
                )

                (
                    schism_temperature_value,
                    support_temperature,
                ) = interpolate_schism_point_value(
                    schism_node_z_profiles,
                    schism_profiles[
                        "temperature"
                    ],
                    horizontal_weights,
                    target_z,
                )

                if (
                    np.isfinite(
                        schism_velocity_x_value
                    )
                    and np.isfinite(
                        schism_velocity_y_value
                    )
                ):

                    schism_normal_velocity = (
                        schism_velocity_x_value
                        * segment["normal_x"]
                        + schism_velocity_y_value
                        * segment["normal_y"]
                    )

                else:

                    schism_normal_velocity = np.nan

                for (
                    model_name,
                    normal_velocity,
                    salinity_value,
                    temperature_value,
                    support,
                ) in [
                    (
                        "TUFLOW",
                        tuflow_normal_velocity,
                        tuflow_salinity_value,
                        tuflow_temperature_value,
                        number_vertical_layers,
                    ),
                    (
                        "SCHISM",
                        schism_normal_velocity,
                        schism_salinity_value,
                        schism_temperature_value,
                        min(
                            support_x,
                            support_y,
                            support_salinity,
                            support_temperature,
                        ),
                    ),
                ]:

                    if (
                        np.isfinite(
                            normal_velocity
                        )
                        and np.isfinite(
                            salinity_value
                        )
                    ):

                        salt_advection_index = (
                            normal_velocity
                            * salinity_value
                        )

                    else:

                        salt_advection_index = np.nan

                    segment_records.append(
                        {
                            "time_aest": timestamp,
                            "model": model_name,
                            "level": level_name,
                            "segment_number": (
                                segment[
                                    "segment_number"
                                ]
                            ),
                            "element_id": (
                                segment[
                                    "element_id"
                                ]
                            ),
                            "cell_number_one_based": (
                                segment[
                                    "cell_number_one_based"
                                ]
                            ),
                            "cell_x": (
                                segment["cell_x"]
                            ),
                            "cell_y": (
                                segment["cell_y"]
                            ),
                            "segment_length_m": (
                                segment[
                                    "length_m"
                                ]
                            ),
                            "normal_x": (
                                segment["normal_x"]
                            ),
                            "normal_y": (
                                segment["normal_y"]
                            ),
                            "tuflow_water_level_m": (
                                tuflow_water_level
                            ),
                            "schism_water_level_m": (
                                schism_water_level
                            ),
                            "common_surface_z_m": (
                                common_surface_z
                            ),
                            "common_bed_z_m": (
                                common_bed_z
                            ),
                            "target_z_m": (
                                target_z
                            ),
                            "normal_velocity_inward_positive_ms": (
                                normal_velocity
                            ),
                            "salinity_psu": (
                                salinity_value
                            ),
                            "temperature_degC": (
                                temperature_value
                            ),
                            "salt_advection_index_psu_ms": (
                                salt_advection_index
                            ),
                            "valid_support": int(
                                support
                            ),
                        }
                    )


segment_df = pd.DataFrame(
    segment_records
)

segment_df = segment_df.sort_values(
    [
        "model",
        "level",
        "segment_number",
        "time_aest",
    ]
).reset_index(drop=True)

segment_df.to_csv(
    SEGMENT_CSV,
    index=False,
    float_format="%.10f",
)


# ============================================================
# 12. Build length-weighted six-cell section time series
# ============================================================

section_records = []


for (
    timestamp,
    model_name,
    level_name,
), group in segment_df.groupby(
    [
        "time_aest",
        "model",
        "level",
    ],
    sort=True,
):

    weights = group[
        "segment_length_m"
    ].to_list()

    normal_velocity = (
        weighted_finite_mean(
            group[
                "normal_velocity_inward_positive_ms"
            ].to_list(),
            weights,
        )
    )

    salinity_value = (
        weighted_finite_mean(
            group[
                "salinity_psu"
            ].to_list(),
            weights,
        )
    )

    temperature_value = (
        weighted_finite_mean(
            group[
                "temperature_degC"
            ].to_list(),
            weights,
        )
    )

    salt_advection_index = (
        weighted_finite_mean(
            group[
                "salt_advection_index_psu_ms"
            ].to_list(),
            weights,
        )
    )

    section_records.append(
        {
            "time_aest": timestamp,
            "model": model_name,
            "level": level_name,
            "normal_velocity_inward_positive_ms": (
                normal_velocity
            ),
            "salinity_psu": salinity_value,
            "temperature_degC": (
                temperature_value
            ),
            "salt_advection_index_psu_ms": (
                salt_advection_index
            ),
            "valid_segment_count": int(
                group[
                    "normal_velocity_inward_positive_ms"
                ].notna().sum()
            ),
        }
    )


section_df = pd.DataFrame(
    section_records
)

section_df = section_df.sort_values(
    [
        "model",
        "level",
        "time_aest",
    ]
).reset_index(drop=True)

section_df.to_csv(
    SECTION_CSV,
    index=False,
    float_format="%.10f",
)


# ============================================================
# 13. Summary statistics
# ============================================================

summary_records = []


for (
    model_name,
    level_name,
), group in section_df.groupby(
    [
        "model",
        "level",
    ],
    sort=False,
):

    velocity = group[
        "normal_velocity_inward_positive_ms"
    ].dropna()

    inflow_mask = (
        group[
            "normal_velocity_inward_positive_ms"
        ] > 0.0
    )

    outflow_mask = (
        group[
            "normal_velocity_inward_positive_ms"
        ] < 0.0
    )

    summary_records.append(
        {
            "model": model_name,
            "level": level_name,
            "record_count": int(
                velocity.count()
            ),
            "mean_normal_velocity_ms": float(
                velocity.mean()
            ),
            "minimum_normal_velocity_ms": float(
                velocity.min()
            ),
            "maximum_normal_velocity_ms": float(
                velocity.max()
            ),
            "inward_time_fraction": float(
                np.mean(velocity > 0.0)
            ),
            "outward_time_fraction": float(
                np.mean(velocity < 0.0)
            ),
            "mean_salinity_all_psu": float(
                group[
                    "salinity_psu"
                ].mean()
            ),
            "mean_salinity_during_inflow_psu": (
                float(
                    group.loc[
                        inflow_mask,
                        "salinity_psu",
                    ].mean()
                )
                if np.any(inflow_mask)
                else np.nan
            ),
            "mean_salinity_during_outflow_psu": (
                float(
                    group.loc[
                        outflow_mask,
                        "salinity_psu",
                    ].mean()
                )
                if np.any(outflow_mask)
                else np.nan
            ),
            "mean_temperature_all_degC": float(
                group[
                    "temperature_degC"
                ].mean()
            ),
            "mean_salt_advection_index_psu_ms": float(
                group[
                    "salt_advection_index_psu_ms"
                ].mean()
            ),
            "minimum_valid_segment_count": int(
                group[
                    "valid_segment_count"
                ].min()
            ),
            "maximum_valid_segment_count": int(
                group[
                    "valid_segment_count"
                ].max()
            ),
        }
    )


summary_df = pd.DataFrame(
    summary_records
)

summary_df.to_csv(
    SUMMARY_CSV,
    index=False,
    float_format="%.10f",
)


# ============================================================
# 14. Per-segment summary
# ============================================================

segment_summary_records = []


for (
    model_name,
    level_name,
    segment_number,
), group in segment_df.groupby(
    [
        "model",
        "level",
        "segment_number",
    ],
    sort=False,
):

    velocity = group[
        "normal_velocity_inward_positive_ms"
    ].dropna()

    segment_summary_records.append(
        {
            "model": model_name,
            "level": level_name,
            "segment_number": (
                segment_number
            ),
            "element_id": int(
                group[
                    "element_id"
                ].iloc[0]
            ),
            "cell_number_one_based": int(
                group[
                    "cell_number_one_based"
                ].iloc[0]
            ),
            "cell_x": float(
                group["cell_x"].iloc[0]
            ),
            "cell_y": float(
                group["cell_y"].iloc[0]
            ),
            "record_count": int(
                velocity.count()
            ),
            "mean_normal_velocity_ms": float(
                velocity.mean()
            ),
            "inward_time_fraction": float(
                np.mean(velocity > 0.0)
            ),
            "mean_salinity_psu": float(
                group[
                    "salinity_psu"
                ].mean()
            ),
            "mean_salt_advection_index_psu_ms": float(
                group[
                    "salt_advection_index_psu_ms"
                ].mean()
            ),
        }
    )


segment_summary_df = pd.DataFrame(
    segment_summary_records
)

segment_summary_df.to_csv(
    SEGMENT_SUMMARY_CSV,
    index=False,
    float_format="%.10f",
)


# ============================================================
# 15. Shared plotting utilities
# ============================================================

MODEL_LINE_WIDTH = 1.15
ZERO_LINE_WIDTH = 0.75
GRID_ALPHA = 0.28


def add_shared_model_legend(
    fig,
    handles,
) -> None:
    """
    Add one shared legend in the lower-left margin of the figure.

    The legend is outside the axes so it does not cover the time-series
    curves. Only one legend is used for all three vertical levels.
    """

    fig.legend(
        handles=handles,
        labels=["TUFLOW FV", "SCHISM"],
        loc="lower left",
        bbox_to_anchor=(0.085, 0.012),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.35",
        fontsize=9.5,
        borderpad=0.55,
        labelspacing=0.35,
        handlelength=2.4,
        columnspacing=1.4,
    )


# ============================================================
# 16. Plot same-coordinate normal velocity comparison
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 10),
    sharex=True,
)

legend_handles = None

for axis, level_name in zip(
    axes,
    LEVEL_NAMES,
):

    current_handles = []

    for model_name in [
        "TUFLOW",
        "SCHISM",
    ]:

        group = section_df[
            (
                section_df["model"]
                == model_name
            )
            & (
                section_df["level"]
                == level_name
            )
        ].sort_values(
            "time_aest"
        )

        line, = axis.plot(
            group["time_aest"],
            group[
                "normal_velocity_inward_positive_ms"
            ],
            linewidth=MODEL_LINE_WIDTH,
        )

        current_handles.append(line)

    if legend_handles is None:
        legend_handles = current_handles

    axis.axhline(
        0.0,
        linewidth=ZERO_LINE_WIDTH,
    )

    tuflow_mean = summary_df[
        (
            summary_df["model"]
            == "TUFLOW"
        )
        & (
            summary_df["level"]
            == level_name
        )
    ][
        "mean_normal_velocity_ms"
    ].iloc[0]

    schism_mean = summary_df[
        (
            summary_df["model"]
            == "SCHISM"
        )
        & (
            summary_df["level"]
            == level_name
        )
    ][
        "mean_normal_velocity_ms"
    ].iloc[0]

    axis.set_title(
        f"{level_name}: "
        f"TUFLOW FV mean = "
        f"{tuflow_mean:.5f} m s$^{{-1}}$; "
        f"SCHISM mean = "
        f"{schism_mean:.5f} m s$^{{-1}}$",
        fontsize=10.5,
    )

    axis.set_ylabel(
        "Normal velocity\n(m s$^{-1}$)"
    )

    axis.grid(
        True,
        alpha=GRID_ALPHA,
    )

axes[-1].set_xlabel(
    "Time (AEST)"
)

configure_time_axis(
    axes[-1]
)

fig.suptitle(
    "Ocean residual normal velocity at identical coordinates\n"
    "Positive values indicate flow toward the model interior",
    fontsize=14,
)

add_shared_model_legend(
    fig,
    legend_handles,
)

fig.tight_layout(
    rect=[0, 0.065, 1, 0.94]
)

fig.savefig(
    OUTPUT_DIR
    / "same_coordinate_normal_velocity_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 17. Plot same-coordinate salinity comparison
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 10),
    sharex=True,
)

legend_handles = None

for axis, level_name in zip(
    axes,
    LEVEL_NAMES,
):

    current_handles = []

    for model_name in [
        "TUFLOW",
        "SCHISM",
    ]:

        group = section_df[
            (
                section_df["model"]
                == model_name
            )
            & (
                section_df["level"]
                == level_name
            )
        ].sort_values(
            "time_aest"
        )

        line, = axis.plot(
            group["time_aest"],
            group[
                "salinity_psu"
            ],
            linewidth=MODEL_LINE_WIDTH,
        )

        current_handles.append(line)

    if legend_handles is None:
        legend_handles = current_handles

    axis.set_title(
        level_name,
        fontsize=10.5,
    )

    axis.set_ylabel(
        "Salinity (psu)"
    )

    axis.grid(
        True,
        alpha=GRID_ALPHA,
    )

axes[-1].set_xlabel(
    "Time (AEST)"
)

configure_time_axis(
    axes[-1]
)

fig.suptitle(
    "Ocean salinity comparison at identical coordinates",
    fontsize=14,
)

add_shared_model_legend(
    fig,
    legend_handles,
)

fig.tight_layout(
    rect=[0, 0.065, 1, 0.95]
)

fig.savefig(
    OUTPUT_DIR
    / "same_coordinate_salinity_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 18. Plot same-coordinate salt-advection comparison
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 10),
    sharex=True,
)

legend_handles = None

for axis, level_name in zip(
    axes,
    LEVEL_NAMES,
):

    current_handles = []

    for model_name in [
        "TUFLOW",
        "SCHISM",
    ]:

        group = section_df[
            (
                section_df["model"]
                == model_name
            )
            & (
                section_df["level"]
                == level_name
            )
        ].sort_values(
            "time_aest"
        )

        line, = axis.plot(
            group["time_aest"],
            group[
                "salt_advection_index_psu_ms"
            ],
            linewidth=MODEL_LINE_WIDTH,
        )

        current_handles.append(line)

    if legend_handles is None:
        legend_handles = current_handles

    axis.axhline(
        0.0,
        linewidth=ZERO_LINE_WIDTH,
    )

    axis.set_title(
        level_name,
        fontsize=10.5,
    )

    axis.set_ylabel(
        "Velocity × salinity\n(psu m s$^{-1}$)"
    )

    axis.grid(
        True,
        alpha=GRID_ALPHA,
    )

axes[-1].set_xlabel(
    "Time (AEST)"
)

configure_time_axis(
    axes[-1]
)

fig.suptitle(
    "Signed salt-advection index at identical coordinates\n"
    "Positive values indicate saline transport toward the model interior",
    fontsize=14,
)

add_shared_model_legend(
    fig,
    legend_handles,
)

fig.tight_layout(
    rect=[0, 0.065, 1, 0.94]
)

fig.savefig(
    OUTPUT_DIR
    / "same_coordinate_salt_advection_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 19. Write final report
# ============================================================

report_lines = [
    "===== SAME-COORDINATE OCEAN RESIDUAL COMPARISON =====",
    "",
    f"TUFLOW result: {TUFLOW_NC}",
    f"SCHISM outputs: {SCHISM_OUTPUTS}",
    f"SCHISM hgrid title: {hgrid_title}",
    "",
    f"Records: {len(common_times)}",
    f"First  : {common_times[0]}",
    f"Last   : {common_times[-1]}",
    "",
    (
        "Positive normal velocity means movement "
        "from the Ocean boundary toward the model interior."
    ),
    "",
    (
        "Both models are evaluated at the exact TUFLOW "
        "cell_X/cell_Y coordinates of the six cells "
        "immediately adjacent to the Ocean boundary."
    ),
    "",
    (
        "Common vertical targets:"
    ),
    (
        "  Common reference surface = "
        "0.5 × (TUFLOW water level + SCHISM water level)"
    ),
    (
        f"  Surface = common surface "
        f"- {SURFACE_OFFSET_M:.3f} m"
    ),
    (
        "  Mid-depth = midpoint between "
        "common surface and common bed"
    ),
    (
        f"  Near-bottom = common bed "
        f"+ {BOTTOM_OFFSET_M:.3f} m"
    ),
    "",
    (
        "Common bed = 0.5 × "
        "(TUFLOW cell bed + "
        "SCHISM interpolated bed)"
    ),
    "",
    "===== OVERALL SUMMARY =====",
    "",
    summary_df.to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.8f}"
        ),
    ),
    "",
    "===== DIRECT MEAN-VELOCITY COMPARISON =====",
    "",
]


for level_name in LEVEL_NAMES:

    tuflow_row = summary_df[
        (
            summary_df["model"]
            == "TUFLOW"
        )
        & (
            summary_df["level"]
            == level_name
        )
    ].iloc[0]

    schism_row = summary_df[
        (
            summary_df["model"]
            == "SCHISM"
        )
        & (
            summary_df["level"]
            == level_name
        )
    ].iloc[0]

    report_lines.append(
        f"{level_name}: "
        f"TUFLOW mean="
        f"{tuflow_row['mean_normal_velocity_ms']:.8f} m/s; "
        f"SCHISM mean="
        f"{schism_row['mean_normal_velocity_ms']:.8f} m/s; "
        f"difference SCHISM-TUFLOW="
        f"{schism_row['mean_normal_velocity_ms'] - tuflow_row['mean_normal_velocity_ms']:.8f} m/s"
    )


report_lines.extend(
    [
        "",
        "===== PER-SEGMENT SUMMARY =====",
        "",
        segment_summary_df.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.8f}"
            ),
        ),
        "",
        "===== OUTPUT FILES =====",
        "",
        f"Section CSV: {SECTION_CSV}",
        f"Segment CSV: {SEGMENT_CSV}",
        f"Summary CSV: {SUMMARY_CSV}",
        f"Segment summary CSV: {SEGMENT_SUMMARY_CSV}",
        f"Segment definitions: {SEGMENT_REPORT}",
    ]
)


SUMMARY_REPORT.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8",
)


# ============================================================
# 19. Final terminal output
# ============================================================

print()
print(
    "===== SAME-COORDINATE COMPARISON COMPLETE ====="
)

print()
print(
    summary_df.to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.8f}"
        ),
    )
)

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("Summary report:")
print(SUMMARY_REPORT)

print()
print("Main figure:")
print(
    OUTPUT_DIR
    / "same_coordinate_normal_velocity_comparison.png"
)