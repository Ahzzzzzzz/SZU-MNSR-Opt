"""Geometry builder for the unified MNSR-SZ reference scenario.

The model in this module is a finite-height, three-dimensional simplification
based on the stated literature reference dimensions. It intentionally contains
no optimization algorithm, settings, tallies, XML export, or OpenMC execution.
"""

from __future__ import annotations

import math
from numbers import Real

import openmc

__all__ = ["build_geometry"]


def _validate_material_map(
    material_map: dict[str, openmc.Material],
) -> None:
    """Validate the role-based material mapping used by the geometry."""
    if not isinstance(material_map, dict):
        raise TypeError("material_map must be a dictionary.")

    required_roles = {
        "fuel",
        "fuel_clad",
        "dummy",
        "tie_rod",
        "control_rod_clad",
        "control_rod_inner",
        "guide_tube",
        "vessel",
        "water",
        "beryllium",
        "cadmium",
    }
    missing_roles = required_roles.difference(material_map)
    if missing_roles:
        missing_text = ", ".join(sorted(missing_roles))
        raise KeyError(f"material_map is missing required roles: {missing_text}")

    invalid_roles = [
        role
        for role in sorted(required_roles)
        if not isinstance(material_map[role], openmc.Material)
    ]
    if invalid_roles:
        invalid_text = ", ".join(invalid_roles)
        raise TypeError(
            f"These material_map entries are not openmc.Material objects: "
            f"{invalid_text}"
        )


def _validate_ring_offsets(
    ring_offsets: list[float] | None,
    number_of_rings: int,
) -> list[float]:
    """Return one finite angular offset in radians for every ring."""
    if ring_offsets is None:
        return [0.0] * number_of_rings
    if not isinstance(ring_offsets, list):
        raise TypeError("ring_offsets must be a list of angles in radians or None.")
    if len(ring_offsets) != number_of_rings:
        raise ValueError(
            f"ring_offsets must contain exactly {number_of_rings} values; "
            f"received {len(ring_offsets)}."
        )

    validated_offsets: list[float] = []
    for ring_index, offset in enumerate(ring_offsets):
        if isinstance(offset, bool) or not isinstance(offset, Real):
            raise TypeError(
                f"ring_offsets[{ring_index}] must be a real angle in radians."
            )
        angle = float(offset)
        if not math.isfinite(angle):
            raise ValueError(f"ring_offsets[{ring_index}] must be finite.")
        validated_offsets.append(angle)

    return validated_offsets


def _validate_dummy_position_ids(
    dummy_position_ids: set[int] | None,
    default_dummy_ids: set[int],
) -> set[int]:
    """Validate or supply the five ordinary-position IDs used for dummies."""
    if dummy_position_ids is None:
        return set(default_dummy_ids)
    if not isinstance(dummy_position_ids, set):
        raise TypeError("dummy_position_ids must be a set of integers or None.")
    if any(
        isinstance(position_id, bool) or not isinstance(position_id, int)
        for position_id in dummy_position_ids
    ):
        raise TypeError("Every dummy position ID must be an integer.")
    if len(dummy_position_ids) != 5:
        raise ValueError(
            "dummy_position_ids must contain exactly five distinct IDs."
        )
    if any(position_id < 0 or position_id > 349 for position_id in dummy_position_ids):
        raise ValueError("Every dummy position ID must be in the range 0 through 349.")

    # A set cannot contain duplicate values; copying also prevents caller mutation
    # from changing the geometry metadata after construction.
    return set(dummy_position_ids)


def build_geometry(
    material_map: dict[str, openmc.Material],
    dummy_position_ids: set[int] | None = None,
    control_rod_state: str = "withdrawn",
    ring_offsets: list[float] | None = None,
) -> tuple[openmc.Geometry, dict]:
    """Build the finite-height MNSR-SZ unified reference geometry.

    Parameters
    ----------
    material_map
        Role-based materials returned by :func:`materials.build_materials`.
    dummy_position_ids
        Exactly five ordinary-position IDs to fill with solid Al-303-1 dummy
        elements. If omitted, five positions on ring 10 at local indices
        0, 13, 26, 39, and 52 are used.
    control_rod_state
        Either ``"withdrawn"`` for a water-filled central guide tube or
        ``"inserted"`` for the simplified Al/Cd/SS-304 control assembly.
    ring_offsets
        Initial azimuth of each of the eleven rings, in radians. The default
        is zero for every ring.

    Returns
    -------
    geometry
        The constructed OpenMC geometry. No XML file is exported.
    metadata
        Stable position records, role-specific position IDs and cell lookups,
        model state, dimensions, and enforced object counts.
    """
    _validate_material_map(material_map)

    ring_distribution = [1, 6, 12, 19, 26, 32, 39, 45, 52, 58, 65]
    pitch = 1.095
    assert sum(ring_distribution) == 355, "The physical-position count must be 355."

    validated_offsets = _validate_ring_offsets(
        ring_offsets,
        number_of_rings=len(ring_distribution),
    )
    if control_rod_state not in {"withdrawn", "inserted"}:
        raise ValueError(
            "control_rod_state must be either 'withdrawn' or 'inserted'."
        )

    tie_rod_local_indices = {0, 13, 26, 39}
    position_table: list[dict[str, object]] = []
    ordinary_positions: list[dict[str, object]] = []
    tie_rod_positions: list[dict[str, object]] = []
    ordinary_position_id = 0
    global_position_id = 0

    for ring_index, number_of_positions in enumerate(ring_distribution):
        ring_radius = ring_index * pitch
        initial_angle = validated_offsets[ring_index]

        for local_index in range(number_of_positions):
            angle = initial_angle + 2.0 * math.pi * local_index / number_of_positions
            x = ring_radius * math.cos(angle)
            y = ring_radius * math.sin(angle)

            is_central = ring_index == 0
            is_tie_rod = (
                ring_index == 8 and local_index in tie_rod_local_indices
            )
            record: dict[str, object] = {
                "ring_index": ring_index,
                "local_index": local_index,
                "x": x,
                "y": y,
                "global_position_id": global_position_id,
                "ordinary_position_id": None,
                "position_type": (
                    "control_rod"
                    if is_central
                    else "tie_rod"
                    if is_tie_rod
                    else "ordinary"
                ),
            }

            if is_tie_rod:
                tie_rod_positions.append(record)
            elif not is_central:
                record["ordinary_position_id"] = ordinary_position_id
                ordinary_positions.append(record)
                ordinary_position_id += 1

            position_table.append(record)
            global_position_id += 1

    coordinate_keys = {
        (round(float(record["x"]), 12), round(float(record["y"]), 12))
        for record in position_table
    }
    ordinary_ids_in_order = [
        int(record["ordinary_position_id"]) for record in ordinary_positions
    ]

    assert len(position_table) == 355, "The physical-position count must be 355."
    assert len(coordinate_keys) == 355, "Every physical position must be unique."
    assert len(ordinary_positions) == 350, "The ordinary-position count must be 350."
    assert len(tie_rod_positions) == 4, "The tie-rod count must be four."
    assert ordinary_ids_in_order == list(range(350)), (
        "Ordinary-position IDs must be stable and contiguous from 0 through 349."
    )

    default_dummy_local_indices = {0, 13, 26, 39, 52}
    default_dummy_ids = {
        int(record["ordinary_position_id"])
        for record in ordinary_positions
        if record["ring_index"] == 10
        and record["local_index"] in default_dummy_local_indices
    }
    selected_dummy_ids = _validate_dummy_position_ids(
        dummy_position_ids,
        default_dummy_ids,
    )
    fuel_position_ids = set(range(350)).difference(selected_dummy_ids)

    for record in ordinary_positions:
        position_id = int(record["ordinary_position_id"])
        record["position_type"] = (
            "dummy" if position_id in selected_dummy_ids else "fuel"
        )

    assert len(default_dummy_ids) == 5, "The reference layout must select five dummies."
    assert len(selected_dummy_ids) == 5, "The dummy-element count must be five."
    assert len(fuel_position_ids) == 345, "The fuel-element count must be 345."
    assert selected_dummy_ids.isdisjoint(fuel_position_ids), (
        "Dummy and fuel position IDs must not overlap."
    )

    # Unified finite-height model.
    axial_height = 23.0
    z_bottom_cm = -11.5
    z_top_cm = 11.5
    z_bottom = openmc.ZPlane(z0=z_bottom_cm, boundary_type="vacuum")
    z_top = openmc.ZPlane(z0=z_top_cm, boundary_type="vacuum")
    axial_region = +z_bottom & -z_top

    r_fuel_meat = 0.215
    r_fuel_clad = 0.275
    r_dummy = 0.275
    # Unified simplification necessitated by insufficient published tie-rod
    # dimensional data; this retains the former assumed radius.
    r_tie_rod = 0.215

    r_core = 11.5
    r_be_in = 11.55
    r_be_out = 21.75
    r_vessel_in = 30.0
    r_vessel_out = 31.0

    all_cells: list[openmc.Cell] = []
    fuel_meat_cells: dict[int, openmc.Cell] = {}
    fuel_clad_cells: dict[int, openmc.Cell] = {}
    dummy_cells: dict[int, openmc.Cell] = {}
    tie_rod_cells: dict[int, openmc.Cell] = {}
    ring_obstacle_surfaces: dict[int, list[openmc.ZCylinder]] = {
        ring_index: [] for ring_index in range(len(ring_distribution))
    }

    for record in ordinary_positions:
        ring_index = int(record["ring_index"])
        local_index = int(record["local_index"])
        position_id = int(record["ordinary_position_id"])
        x = float(record["x"])
        y = float(record["y"])

        if position_id in selected_dummy_ids:
            dummy_surface = openmc.ZCylinder(
                x0=x,
                y0=y,
                r=r_dummy,
                name=f"dummy_outer_pos_{position_id}",
            )
            dummy_cell = openmc.Cell(
                name=f"dummy_pos_{position_id}",
                fill=material_map["dummy"],
                region=-dummy_surface & axial_region,
            )
            dummy_cells[position_id] = dummy_cell
            ring_obstacle_surfaces[ring_index].append(dummy_surface)
            all_cells.append(dummy_cell)
        else:
            fuel_meat_surface = openmc.ZCylinder(
                x0=x,
                y0=y,
                r=r_fuel_meat,
                name=f"fuel_meat_outer_pos_{position_id}",
            )
            fuel_clad_surface = openmc.ZCylinder(
                x0=x,
                y0=y,
                r=r_fuel_clad,
                name=f"fuel_clad_outer_pos_{position_id}",
            )
            fuel_meat_cell = openmc.Cell(
                name=f"fuel_meat_pos_{position_id}",
                fill=material_map["fuel"],
                region=-fuel_meat_surface & axial_region,
            )
            fuel_clad_cell = openmc.Cell(
                name=f"fuel_clad_pos_{position_id}",
                fill=material_map["fuel_clad"],
                region=+fuel_meat_surface & -fuel_clad_surface & axial_region,
            )
            fuel_meat_cells[position_id] = fuel_meat_cell
            fuel_clad_cells[position_id] = fuel_clad_cell
            ring_obstacle_surfaces[ring_index].append(fuel_clad_surface)
            all_cells.extend([fuel_meat_cell, fuel_clad_cell])

        # These values make later metadata inspection independent of list order.
        record["cell_kind"] = record["position_type"]
        record["ring_radius"] = ring_index * pitch
        record["azimuth"] = (
            validated_offsets[ring_index]
            + 2.0 * math.pi * local_index / ring_distribution[ring_index]
        )

    for record in tie_rod_positions:
        ring_index = int(record["ring_index"])
        local_index = int(record["local_index"])
        global_id = int(record["global_position_id"])
        tie_rod_surface = openmc.ZCylinder(
            x0=float(record["x"]),
            y0=float(record["y"]),
            r=r_tie_rod,
            name=f"tie_rod_outer_global_{global_id}",
        )
        tie_rod_cell = openmc.Cell(
            name=f"tie_rod_ring_{ring_index}_local_{local_index}",
            fill=material_map["tie_rod"],
            region=-tie_rod_surface & axial_region,
        )
        tie_rod_cells[global_id] = tie_rod_cell
        ring_obstacle_surfaces[ring_index].append(tie_rod_surface)
        all_cells.append(tie_rod_cell)
        record["cell_kind"] = "tie_rod"
        record["ring_radius"] = ring_index * pitch
        record["azimuth"] = (
            validated_offsets[ring_index]
            + 2.0 * math.pi * local_index / ring_distribution[ring_index]
        )

    central_record = position_table[0]
    central_record["cell_kind"] = "control_rod"
    central_record["ring_radius"] = 0.0
    central_record["azimuth"] = validated_offsets[0]

    r_guide_inner = 0.45
    r_guide_outer = 0.60
    guide_inner_surface = openmc.ZCylinder(
        r=r_guide_inner,
        name="control_guide_inner",
    )
    guide_outer_surface = openmc.ZCylinder(
        r=r_guide_outer,
        name="control_guide_outer",
    )
    control_rod_cells: dict[str, openmc.Cell] = {}

    if control_rod_state == "withdrawn":
        central_water_cell = openmc.Cell(
            name="withdrawn_control_guide_water",
            fill=material_map["water"],
            region=-guide_inner_surface & axial_region,
        )
        guide_tube_cell = openmc.Cell(
            name="control_guide_tube",
            fill=material_map["guide_tube"],
            region=+guide_inner_surface & -guide_outer_surface & axial_region,
        )
        control_rod_cells.update(
            {
                "central_water": central_water_cell,
                "guide_tube": guide_tube_cell,
            }
        )
        all_cells.extend([central_water_cell, guide_tube_cell])
    else:
        r_control_inner = 0.095
        r_cadmium_outer = 0.195
        r_control_clad_inner = 0.20
        r_control_clad_outer = 0.25

        control_inner_surface = openmc.ZCylinder(
            r=r_control_inner,
            name="control_inner_outer",
        )
        cadmium_outer_surface = openmc.ZCylinder(
            r=r_cadmium_outer,
            name="cadmium_outer",
        )
        control_clad_inner_surface = openmc.ZCylinder(
            r=r_control_clad_inner,
            name="control_clad_inner",
        )
        control_clad_outer_surface = openmc.ZCylinder(
            r=r_control_clad_outer,
            name="control_clad_outer",
        )

        control_inner_cell = openmc.Cell(
            name="control_rod_inner",
            fill=material_map["control_rod_inner"],
            region=-control_inner_surface & axial_region,
        )
        cadmium_cell = openmc.Cell(
            name="cadmium_control_tube",
            fill=material_map["cadmium"],
            region=(
                +control_inner_surface
                & -cadmium_outer_surface
                & axial_region
            ),
        )
        # Unified simplification: the undocumented 0.195--0.20 cm clearance
        # between cadmium and SS-304 is represented by light water.
        cadmium_gap_water_cell = openmc.Cell(
            name="cadmium_to_clad_water_gap",
            fill=material_map["water"],
            region=(
                +cadmium_outer_surface
                & -control_clad_inner_surface
                & axial_region
            ),
        )
        control_clad_cell = openmc.Cell(
            name="control_rod_ss304_clad",
            fill=material_map["control_rod_clad"],
            region=(
                +control_clad_inner_surface
                & -control_clad_outer_surface
                & axial_region
            ),
        )
        control_outer_water_cell = openmc.Cell(
            name="control_rod_to_guide_water",
            fill=material_map["water"],
            region=(
                +control_clad_outer_surface
                & -guide_inner_surface
                & axial_region
            ),
        )
        guide_tube_cell = openmc.Cell(
            name="control_guide_tube",
            fill=material_map["guide_tube"],
            region=+guide_inner_surface & -guide_outer_surface & axial_region,
        )
        control_rod_cells.update(
            {
                "inner": control_inner_cell,
                "cadmium": cadmium_cell,
                "cadmium_gap_water": cadmium_gap_water_cell,
                "clad": control_clad_cell,
                "outer_water": control_outer_water_cell,
                "guide_tube": guide_tube_cell,
            }
        )
        all_cells.extend(control_rod_cells.values())

    core_boundary = openmc.ZCylinder(r=r_core, name="core_outer")
    ring_boundary_radii = [
        (ring_index + 0.5) * pitch
        for ring_index in range(len(ring_distribution) - 1)
    ]
    ring_boundaries = [
        openmc.ZCylinder(r=radius, name=f"core_water_ring_boundary_{index}")
        for index, radius in enumerate(ring_boundary_radii)
    ]
    core_water_cells: dict[int, openmc.Cell] = {}

    for ring_index in range(len(ring_distribution)):
        if ring_index == 0:
            # The central guide radius (0.60 cm) exceeds the exact ring-0/ring-1
            # midpoint (0.5475 cm). The whole central band is consequently
            # occupied by the guide assembly, so it has no separate bulk-water
            # cell.
            continue
        elif ring_index == len(ring_distribution) - 1:
            radial_band = +ring_boundaries[-1] & -core_boundary
        else:
            radial_band = (
                +ring_boundaries[ring_index - 1]
                & -ring_boundaries[ring_index]
            )

        water_region = radial_band & axial_region
        for obstacle_surface in ring_obstacle_surfaces[ring_index]:
            water_region &= +obstacle_surface

        if ring_index == 1 and r_guide_outer > ring_boundary_radii[0]:
            # The 0.60 cm guide crosses the exact ring-0/ring-1 midpoint
            # (0.5475 cm), so ring 1 must also exclude this one central surface
            # to keep the cell partition overlap-free.
            water_region &= +guide_outer_surface

        water_cell = openmc.Cell(
            name=f"core_water_ring_{ring_index}",
            fill=material_map["water"],
            region=water_region,
        )
        core_water_cells[ring_index] = water_cell
        all_cells.append(water_cell)

    be_inner_boundary = openmc.ZCylinder(r=r_be_in, name="beryllium_inner")
    be_outer_boundary = openmc.ZCylinder(r=r_be_out, name="beryllium_outer")
    vessel_inner_boundary = openmc.ZCylinder(r=r_vessel_in, name="vessel_inner")
    vessel_outer_boundary = openmc.ZCylinder(
        r=r_vessel_out,
        boundary_type="vacuum",
        name="vessel_outer",
    )

    core_to_beryllium_water = openmc.Cell(
        name="core_to_beryllium_water",
        fill=material_map["water"],
        region=+core_boundary & -be_inner_boundary & axial_region,
    )
    beryllium_reflector = openmc.Cell(
        name="beryllium_reflector",
        fill=material_map["beryllium"],
        region=+be_inner_boundary & -be_outer_boundary & axial_region,
    )
    reflector_to_vessel_water = openmc.Cell(
        name="reflector_to_vessel_water",
        fill=material_map["water"],
        region=+be_outer_boundary & -vessel_inner_boundary & axial_region,
    )
    vessel_wall = openmc.Cell(
        name="reactor_vessel",
        fill=material_map["vessel"],
        region=+vessel_inner_boundary & -vessel_outer_boundary & axial_region,
    )
    all_cells.extend(
        [
            core_to_beryllium_water,
            beryllium_reflector,
            reflector_to_vessel_water,
            vessel_wall,
        ]
    )

    counts = {
        "total_positions": len(position_table),
        "total_physical_positions": len(position_table),
        "ordinary_positions": len(ordinary_positions),
        "tie_rod_positions": len(tie_rod_positions),
        "tie_rods": len(tie_rod_positions),
        "fuel_positions": len(fuel_position_ids),
        "fuel_elements": len(fuel_position_ids),
        "dummy_positions": len(selected_dummy_ids),
        "dummy_elements": len(selected_dummy_ids),
        "central_positions": 1,
    }
    assert counts["total_physical_positions"] == 355
    assert counts["ordinary_positions"] == 350
    assert counts["tie_rods"] == 4
    assert counts["fuel_elements"] == 345
    assert counts["dummy_elements"] == 5
    assert counts["central_positions"] == 1
    assert len(fuel_meat_cells) == 345
    assert len(fuel_clad_cells) == 345
    assert len(dummy_cells) == 5

    root_universe = openmc.Universe(name="mnsr_sz_root", cells=all_cells)
    geometry = openmc.Geometry(root_universe)

    metadata = {
        "positions": position_table,
        "position_table": position_table,
        "ordinary_positions": ordinary_positions,
        "fuel_position_ids": fuel_position_ids,
        "dummy_position_ids": selected_dummy_ids,
        "tie_rod_positions": tie_rod_positions,
        "fuel_meat_cells": fuel_meat_cells,
        "fuel_clad_cells": fuel_clad_cells,
        "dummy_cells": dummy_cells,
        "tie_rod_cells": tie_rod_cells,
        "control_rod_cells": control_rod_cells,
        "core_water_cells": core_water_cells,
        "control_rod_state": control_rod_state,
        "counts": counts,
        "ring_distribution": ring_distribution,
        "ring_offsets": validated_offsets,
        "dimensions_cm": {
            "pitch": pitch,
            "axial_height": axial_height,
            "z_bottom": z_bottom_cm,
            "z_top": z_top_cm,
            "r_fuel_meat": r_fuel_meat,
            "r_fuel_clad": r_fuel_clad,
            "r_dummy": r_dummy,
            "r_tie_rod": r_tie_rod,
            "r_guide_inner": r_guide_inner,
            "r_guide_outer": r_guide_outer,
            "r_core": r_core,
            "r_be_in": r_be_in,
            "r_be_out": r_be_out,
            "r_vessel_in": r_vessel_in,
            "r_vessel_out": r_vessel_out,
        },
    }

    print("基于统一文献参考情景的三维简化模型构建完成")
    return geometry, metadata
