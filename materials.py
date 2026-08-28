"""Material definitions for the unified MNSR-SZ LEU calculation scenario.

This module only constructs OpenMC materials.  Geometry, settings, tallies, XML
export, and transport execution are intentionally left to the caller.
"""

from __future__ import annotations

import math
import os
from numbers import Real
from pathlib import Path
from typing import Mapping, cast

import openmc
from openmc.data import atomic_mass

__all__ = ["build_materials"]


def _resolve_cross_sections_path(
    cross_sections_path: str | Path | None,
) -> Path:
    """Resolve and validate the OpenMC cross-section XML path."""
    if cross_sections_path is None:
        configured_path = os.environ.get("OPENMC_CROSS_SECTIONS")
        if not configured_path:
            raise FileNotFoundError(
                "No OpenMC cross-section library was specified. Pass "
                "cross_sections_path to build_materials() or set the "
                "OPENMC_CROSS_SECTIONS environment variable."
            )
        path_value: str | Path = configured_path
        path_source = "OPENMC_CROSS_SECTIONS"
    else:
        path_value = cross_sections_path
        path_source = "cross_sections_path"

    try:
        resolved_path = Path(path_value).expanduser().resolve()
    except TypeError as exc:
        raise TypeError(
            "cross_sections_path must be a string, pathlib.Path, or None."
        ) from exc

    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"OpenMC cross-section XML from {path_source} does not exist or "
            f"is not a file: {resolved_path}"
        )

    return resolved_path


def _validate_temperature(temperature_k: float) -> float:
    """Return a finite, positive material temperature in kelvin."""
    if isinstance(temperature_k, bool) or not isinstance(temperature_k, Real):
        raise TypeError("temperature_k must be a real number in kelvin.")

    validated_temperature = float(temperature_k)
    if not math.isfinite(validated_temperature) or validated_temperature <= 0.0:
        raise ValueError("temperature_k must be finite and greater than zero.")

    return validated_temperature


def _add_normalized_weight_fractions(
    material: openmc.Material,
    composition: Mapping[str, float],
) -> None:
    """Normalize positive elemental mass parts and add them to *material*."""
    if not composition:
        raise ValueError("A material composition cannot be empty.")

    validated_composition: dict[str, float] = {}
    for element, mass_part in composition.items():
        if not isinstance(element, str) or not element:
            raise TypeError("Each element symbol must be a non-empty string.")
        if isinstance(mass_part, bool) or not isinstance(mass_part, Real):
            raise TypeError(f"Mass part for {element} must be a real number.")

        value = float(mass_part)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Mass part for {element} must be finite and positive.")
        validated_composition[element] = value

    total = math.fsum(validated_composition.values())
    for element, mass_part in validated_composition.items():
        material.add_element(element, mass_part / total, percent_type="wo")


def build_materials(
    cross_sections_path: str | Path | None = None,
    temperature_k: float = 293.15,
) -> tuple[openmc.Materials, dict[str, openmc.Material]]:
    """Build the complete material library for the unified MNSR-SZ LEU model.

    Parameters
    ----------
    cross_sections_path
        Path to an OpenMC ``cross_sections.xml`` file. If omitted, the value
        of ``OPENMC_CROSS_SECTIONS`` is used. The resolved path must identify
        an existing file.
    temperature_k
        Temperature assigned to every material, in kelvin. It must be finite
        and greater than zero.

    Returns
    -------
    materials
        The eight unique materials required by the unified calculation
        scenario.
    material_map
        Role-based aliases used by the geometry-building code. Some roles
        intentionally refer to the same material object.

    Notes
    -----
    This function does not export ``materials.xml``. Call
    ``materials.export_to_xml()`` explicitly when an input file is needed.
    """
    resolved_cross_sections = _resolve_cross_sections_path(cross_sections_path)
    # OpenMC annotates Material.temperature as numbers.Real. Pylance does not
    # infer that a built-in float satisfies this runtime numeric ABC, so retain
    # the validated float value while making that relationship explicit.
    temperature = cast(Real, _validate_temperature(temperature_k))

    # Configure OpenMC before add_element() creates any natural-element mixture.
    # The environment is deliberately left unchanged.
    openmc.config["cross_sections"] = str(resolved_cross_sections)

    leu_fuel = openmc.Material(name="LEU UO2 12.5wt%")
    # IAEA-TECDOC-1844 uranium-isotope mass fractions. These values describe
    # the uranium constituent only, rather than the complete UO2 material.
    uranium_mass_fractions = {
        "U234": 0.0020,
        "U235": 0.1250,
        "U236": 0.0025,
        "U238": 0.8705,
    }
    uranium_mole_amounts = {
        nuclide: mass_fraction / atomic_mass(nuclide)
        for nuclide, mass_fraction in uranium_mass_fractions.items()
    }
    total_uranium_moles = math.fsum(uranium_mole_amounts.values())
    uranium_atom_fractions = {
        nuclide: mole_amount / total_uranium_moles
        for nuclide, mole_amount in uranium_mole_amounts.items()
    }

    assert math.isclose(
        math.fsum(uranium_atom_fractions.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ), "The uranium isotope atom fractions must sum to one."

    reconstructed_uranium_mass = math.fsum(
        atom_fraction * atomic_mass(nuclide)
        for nuclide, atom_fraction in uranium_atom_fractions.items()
    )
    reconstructed_u235_enrichment = (
        uranium_atom_fractions["U235"]
        * atomic_mass("U235")
        / reconstructed_uranium_mass
    )
    assert math.isclose(
        reconstructed_u235_enrichment,
        0.125,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ), "The reconstructed U-235 uranium-internal mass enrichment must be 12.5%."

    for nuclide in ("U234", "U235", "U236", "U238"):
        leu_fuel.add_nuclide(nuclide, uranium_atom_fractions[nuclide])
    # The normalized uranium atom fractions constitute one uranium atom part;
    # adding two oxygen atom parts preserves the required U:O ratio of 1:2.
    leu_fuel.add_element("O", 2.0)
    leu_fuel.set_density("g/cm3", 10.6)
    leu_fuel.temperature = temperature

    zircaloy4 = openmc.Material(name="Zircaloy-4")
    for element, weight_percent in {
        "Zr": 97.93,
        "Sn": 1.70,
        "Fe": 0.24,
        "Cr": 0.13,
    }.items():
        zircaloy4.add_element(element, weight_percent, percent_type="wo")
    zircaloy4.set_density("g/cm3", 6.5)
    zircaloy4.temperature = temperature

    al_303_1 = openmc.Material(name="Al-303-1")
    _add_normalized_weight_fractions(
        al_303_1,
        {
            "Al": 97.75,
            "Fe": 1.20,
            "Si": 1.00,
            "Cu": 0.005,
            "Zn": 0.003,
            "Mn": 0.003,
            "Mg": 0.005,
            "Ti": 0.005,
            "B": 0.0001,
        },
    )
    al_303_1.set_density("g/cm3", 2.7)
    al_303_1.temperature = temperature

    al_lt21 = openmc.Material(name="LT-21")
    _add_normalized_weight_fractions(
        al_lt21,
        {
            "Al": 98.1648,
            "Si": 0.9,
            "Mg": 0.675,
            "Fe": 0.2,
            "Ni": 0.03,
            "Cu": 0.01,
            "Mn": 0.01,
            "Ti": 0.01,
            "Cd": 0.0001,
            "B": 0.0001,
        },
    )
    al_lt21.set_density("g/cm3", 2.7)
    al_lt21.temperature = temperature

    ss304 = openmc.Material(name="SS-304")
    for element, weight_percent in {
        "Fe": 70.845,
        "Cr": 18.0,
        "Ni": 8.0,
        "Mn": 2.0,
        "Si": 1.0,
        "C": 0.08,
        "P": 0.045,
        "S": 0.03,
    }.items():
        ss304.add_element(element, weight_percent, percent_type="wo")
    ss304.set_density("g/cm3", 7.8)
    ss304.temperature = temperature

    water = openmc.Material(name="Light Water")
    water.add_nuclide("H1", 2.0)
    water.add_element("O", 1.0)
    water.set_density("g/cm3", 0.99825)
    water.temperature = temperature
    water.add_s_alpha_beta("c_H_in_H2O")

    beryllium = openmc.Material(name="Beryllium Reflector")
    # Unified-model simplification: natural metallic Be with trace impurities
    # neglected.
    beryllium.add_element("Be", 1.0)
    beryllium.set_density("g/cm3", 1.85)
    beryllium.temperature = temperature
    beryllium.add_s_alpha_beta("c_Be_in_Be")

    cadmium = openmc.Material(name="Natural Cadmium")
    cadmium.add_element("Cd", 1.0)
    cadmium.set_density("g/cm3", 8.65)
    cadmium.temperature = temperature

    materials = openmc.Materials(
        [
            leu_fuel,
            zircaloy4,
            al_303_1,
            al_lt21,
            ss304,
            water,
            beryllium,
            cadmium,
        ]
    )
    material_map = {
        "fuel": leu_fuel,
        "fuel_clad": zircaloy4,
        "dummy": al_303_1,
        "tie_rod": al_303_1,
        "control_rod_clad": ss304,
        "control_rod_inner": al_303_1,
        "guide_tube": al_lt21,
        "vessel": al_lt21,
        "water": water,
        "beryllium": beryllium,
        "cadmium": cadmium,
    }

    return materials, material_map
