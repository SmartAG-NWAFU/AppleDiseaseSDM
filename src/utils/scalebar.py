import numpy as np
from typing import Callable, Tuple, Optional, Dict
import matplotlib.axes
import cartopy.crs as ccrs
import cartopy.geodesic as cgeo


def _axes_to_lonlat(ax: matplotlib.axes.Axes, coords: Tuple[float, float]) -> Tuple[float, float]:
    """Convert axes (relative) coordinates to (lon, lat) in PlateCarree."""
    display = ax.transAxes.transform(coords)
    data = ax.transData.inverted().transform(display)
    lonlat = ccrs.PlateCarree().transform_point(*data, ax.projection)
    return lonlat


def _upper_bound(
    start: np.ndarray,
    direction: np.ndarray,
    distance: float,
    dist_func: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    """Find a point in a given direction past the given distance."""
    if distance <= 0:
        raise ValueError(f"Minimum distance is not positive: {distance}")
    if np.linalg.norm(direction) == 0:
        raise ValueError("Direction vector must not be zero.")

    length = 0.1
    end = start + length * direction

    while dist_func(start, end) < distance:
        length *= 2
        end = start + length * direction

    return end


def _distance_along_line(
    start: np.ndarray,
    end: np.ndarray,
    distance: float,
    dist_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float
) -> np.ndarray:
    """Find the point at a given distance along the line from start to end."""
    if tol <= 0:
        raise ValueError(f"Tolerance must be positive, got {tol}")

    initial_distance = dist_func(start, end)
    if initial_distance < distance:
        raise ValueError(f"End is closer to start ({initial_distance}) than distance ({distance})")

    left = start
    right = end

    while not np.isclose(dist_func(start, right), distance, rtol=tol):
        midpoint = (left + right) / 2
        if dist_func(start, midpoint) < distance:
            left = midpoint
        else:
            right = midpoint

    return right


def _point_along_line(
    ax: matplotlib.axes.Axes,
    start: np.ndarray,
    distance: float,
    angle: float = 0,
    tol: float = 0.01
) -> np.ndarray:
    """Compute point at a specific geodesic distance and angle from a start point."""
    direction = np.array([np.cos(angle), np.sin(angle)])
    geodesic = cgeo.Geodesic()

    def dist_func(a_axes: np.ndarray, b_axes: np.ndarray) -> float:
        a_phys = _axes_to_lonlat(ax, a_axes)
        b_phys = _axes_to_lonlat(ax, b_axes)
        return geodesic.inverse(a_phys, b_phys)[0][0]  # Fixed

    end = _upper_bound(start, direction, distance, dist_func)
    return _distance_along_line(start, end, distance, dist_func, tol)


def scale_bar(
    ax: matplotlib.axes.Axes,
    location: Tuple[float, float],
    length: float,
    metres_per_unit: float = 1000,
    unit_name: str = 'km',
    tol: float = 0.01,
    angle: float = 0,
    color: str = 'black',
    linewidth: float = 3,
    text_offset: float = 0.005,
    ha: str = 'center',
    va: str = 'bottom',
    plot_kwargs: Optional[Dict] = None,
    text_kwargs: Optional[Dict] = None,
    **kwargs
) -> None:
    """Draw a geodesic scale bar on CartoPy Axes."""
    plot_kwargs = {'linewidth': linewidth, 'color': color, **(plot_kwargs or {}), **kwargs}
    text_kwargs = {
        'ha': ha,
        'va': va,
        'rotation': angle,
        'color': color,
        **(text_kwargs or {}),
        **kwargs
    }

    location = np.asarray(location)
    length_metres = length * metres_per_unit
    angle_rad = np.radians(angle)

    end = _point_along_line(ax, location, length_metres, angle=angle_rad, tol=tol)

    ax.plot(*zip(location, end), transform=ax.transAxes, **plot_kwargs)

    midpoint = (location + end) / 2
    offset = text_offset * np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    text_location = midpoint + offset

    ax.text(*text_location, f"{length} {unit_name}", transform=ax.transAxes,
            rotation_mode='anchor', **text_kwargs)