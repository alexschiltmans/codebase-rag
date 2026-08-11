"""Great-circle distance and bounding box helpers for coordinates."""

import math

EARTH_RADIUS_KM = 6371.0088
MAX_LATITUDE = 90.0
MAX_LONGITUDE = 180.0


def validate_coordinate(latitude, longitude):
    """Reject coordinates outside the valid latitude and longitude ranges."""
    if not -MAX_LATITUDE <= latitude <= MAX_LATITUDE:
        raise ValueError(f"latitude {latitude} is outside [-90, 90]")
    if not -MAX_LONGITUDE <= longitude <= MAX_LONGITUDE:
        raise ValueError(f"longitude {longitude} is outside [-180, 180]")


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres between two coordinates.

    Uses the haversine form rather than the spherical law of cosines because the
    latter loses precision catastrophically for points a few metres apart, where
    the cosine of the central angle rounds to exactly 1.
    """
    validate_coordinate(lat1, lon1)
    validate_coordinate(lat2, lon2)

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def bounding_box(latitude, longitude, radius_km):
    """Return a latitude and longitude box enclosing a radius around a point.

    A box is a coarse prefilter: it always contains the circle, so a query can
    use it to narrow candidates cheaply and then apply the exact haversine
    distance to whatever survives.
    """
    validate_coordinate(latitude, longitude)
    lat_delta = math.degrees(radius_km / EARTH_RADIUS_KM)
    lon_delta = math.degrees(radius_km / (EARTH_RADIUS_KM * math.cos(math.radians(latitude))))
    return (
        latitude - lat_delta,
        longitude - lon_delta,
        latitude + lat_delta,
        longitude + lon_delta,
    )


def within_radius(origin, candidates, radius_km):
    """Filter candidate coordinates to those inside a radius of the origin."""
    lat, lon = origin
    return [c for c in candidates if haversine_km(lat, lon, c[0], c[1]) <= radius_km]
