import math
from pathlib import Path


MU = 398600.4418
EARTH_RADIUS_KM = 6378.137
SECONDS = 7200

CONSTELLATIONS = [
    (160, 32, 2, 1150, 53.0),
    (160, 32, 2, 1110, 53.8),
    (40, 8, 2, 1130, 74.0),
    (40, 5, 2, 1275, 81.0),
    (48, 6, 2, 1325, 70.0),
    (552, 46, 2, 550, 45.0),
]


def main() -> None:
    out = Path("SatEdgeSim/settings/locationflie/edge_devices/mist Fixed Position.csv")
    sat_id = 1
    with out.open("w", encoding="utf-8", newline="") as handle:
        for total, planes, phase_factor, altitude, inclination_deg in CONSTELLATIONS:
            per_plane = total // planes
            radius = EARTH_RADIUS_KM + altitude
            angular_velocity = math.sqrt(MU / (radius ** 3))
            inclination = math.radians(inclination_deg)
            for plane in range(planes):
                raan = 2 * math.pi * plane / planes
                for slot in range(per_plane):
                    phase = 2 * math.pi * (slot / per_plane + phase_factor * plane / total)
                    handle.write(
                        f'"Time (EpSec)","mist{sat_id} - x (km)",'
                        f'"mist{sat_id} - y (km)","mist{sat_id} - z (km)"\r\n'
                    )
                    for second in range(SECONDS + 1):
                        u = phase + angular_velocity * second
                        x_orb = radius * math.cos(u)
                        y_orb = radius * math.sin(u)
                        x = x_orb * math.cos(raan) - y_orb * math.cos(inclination) * math.sin(raan)
                        y = x_orb * math.sin(raan) + y_orb * math.cos(inclination) * math.cos(raan)
                        z = y_orb * math.sin(inclination)
                        handle.write(f"{second:.3f},{x:.6f},{y:.6f},{z:.6f}\r\n")
                    handle.write("\r\n")
                    sat_id += 1
    print(f"Generated {sat_id - 1} mist satellites at {out}")


if __name__ == "__main__":
    main()
