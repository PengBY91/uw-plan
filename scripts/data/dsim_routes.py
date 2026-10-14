"""Write the D-sim route JSONs (checklist W5-11, runbook R1 D-sim row) into configs/sim/.

D-sim is the simulated supplement whose failure axes are meant to be mutually independent, so that per-axis and
scalar detection can be told apart (H-A2). The 14 pilot routes all fail through one mechanism (heading error times
lateral travel), so x, y and heading exceed together. Three route kinds:

- vstripe_heave_*: the vertical-stripe wall of stripe_wall_v_v1, but the vehicle heaves along the stripes instead of
  traversing across them. Stripes parallel to the motion should leave z (vertical) unconstrained, as horizontal stripes
  did for y; there is no horizontal travel, so a heading error cannot turn into x or y error.
- far_approach_*: a large textured wall 12-24 m ahead, approached and left along the optical axis. Far features give
  little parallax for forward translation (weak x) while still fixing heading; the travel is along x, so a heading
  error would show up in y, which is what has to stay small.
- normal_*: benign routes in well-textured scenes (three families), used for the false-alarm budget.

Every route keeps the W2-11 conventions: OpenWater/Dam, 200 tick/s, same camera/IMU/depth config and assumed noise
(the gyro is replaced afterwards by scripts/data/sim_imu_grade.py at grade g1), raycast sonar so props are visible.
usage: python scripts/data/dsim_routes.py [--only NAME ...]
"""
import argparse
import copy
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = json.loads((ROOT/'configs/sim/stripe_wall_v_v1.json').read_text())
INIT = [(0, 0, 0, 0, 0), (6, 0, 0, 0, 0), (8, 0.3, 0.3, 10, 0), (10, -0.3, -0.3, -10, 0), (12, 0, 0, 0, 0)]  # dyn-init wiggle


def sensors(range_max=10.0, range_bins=256):
    s = copy.deepcopy(BASE['sensors'])
    s['sonar_model'] = 'raycast'
    s['sonar'].update({'RangeMax': range_max, 'RangeBins': range_bins})
    return s


def route(name, purpose, world, waypoints, props, fog, son=None, extra=None):
    r = {'name': name, 'purpose': purpose, 'world': world, 'package': 'Ocean', 'ticks_per_sec': 200, 'viewport': [480, 640],
         'assumptions': {**BASE['assumptions'],
                         'imu_noise': BASE['assumptions']['imu_noise'] + '; D-sim replaces the gyro with grade g1 '
                                      '(EuRoC ADIS16448 Kalibr values) via sim_imu_grade.py as a pre-Allan placeholder'},
         'sensors': son or sensors(), 'waypoints': [{'t': round(t, 3), 'pose': [round(v, 4) for v in p]} for t, p in waypoints],
         'estimator_overrides': BASE['estimator_overrides'], 'props': props, 'fog': fog}
    r.update(extra or {})
    return r


def with_init(start, yaw, legs):
    """Init wiggle around start (x, y, z) facing yaw, then legs [(duration_s, (x, y, z)), ...] at constant yaw."""
    x0, y0, z0 = start
    wps = []
    for t, dx, dz, dyaw, _ in INIT:
        # the wiggle moves along the facing direction and in z, like the W2-11 routes
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        wps.append((t, [x0 + dx*c, y0 + dx*s, z0 + dz, 0, 0, yaw + dyaw]))
    t, pos = 12.0, (x0, y0, z0)
    for dur, nxt in legs:
        t += dur
        wps.append((t, [*nxt, 0, 0, yaw]))
        pos = nxt
    return wps


def vstripe_wall(x0=0.0):
    """stripe_wall_v_v1's 20 vertical 10 m x 1 m brick columns (wall plane y = 0, z -25..-15), shifted by x0."""
    return [p for p in BASE['props'] if p['scale'] == [10, 0.5, 1.0]] if x0 == 0 else \
        [{**p, 'location': [p['location'][0] + x0, *p['location'][1:]]} for p in BASE['props'] if p['scale'] == [10, 0.5, 1.0]]


def heave(name, standoff, z_lo, z_hi, speed, cycles, fog, x=0.0, purpose=''):
    zc = -20.0
    legs = [(abs(zc - z_lo)/speed, (x, -standoff, z_lo))]
    for _ in range(cycles):
        legs += [(3, (x, -standoff, z_lo)), ((z_hi - z_lo)/speed, (x, -standoff, z_hi)), (3, (x, -standoff, z_hi)),
                 ((z_hi - z_lo)/speed, (x, -standoff, z_lo))]
    legs += [(abs(zc - z_lo)/speed, (x, -standoff, zc)), (3, (x, -standoff, zc))]
    return route(name, 'D-sim z-only candidate: heave along the vertical stripes of the stripe_wall_v_v1 wall, no horizontal '
                 f'travel (standoff {standoff} m, z {z_lo}..{z_hi}, {speed} m/s, {cycles} cycles, fog {fog}). {purpose}',
                 'OpenWater', with_init((x, -standoff, zc), 90.0, legs), vstripe_wall(), [{'t': 0, 'density': fog, 'depth': 0.5}])


def big_wall(y_wall, width=36.0, top=-2.0, bottom=-38.0, tile=6.0):
    """Large wall facing -y at y = y_wall: tiles of brick/cobblestone (checkerboard) so corners and edges exist at range."""
    props, nx, nz = [], int(width/tile), int((top - bottom)/tile)
    for i in range(nx):
        for k in range(nz):
            props.append({'type': 'box', 'location': [-width/2 + (i + 0.5)*tile, y_wall, bottom + (k + 0.5)*tile],
                          'scale': [tile, 0.5, tile], 'material': 'brick' if (i + k) % 2 else 'cobblestone'})
    return props


def approach(name, y_wall, near, far, speed, cycles, fog, purpose=''):
    z = -20.0
    legs = []
    for _ in range(cycles):
        legs += [((far - near)/speed, (0.0, y_wall - near, z)), (3, (0.0, y_wall - near, z)),
                 ((far - near)/speed, (0.0, y_wall - far, z)), (3, (0.0, y_wall - far, z))]
    return route(name, f'D-sim x-only candidate: forward approach and retreat along the optical axis to a large textured wall '
                 f'{near}-{far} m ahead (fog {fog}, {speed} m/s, {cycles} cycles), no lateral travel. Sonar range raised to 25 m '
                 f'(640 bins, same 3.8 cm bin) so the wall stays in the sonar fan. {purpose}',
                 'OpenWater', with_init((0.0, y_wall - far, z), 90.0, legs), big_wall(y_wall),
                 [{'t': 0, 'density': fog, 'depth': 0.5}], son=sensors(25.0, 640))


def shuttle(name, standoff, x_start, x_end, speed, cycles, fog, purpose=''):
    """Horizontal shuttle across the vertical stripes of the stripe_wall_v_v1 wall, starting where that route starts.

    On stripe_wall_v_v1 the z error grew roughly with the square of the distance from the start (0.15 m at about 11 m,
    0.86 m at 26 m) while x, which comes from heading error times lateral travel, only passed 0.15 m beyond about 22 m.
    A span of about 16 m should therefore exceed z alone.
    """
    z = -20.0
    legs = []
    for _ in range(cycles):
        legs += [(abs(x_end - x_start)/speed, (x_end, -standoff, z)), (3, (x_end, -standoff, z)),
                 (abs(x_end - x_start)/speed, (x_start, -standoff, z)), (3, (x_start, -standoff, z))]
    return route(name, f'D-sim z-only candidate (second design): horizontal shuttle across the vertical stripes of the '
                 f'stripe_wall_v_v1 wall from x {x_start} to {x_end} m (standoff {standoff} m, {speed} m/s, {cycles} cycles, '
                 f'fog {fog}). The first design (heave along the stripes, probe_vstripe_heave_v1) did not degrade z. {purpose}',
                 'OpenWater', with_init((x_start, -standoff, z), 90.0, legs), BASE['props'],
                 [{'t': 0, 'density': fog, 'depth': 0.5}])


ROUTES = {}


def add(r):
    ROUTES[r['name']] = r


# probes (short) -------------------------------------------------------------------------------------------------
add(heave('probe_vstripe_heave_v1', 2.5, -22.0, -18.0, 0.2, 1, 2.0, purpose='Probe.'))
add(approach('probe_far_approach_v1', 0.0, 12.0, 22.0, 0.25, 1, 1.0, purpose='Probe.'))
add(approach('probe_far_approach_fog0_v1', 0.0, 12.0, 22.0, 0.25, 1, 0.3, purpose='Probe, light fog.'))


def legs_from(points, speed, dwell=3.0):
    """Constant-speed legs through points [(x, y, z), ...] with a short dwell at each."""
    legs, prev = [], points[0]
    for q in points[1:]:
        d = math.dist(prev, q)
        legs += [(max(d/speed, 1.0), q), (dwell, q)]
        prev = q
    return legs


def normal_dam(name, levels, span, speed, purpose=''):
    """Dam buttress traverse (dam_wall_v1 geometry, no props), repeated at several depths and two standoffs."""
    import numpy as np
    c = np.array([25.649, -26.249, -55.0])
    along = np.array([26.041 - 25.257, -21.766 + 30.732, 0.0])
    along /= np.linalg.norm(along)
    back = np.array([23.159 - 25.649, -26.031 + 26.249, 0.0])
    back /= np.linalg.norm(back)
    pts = [tuple(c)]
    for k, z in enumerate(levels):
        off = back*(2.5 if k % 2 else 0.0)
        pts += [tuple(c + along*span/2 + off + [0, 0, z - c[2]]), tuple(c - along*span/2 + off + [0, 0, z - c[2]])]
    pts.append(tuple(c))
    return route(name, f'D-sim normal family dam: Dam buttress traverse, span {span} m at depths {levels}, {speed} m/s, '
                 f'alternating standoff (+0 / +2.5 m). {purpose}', 'Dam', with_init(tuple(c), -5.0, legs_from(pts, speed)), [], [])


def checker_wall(tile=2.0, width=24.0, top=-14.0, bottom=-26.0):
    return big_wall(0.0, width, top, bottom, tile)


def normal_wall(name, props, fog, standoffs, levels, xspan, speed, purpose=''):
    """Lawnmower in front of a wall facing +y: traverse in x at each depth level, alternating standoff."""
    pts = [(0.0, -standoffs[0], -20.0)]
    for k, z in enumerate(levels):
        y = -standoffs[k % len(standoffs)]
        xs = (-xspan/2, xspan/2) if k % 2 == 0 else (xspan/2, -xspan/2)
        pts += [(xs[0], y, z), (xs[1], y, z)]
    pts.append((0.0, -standoffs[0], -20.0))
    return route(name, f'D-sim normal family: lawnmower in front of a wall, x span {xspan} m, depths {levels}, standoffs '
                 f'{standoffs}, {speed} m/s, fog {fog}. {purpose}', 'OpenWater',
                 with_init(pts[0], 90.0, legs_from(pts, speed)), props, [{'t': 0, 'density': fog, 'depth': 0.5}])


PILASTER = json.loads((ROOT/'configs/sim/pilaster_deep_v1.json').read_text())['props']
add(normal_dam('normal_dam_v1', [-55.0, -54.0, -56.0, -55.0], 9.0, 0.25))
add(normal_dam('normal_dam_b_v1', [-55.0, -56.0, -54.0, -55.5], 7.0, 0.2,
               purpose='Shorter span, slower.'))
add(normal_wall('normal_checker_v1', checker_wall(), 2.0, [3.0, 4.0], [-20, -19, -21, -18.5], 14.0, 0.3,
                purpose='Checkerboard of 2 m brick/cobblestone tiles: texture in both directions.'))
add(normal_wall('normal_checker_fog1_v1', checker_wall(), 1.0, [3.5, 2.5], [-20, -21, -19, -21.5], 12.0, 0.25,
                purpose='Second realisation, light fog.'))
add(normal_wall('normal_pilaster_v1', PILASTER, 2.0, [2.5, 3.0], [-20, -19.5, -20.5, -20], 16.0, 0.35,
                purpose='pilaster_deep_v1 wall (0.3 m pilasters every 2 m on horizontal-stripe brick).'))
add(normal_wall('normal_pilaster_b_v1', PILASTER, 2.0, [2.5], [-20, -20.5, -19.5, -20], 14.0, 0.3,
                purpose='Second realisation, single standoff.'))

add(shuttle('vstripe_shuttle_v1', 2.5, -13.0, 3.0, 0.4, 2, 2.0))
add(shuttle('vstripe_shuttle_b_v1', 3.0, 13.0, -3.0, 0.35, 2, 2.0, purpose='Mirrored start, larger standoff.'))
add(shuttle('vstripe_shuttle_fog4_v1', 2.5, -13.0, 4.0, 0.4, 2, 4.0, purpose='Denser fog.'))



def yaw_excitation(name, standoff, amp_deg, half_period_s, cycles, fog, purpose=''):
    """Hold in front of the checkerboard and oscillate yaw (+-amp_deg, half_period_s per swing): the rotational
    excitation that the M3.4 time-offset check needs, i.e. the pre-mission offset-check manoeuvre. No translation."""
    pos = (0.0, -standoff, -20.0)
    wps = with_init(pos, 90.0, [(5.0, pos)])
    t = wps[-1][0]
    for k in range(2*cycles):
        t += half_period_s
        wps.append((t, [*pos, 0, 0, 90.0 + (amp_deg if k % 2 == 0 else -amp_deg)]))
    wps += [(t + half_period_s/2, [*pos, 0, 0, 90.0]), (t + half_period_s/2 + 5.0, [*pos, 0, 0, 90.0])]
    return route(name, f'M3.4 offset-check manoeuvre: hold {standoff} m from the checkerboard, yaw +-{amp_deg} deg, '
                 f'{half_period_s} s per swing, {cycles} cycles, fog {fog}. {purpose}', 'OpenWater', wps,
                 checker_wall(), [{'t': 0, 'density': fog, 'depth': 0.5}])


add(yaw_excitation('offset_check_checker_v1', 3.5, 12.0, 2.0, 8, 2.0,
                   purpose='Parent for software-injected IMU offsets (inject_fault.py) in the input-integrity check.'))


def retreat_approach(name, near_init, far, near, speed, fog, tile=3.0, purpose=''):
    """Second x-only design: initialise near the wall (good parallax), retreat along the optical axis to far, then
    approach to near and retreat again. probe_far_approach_fog0_v1 initialised at 22 m and diverged from the start."""
    z, yw = -20.0, 0.0
    pts = [(0.0, yw - near_init, z), (0.0, yw - far, z), (0.0, yw - near, z), (0.0, yw - far, z)]
    return route(name, f'D-sim x-only candidate (second design): initialise {near_init} m from a large {tile} m-tile textured '
                 f'wall, then retreat to {far} m, approach to {near} m and retreat again along the optical axis ({speed} m/s, '
                 f'fog {fog}), no lateral travel. Sonar range 25 m (640 bins). {purpose}', 'OpenWater',
                 with_init(pts[0], 90.0, legs_from(pts, speed)), big_wall(yw, 36.0, -2.0, -38.0, tile),
                 [{'t': 0, 'density': fog, 'depth': 0.5}], son=sensors(25.0, 640))


add(retreat_approach('probe_retreat_approach_v1', 5.0, 20.0, 10.0, 0.25, 0.3, purpose='Probe.'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--only', nargs='*')
    args = p.parse_args()
    for name, r in ROUTES.items():
        if args.only and name not in args.only:
            continue
        path = ROOT/'configs/sim'/f'{name}.json'
        path.write_text(json.dumps(r, indent=2) + '\n')
        print(path.relative_to(ROOT), r['waypoints'][-1]['t'], 's', len(r['props']), 'props')


if __name__ == '__main__':
    main()
