"""
Synthetic effector–target movies with a known ground truth.

Targets are disks that jiggle in place and are only visible in channel A.
Effectors are smaller disks that roam, approach a target, stay docked to it
for a scheduled number of frames and leave again; they are only visible in
channel B. A third channel shows both populations at once, targets at half
the brightness of effectors. Every channel is a flat background with
Gaussian noise, so a single threshold segments each population.

Every contact is scripted, so the ground truth is exact: the label images,
the trajectories, the per-frame mask contacts and the contact episodes are
returned with the movie. The defaults build a 160 x 160 px, 30-frame movie
in well under a second, light enough for unit tests.

Examples
--------
>>> from celldetective.utils.synthetic_interactions import (
...     simulate_interactions, write_experiment)
>>> sim = simulate_interactions(seed=0)
>>> sim.movie.shape  # (T, C, Y, X)
(30, 3, 160, 160)
>>> pos = write_experiment("SyntheticExp", sim)  # doctest: +SKIP

From a shell::

    python -m celldetective.utils.synthetic_interactions SyntheticExp --frames 60
"""

import argparse
import os
from configparser import ConfigParser
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TARGET_CHANNEL = "live_nuclei_channel"
EFFECTOR_CHANNEL = "effector_fluo_channel"
COMBINED_CHANNEL = "brightfield_channel"
DEATH_CHANNEL = "dead_nuclei_channel"

# Docked effectors sink this many pixels into their target, so that the two
# masks overlap; free effectors keep this margin away from every target.
_DOCK_OVERLAP = 2.0
_FREE_MARGIN = 3.0


@dataclass
class SyntheticInteractions:
    """A simulated movie and its ground truth.

    Attributes
    ----------
    movie : ndarray
        ``(T, C, Y, X)`` uint16 stack. Channel 0 shows the targets, channel 1
        the effectors, the combined channel both, and, when deaths are
        simulated, the last channel the dead targets.
    channel_names : list of str
        Config names of the channels, in stack order.
    labels_targets, labels_effectors : ndarray
        ``(T, Y, X)`` label images; a cell's label is its ``TRACK_ID`` in
        every frame.
    targets, effectors : DataFrame
        Trajectories with ``TRACK_ID``, ``FRAME``, ``POSITION_X``,
        ``POSITION_Y``, ``class_id`` and ``radius``. Effectors also carry
        ``state`` (``free``, ``approach``, ``docked``) and ``docked_to``; with
        deaths, targets carry ``status`` (1 once dead) and ``t_death``.
    contacts : DataFrame
        One row per frame and pair whose masks overlap or touch:
        ``FRAME``, ``TARGET_ID``, ``EFFECTOR_ID``, ``distance`` (between
        centres) and ``overlap`` (shared pixels).
    episodes : DataFrame
        One row per scheduled docking: ``TARGET_ID``, ``EFFECTOR_ID``,
        ``t_start``, ``t_end`` and ``duration`` in frames.
    """

    movie: np.ndarray
    channel_names: List[str]
    labels_targets: np.ndarray
    labels_effectors: np.ndarray
    targets: pd.DataFrame
    effectors: pd.DataFrame
    contacts: pd.DataFrame
    episodes: pd.DataFrame
    params: dict = field(default_factory=dict)


@dataclass
class _Effector:
    pos: np.ndarray
    heading: float
    state: str = "free"
    timer: int = 0
    target: Optional[int] = None
    dock_angle: float = 0.0
    t_start: int = 0
    n_contacts: int = 0
    last_target: Optional[int] = None


def _place_targets(
    rng: np.random.Generator, n: int, shape: Tuple[int, int], radius: float, spacing: float
) -> np.ndarray:
    """Place target centres on a jittered grid, at least ``spacing`` apart and
    ``spacing / 2`` from the borders."""
    cols = int(np.ceil(np.sqrt(n * shape[1] / shape[0])))
    rows = int(np.ceil(n / cols))
    cell = np.array([shape[1] / cols, shape[0] / rows])
    if np.any(cell < spacing):
        raise ValueError(
            f"Cannot fit {n} targets of radius {radius:g} px in a {shape[1]} x {shape[0]} field; "
            "use a larger image or fewer targets."
        )
    jitter = (cell - spacing) / 2
    slots = rng.permutation(rows * cols)[:n]
    centres = [
        (np.array([s % cols, s // cols]) + 0.5) * cell + rng.uniform(-jitter, jitter) for s in slots
    ]
    return np.array(centres)


def _draw_disks(
    shape: Tuple[int, int], centres: np.ndarray, radii: Sequence[float], labels: Sequence[int]
) -> np.ndarray:
    """Rasterize disks into a label image; later disks are drawn on top."""
    img = np.zeros(shape, dtype=np.uint16)
    for (x, y), r, lbl in zip(centres, radii, labels):
        y0, y1 = max(0, int(y - r - 1)), min(shape[0], int(y + r + 2))
        x0, x1 = max(0, int(x - r - 1)), min(shape[1], int(x + r + 2))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        img[y0:y1, x0:x1][(xx - x) ** 2 + (yy - y) ** 2 <= r**2] = lbl
    return img


def _render(
    rng: np.random.Generator, mask: np.ndarray, foreground: float, background: float, noise: float
) -> np.ndarray:
    img = np.where(mask, foreground, background) + rng.normal(0, noise, mask.shape)
    return np.clip(img, 0, 65535).astype(np.uint16)


def _mask_contacts(
    lbl_t: np.ndarray, lbl_e: np.ndarray, frame: int, t_xy: np.ndarray, e_xy: np.ndarray
) -> List[dict]:
    """Pairs whose masks overlap or touch (8-connectivity)."""
    grown = lbl_e.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted = np.roll(np.roll(lbl_e, dy, axis=0), dx, axis=1)
            grown = np.where(grown == 0, shifted, grown)
    rows = []
    hit = (grown > 0) & (lbl_t > 0)
    for t_id, e_id in set(zip(lbl_t[hit].tolist(), grown[hit].tolist())):
        rows.append(
            {
                "FRAME": frame,
                "TARGET_ID": t_id,
                "EFFECTOR_ID": e_id,
                "distance": float(np.hypot(*(t_xy[t_id - 1] - e_xy[e_id - 1]))),
                "overlap": int(np.sum((lbl_t == t_id) & (lbl_e == e_id))),
            }
        )
    return rows


def simulate_interactions(
    shape: Tuple[int, int] = (160, 160),
    n_frames: int = 30,
    n_targets: int = 4,
    n_effectors: int = 4,
    target_radius: float = 12.0,
    effector_radius: float = 6.0,
    jiggle: float = 0.7,
    effector_speed: float = 4.0,
    contact_durations: Sequence[int] = (3, 12),
    free_frames: Tuple[int, int] = (2, 6),
    death_after: Optional[int] = None,
    death_probability: float = 0.0,
    background_death: float = 0.0,
    combined_channel: bool = True,
    background: float = 100.0,
    foreground: float = 1000.0,
    noise: float = 15.0,
    seed: int = 0,
) -> SyntheticInteractions:
    """Simulate jiggling targets and roaming effectors that dock to them.

    Each effector cycles through three states: it wanders for a few frames,
    heads to a target until the two touch, then stays docked for the next
    duration in ``contact_durations`` (cycled per effector, so short and long
    contacts alternate) before it leaves. Docked effectors follow their
    target's jiggle. Free effectors are kept clear of every target and of
    each other, so masks only touch during a docking.

    Parameters
    ----------
    shape : tuple of int
        Image size ``(Y, X)`` in pixels.
    n_frames : int
        Movie length.
    n_targets, n_effectors : int
        Number of cells of each population.
    target_radius, effector_radius : float
        Disk radii in pixels.
    jiggle : float
        Standard deviation, in pixels, of the targets' displacement around
        their rest position.
    effector_speed : float
        Effector step length in pixels per frame.
    contact_durations : sequence of int
        Docking durations in frames, cycled per effector.
    free_frames : tuple of int
        Range of frames an effector wanders between two contacts.
    death_after : int, optional
        When set, a target dies once it has been in contact for this many
        frames in total; a third channel then shows dead targets.
    death_probability : float
        Chance that a target dies after each frame it is in contact with an
        effector (dead from the next frame on): a death caused by contact, at
        a random time.
    background_death : float
        Chance that a target dies at each frame, in contact or not: deaths
        independent of the effectors.
    combined_channel : bool
        Add a channel showing both populations: effectors at ``foreground``
        and targets halfway between ``background`` and ``foreground``, so a
        threshold below the targets' level picks up every cell and one above
        it picks up only the effectors. Effectors are drawn over targets.
    background, foreground, noise : float
        Intensity of the background, of the cells, and the standard deviation
        of the Gaussian noise.
    seed : int
        Random seed; the same seed gives the same movie.

    Returns
    -------
    SyntheticInteractions
    """
    rng = np.random.default_rng(seed)
    deaths = death_after is not None or death_probability > 0 or background_death > 0
    shape = (int(shape[0]), int(shape[1]))
    Rt, Re = float(target_radius), float(effector_radius)
    contact_d = Rt + Re - _DOCK_OVERLAP
    free_d = Rt + Re + _FREE_MARGIN
    lo = np.array([Re + 1, Re + 1])
    hi = np.array([shape[1] - Re - 2, shape[0] - Re - 2])

    rest = _place_targets(rng, n_targets, shape, Rt, spacing=2 * Rt + 4 * Re + 8)
    offset = np.zeros_like(rest)
    effectors: List[_Effector] = []
    for _ in range(n_effectors):
        for _ in range(5000):
            p = rng.uniform(lo, hi)
            if np.all(np.hypot(*(rest - p).T) > free_d + jiggle * 3) and all(
                np.hypot(*(e.pos - p)) > 2 * Re + 2 for e in effectors
            ):
                break
        else:
            raise ValueError("Could not place the effectors; use a larger image.")
        effectors.append(
            _Effector(pos=p, heading=rng.uniform(0, 2 * np.pi), timer=int(rng.integers(*free_frames, endpoint=True)))
        )

    contact_frames = np.zeros(n_targets, dtype=int)
    t_death = np.full(n_targets, -1)
    t_rows, e_rows, c_rows, episodes = [], [], [], []
    movie, lbl_T, lbl_E = [], [], []

    def free_angle(t_idx: int, wanted: float, me: _Effector) -> Optional[float]:
        """Closest docking angle to ``wanted`` that keeps clear of docked neighbours."""
        taken = [e.dock_angle for e in effectors if e is not me and e.state == "docked" and e.target == t_idx]
        min_sep = 2 * np.arcsin(min(1.0, (Re + 1.5) / contact_d))
        for delta in np.linspace(0, np.pi, 19):
            for a in (wanted + delta, wanted - delta):
                if all(abs((a - b + np.pi) % (2 * np.pi) - np.pi) >= min_sep for b in taken):
                    return a
        return None

    for t in range(n_frames):
        # Targets: an Ornstein-Uhlenbeck jiggle around their rest position.
        offset = 0.6 * offset + rng.normal(0, jiggle * 0.8, offset.shape)
        centres = rest + offset
        prev = [e.pos.copy() for e in effectors]
        was_docked = [e.state == "docked" for e in effectors]

        for i, e in enumerate(effectors):
            if e.state == "docked":
                e.timer -= 1
                if e.timer < 0:
                    episodes.append(
                        {"TARGET_ID": e.target + 1, "EFFECTOR_ID": i + 1, "t_start": e.t_start,
                         "t_end": t - 1, "duration": t - e.t_start}
                    )
                    e.state, e.last_target = "free", e.target
                    e.heading = e.dock_angle + rng.normal(0, 0.3)
                    e.timer = int(rng.integers(*free_frames, endpoint=True))
                    e.target = None
                else:
                    c = centres[e.target]
                    e.pos = c + contact_d * np.array([np.cos(e.dock_angle), np.sin(e.dock_angle)])
                    continue

            if e.state == "free":
                e.timer -= 1
                e.heading += rng.normal(0, 0.5)
                if e.timer < 0:
                    choices = [k for k in range(n_targets) if k != e.last_target] or list(range(n_targets))
                    d = [np.hypot(*(centres[k] - e.pos)) for k in choices]
                    e.state, e.target = "approach", choices[int(np.argmin(d))]

            if e.state == "approach":
                to_t = centres[e.target] - e.pos
                e.heading = np.arctan2(to_t[1], to_t[0]) + rng.normal(0, 0.25)
                dist = np.hypot(*to_t)
                if dist - effector_speed <= free_d:
                    angle = free_angle(e.target, np.arctan2(-to_t[1], -to_t[0]), e)
                    if angle is not None:
                        e.state, e.dock_angle, e.t_start = "docked", angle, t
                        e.timer = int(contact_durations[e.n_contacts % len(contact_durations)]) - 1
                        e.n_contacts += 1
                        e.pos = centres[e.target] + contact_d * np.array([np.cos(angle), np.sin(angle)])
                        continue
                    # Target crowded: try another one.
                    e.state, e.last_target, e.timer = "free", e.target, 1
                    e.target = None

            step = effector_speed * np.array([np.cos(e.heading), np.sin(e.heading)])
            e.pos = np.clip(e.pos + step, lo, hi)
            bounced = (e.pos <= lo) | (e.pos >= hi)
            if bounced[0]:
                e.heading = np.pi - e.heading
            if bounced[1]:
                e.heading = -e.heading

        # Keep moving effectors clear of targets and of each other.
        for _ in range(4):
            for i, e in enumerate(effectors):
                if e.state == "docked":
                    continue
                for c in centres:
                    v = e.pos - c
                    d = np.hypot(*v)
                    if d < free_d:
                        e.pos = c + v / max(d, 1e-6) * free_d
                for j, o in enumerate(effectors):
                    if j == i:
                        continue
                    v = e.pos - o.pos
                    d = np.hypot(*v)
                    if d < 2 * Re + 2:
                        e.pos = o.pos + v / max(d, 1e-6) * (2 * Re + 2)
                e.pos = np.clip(e.pos, lo, hi)
        # Free effectors must stay clear of the targets and of each other, so
        # that masks touch only during a docking and a threshold never merges
        # two effectors: keep a move that respects this, else the closest
        # valid position around where the effector came from.
        sep = 2 * Re + 2
        placed = [e.pos for e in effectors if e.state == "docked"]
        for i, e in enumerate(effectors):
            if e.state == "docked":
                continue

            def valid(p: np.ndarray) -> bool:
                return bool(np.all(np.hypot(*(centres - p).T) >= free_d)) and all(
                    np.hypot(*(p - q)) >= sep for q in placed
                )

            if not valid(e.pos):
                origin = prev[i] if not was_docked[i] else e.pos
                ring = np.linspace(0, 2 * np.pi, 16, endpoint=False)
                candidates = [origin] + [
                    np.clip(origin + r * np.array([np.cos(a), np.sin(a)]), lo, hi)
                    for r in (2, 4, 6, 9, 12, 16, 20) for a in ring
                ]
                ok = [c for c in candidates if valid(c)]
                if ok:
                    e.pos = min(ok, key=lambda c: np.hypot(*(c - e.pos)))
                    e.heading += rng.normal(np.pi / 2, 0.5)
            placed.append(e.pos)

        e_xy = np.array([e.pos for e in effectors])
        lt = _draw_disks(shape, centres, [Rt] * n_targets, range(1, n_targets + 1))
        le = _draw_disks(shape, e_xy, [Re] * n_effectors, range(1, n_effectors + 1))
        frame_contacts = _mask_contacts(lt, le, t, centres, e_xy)
        c_rows.extend(frame_contacts)

        for row in frame_contacts:
            contact_frames[row["TARGET_ID"] - 1] += 1
        if death_after is not None:
            newly = (contact_frames >= death_after) & (t_death < 0)
            t_death[newly] = t
        # Only draw when asked, so that a seed gives the same movie as before.
        if death_probability > 0:
            touched = np.zeros(n_targets, dtype=bool)
            touched[[row["TARGET_ID"] - 1 for row in frame_contacts]] = True
            newly = touched & (t_death < 0) & (rng.random(n_targets) < death_probability)
            # The contact comes first: the target is dead from the next frame.
            t_death[newly] = t + 1
        if background_death > 0:
            newly = (t_death < 0) & (rng.random(n_targets) < background_death)
            t_death[newly] = t

        channels = [
            _render(rng, lt > 0, foreground, background, noise),
            _render(rng, le > 0, foreground, background, noise),
        ]
        if combined_channel:
            level = np.where(le > 0, foreground, np.where(lt > 0, (background + foreground) / 2, background))
            channels.append(np.clip(level + rng.normal(0, noise, shape), 0, 65535).astype(np.uint16))
        if deaths:
            dead = np.isin(lt, np.flatnonzero((t_death >= 0) & (t_death <= t)) + 1)
            channels.append(_render(rng, dead, foreground, background, noise))
        movie.append(np.stack(channels))
        lbl_T.append(lt)
        lbl_E.append(le)

        for k, (x, y) in enumerate(centres):
            row = {"TRACK_ID": k + 1, "FRAME": t, "POSITION_X": x, "POSITION_Y": y, "class_id": k + 1, "radius": Rt}
            t_rows.append(row)
        for i, e in enumerate(effectors):
            e_rows.append(
                {"TRACK_ID": i + 1, "FRAME": t, "POSITION_X": e.pos[0], "POSITION_Y": e.pos[1],
                 "class_id": i + 1, "radius": Re, "state": e.state,
                 "docked_to": e.target + 1 if e.state == "docked" else np.nan}
            )

    for i, e in enumerate(effectors):
        if e.state == "docked":
            episodes.append(
                {"TARGET_ID": e.target + 1, "EFFECTOR_ID": i + 1, "t_start": e.t_start,
                 "t_end": n_frames - 1, "duration": n_frames - e.t_start}
            )

    targets = pd.DataFrame(t_rows)
    t_death[t_death >= n_frames] = -1  # killed by a contact at the last frame
    if deaths:
        td = t_death[targets["TRACK_ID"] - 1]
        targets["status"] = ((td >= 0) & (targets["FRAME"] >= td)).astype(int)
        targets["t_death"] = np.where(td >= 0, td, -1)
        targets["class_death"] = np.where(td >= 0, 0, 1)

    names = [TARGET_CHANNEL, EFFECTOR_CHANNEL]
    names += [COMBINED_CHANNEL] if combined_channel else []
    names += [DEATH_CHANNEL] if deaths else []
    episodes_df = pd.DataFrame(episodes, columns=["TARGET_ID", "EFFECTOR_ID", "t_start", "t_end", "duration"])
    return SyntheticInteractions(
        movie=np.stack(movie),
        channel_names=names,
        labels_targets=np.stack(lbl_T),
        labels_effectors=np.stack(lbl_E),
        targets=targets,
        effectors=pd.DataFrame(e_rows),
        contacts=pd.DataFrame(c_rows, columns=["FRAME", "TARGET_ID", "EFFECTOR_ID", "distance", "overlap"])
        .sort_values(["FRAME", "TARGET_ID", "EFFECTOR_ID"])
        .reset_index(drop=True),
        episodes=episodes_df.sort_values(["t_start", "EFFECTOR_ID"]).reset_index(drop=True),
        params={"shape": shape, "n_frames": n_frames, "seed": seed, "target_radius": Rt,
                "effector_radius": Re, "death_after": death_after,
                "death_probability": death_probability, "background_death": background_death,
                "combined_channel": combined_channel},
    )


def write_experiment(
    path: str,
    sim: SyntheticInteractions,
    movie_prefix: str = "synthetic",
    write_labels: bool = False,
    write_tables: bool = False,
    pxtoum: float = 1.0,
    frametomin: float = 1.0,
) -> str:
    """Write a simulation as a one-position celldetective experiment.

    The movie goes to ``W1/100/movie/{movie_prefix}.tif`` and the ground truth
    to ``W1/100/ground_truth/`` (label stacks and CSV tables), so that the
    segmentation, tracking and neighborhood steps can be run and checked
    against it.

    Parameters
    ----------
    path : str
        Experiment folder to create. It must not exist yet.
    sim : SyntheticInteractions
        Output of :func:`simulate_interactions`.
    movie_prefix : str
        Name of the movie file, also written to ``config.ini``.
    write_labels : bool
        Also write the true masks as ``labels_targets`` and
        ``labels_effectors``, to skip segmentation.
    write_tables : bool
        Also write the true trajectories as ``output/tables/trajectories_*.csv``,
        to skip measurement and tracking.
    pxtoum, frametomin : float
        Calibration written to ``config.ini``.

    Returns
    -------
    str
        Path to the position folder, ending with a separator.
    """
    import tifffile

    if os.path.exists(path):
        raise FileExistsError(f"{path} already exists; choose a new experiment folder.")
    pos = os.path.join(path, "W1", "100")
    gt = os.path.join(pos, "ground_truth")
    for d in ("movie", "output/tables", "ground_truth"):
        os.makedirs(os.path.join(pos, d))
    os.makedirs(os.path.join(path, "configs"))

    T, C, Y, X = sim.movie.shape
    config = ConfigParser(interpolation=None)
    config["Populations"] = {"populations": "effectors,targets"}
    config["MovieSettings"] = {
        "pxtoum": str(pxtoum), "frametomin": str(frametomin), "len_movie": str(T),
        "shape_x": str(X), "shape_y": str(Y), "movie_prefix": movie_prefix,
    }
    channels = {n: "nan" for n in (COMBINED_CHANNEL, TARGET_CHANNEL, DEATH_CHANNEL,
                                   EFFECTOR_CHANNEL, "adhesion_channel", "fluo_channel_1", "fluo_channel_2")}
    channels.update({name: str(i) for i, name in enumerate(sim.channel_names)})
    config["Channels"] = channels
    config["Labels"] = {"cell_types": "synthetic", "antibodies": "none",
                        "concentrations": "0", "pharmaceutical_agents": "none"}
    config["Metadata"] = {"concentration_units": "pM"}
    with open(os.path.join(path, "config.ini"), "w") as f:
        config.write(f)

    tifffile.imwrite(
        os.path.join(pos, "movie", f"{movie_prefix}.tif"), sim.movie, imagej=True, metadata={"axes": "TCYX"}
    )

    for population, labels, table in (
        ("targets", sim.labels_targets, sim.targets),
        ("effectors", sim.labels_effectors, sim.effectors),
    ):
        tifffile.imwrite(os.path.join(gt, f"labels_{population}.tif"), labels, imagej=True,
                         metadata={"axes": "TYX"})
        table.to_csv(os.path.join(gt, f"trajectories_{population}.csv"), index=False)
        if write_labels:
            folder = os.path.join(pos, f"labels_{population}")
            os.makedirs(folder)
            for t, frame in enumerate(labels):
                tifffile.imwrite(os.path.join(folder, f"{t:04d}.tif"), frame)
        if write_tables:
            cols = ["TRACK_ID", "FRAME", "POSITION_X", "POSITION_Y", "class_id"]
            cols += [c for c in ("status", "t_death", "class_death") if c in table.columns]
            table[cols].to_csv(os.path.join(pos, "output", "tables", f"trajectories_{population}.csv"), index=False)
    sim.contacts.to_csv(os.path.join(gt, "contacts.csv"), index=False)
    sim.episodes.to_csv(os.path.join(gt, "contact_episodes.csv"), index=False)
    return pos + os.sep


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Write a synthetic effector-target experiment.")
    p.add_argument("path", help="experiment folder to create")
    p.add_argument("--size", type=int, nargs=2, default=(160, 160), metavar=("Y", "X"))
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--targets", type=int, default=4)
    p.add_argument("--effectors", type=int, default=4)
    p.add_argument("--durations", type=int, nargs="+", default=[3, 12], help="contact durations in frames, cycled")
    p.add_argument("--death-after", type=int, default=None, help="kill targets after this many contact frames")
    p.add_argument("--death-probability", type=float, default=0.0,
                   help="chance a target dies at each frame it is in contact")
    p.add_argument("--background-death", type=float, default=0.0,
                   help="chance a target dies at each frame, whatever its contacts")
    p.add_argument("--no-combined", action="store_true", help="leave out the channel showing both populations")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--with-labels", action="store_true", help="write true masks to skip segmentation")
    p.add_argument("--with-tables", action="store_true", help="write true trajectories to skip tracking")
    args = p.parse_args(argv)
    sim = simulate_interactions(
        shape=tuple(args.size), n_frames=args.frames, n_targets=args.targets, n_effectors=args.effectors,
        contact_durations=args.durations, death_after=args.death_after,
        death_probability=args.death_probability, background_death=args.background_death,
        combined_channel=not args.no_combined, seed=args.seed,
    )
    pos = write_experiment(args.path, sim, write_labels=args.with_labels, write_tables=args.with_tables)
    print(f"Wrote {pos}: {len(sim.episodes)} contact episodes, {len(sim.contacts)} contact frames.")


if __name__ == "__main__":
    main()
