"""Small, conservative CPU program search for ARC tasks.

Only programs that reproduce EVERY train pair exactly are kept. Each kept program is
applied to the test inputs, and the distinct predictions are returned in order of
program simplicity. The notebook uses these ONLY to fill empty or duplicate attempt
slots; they never replace a model attempt_1.

Pure numpy + stdlib, no external dependencies. Safe to run in a background process.

Usage as a script:
    python arc_dsl.py CHALLENGES_JSON OUT_JSON [--time-limit SECONDS] [--per-task SECONDS]
writes {task_id: [[cand_grid, ...] per test input]} plus a "_programs" map with the names
of the programs that produced each candidate.
"""
import json
import sys
import time
import signal
import itertools
import numpy as np

# ----------------------------------------------------------------------------- helpers

MAX_SIDE = 30


def _valid(g):
    return (
        isinstance(g, np.ndarray)
        and g.ndim == 2
        and 1 <= g.shape[0] <= MAX_SIDE
        and 1 <= g.shape[1] <= MAX_SIDE
        and g.dtype.kind in "iu"
        and g.min() >= 0
        and g.max() <= 9
    )


DIHEDRAL = [
    ("id", lambda g: g),
    ("rot90", lambda g: np.rot90(g, 1)),
    ("rot180", lambda g: np.rot90(g, 2)),
    ("rot270", lambda g: np.rot90(g, 3)),
    ("flipud", lambda g: g[::-1, :]),
    ("fliplr", lambda g: g[:, ::-1]),
    ("transpose", lambda g: g.T),
    ("antitranspose", lambda g: np.rot90(g, 2).T),
]


def most_common_color(g):
    return int(np.bincount(g.ravel(), minlength=10).argmax())


def components(g, bg, diag, multicolor):
    """Connected components of non-bg cells. Returns list of (mask, colors_set)."""
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    if diag:
        nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for sy in range(h):
        for sx in range(w):
            if seen[sy, sx] or g[sy, sx] == bg:
                continue
            col = g[sy, sx]
            stack = [(sy, sx)]
            seen[sy, sx] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in nbrs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx]:
                        v = g[ny, nx]
                        if v == bg:
                            continue
                        if not multicolor and v != col:
                            continue
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            mask = np.zeros((h, w), dtype=bool)
            ys, xs = zip(*cells)
            mask[list(ys), list(xs)] = True
            comps.append(mask)
    return comps


def bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


# ----------------------------------------------------------------------------- base transforms
# Each "family" is a generator taking the list of train (inp, out) pairs and yielding
# (name, complexity, fn) with fn: np.ndarray -> np.ndarray | None.  Params that must be
# learned from data are derived from the train pairs; everything is verified afterwards.


def fam_dihedral(pairs):
    for i, (name, f) in enumerate(DIHEDRAL):
        yield name, 1 + (i > 0), f


def _ratio(pairs, num):
    """Consistent integer scale factors (out = in * k) or (in = out * k) if num=False."""
    rs = set()
    for a, b in pairs:
        if num:
            if b.shape[0] % a.shape[0] or b.shape[1] % a.shape[1]:
                return None
            rs.add((b.shape[0] // a.shape[0], b.shape[1] // a.shape[1]))
        else:
            if a.shape[0] % b.shape[0] or a.shape[1] % b.shape[1]:
                return None
            rs.add((a.shape[0] // b.shape[0], a.shape[1] // b.shape[1]))
    return rs.pop() if len(rs) == 1 else None


def fam_upscale(pairs):
    r = _ratio(pairs, True)
    if r and r != (1, 1):
        fy, fx = r
        yield f"upscale{fy}x{fx}", 2, lambda g, fy=fy, fx=fx: np.kron(g, np.ones((fy, fx), dtype=g.dtype))


def fam_downscale(pairs):
    r = _ratio(pairs, False)
    if not r or r == (1, 1):
        return
    fy, fx = r

    def mk(mode):
        def f(g):
            h, w = g.shape
            if h % fy or w % fx:
                return None
            blocks = g.reshape(h // fy, fy, w // fx, fx).transpose(0, 2, 1, 3).reshape(h // fy, w // fx, fy * fx)
            out = np.zeros((h // fy, w // fx), dtype=g.dtype)
            bg = most_common_color(g)
            for y in range(out.shape[0]):
                for x in range(out.shape[1]):
                    b = blocks[y, x]
                    if mode == "uniform":
                        if (b != b[0]).any():
                            return None
                        out[y, x] = b[0]
                    elif mode == "majority":
                        cnt = np.bincount(b, minlength=10)
                        if (cnt == cnt.max()).sum() > 1:
                            return None
                        out[y, x] = cnt.argmax()
                    elif mode == "anyfg":
                        fg = b[b != bg]
                        if len(fg) == 0:
                            out[y, x] = bg
                        else:
                            cnt = np.bincount(fg, minlength=10)
                            if (cnt == cnt.max()).sum() > 1:
                                return None
                            out[y, x] = cnt.argmax()
            return out
        return f

    for mode in ("uniform", "majority", "anyfg"):
        yield f"downscale{fy}x{fx}_{mode}", 3, mk(mode)


def fam_tile_dihedral(pairs):
    """out is an (a x b) arrangement of dihedral copies of the input (tiling / mirroring)."""
    r = _ratio(pairs, True)
    if not r or r == (1, 1):
        return
    a, b = r
    if a * b > 16:
        return
    layout = []
    for i in range(a):
        for j in range(b):
            ok_names = None
            for inp, out in pairs:
                h, w = inp.shape
                block = out[i * h:(i + 1) * h, j * w:(j + 1) * w]
                names = {n for n, f in DIHEDRAL if f(inp).shape == block.shape and np.array_equal(f(inp), block)}
                ok_names = names if ok_names is None else ok_names & names
                if not ok_names:
                    return
            # deterministic pick: first in DIHEDRAL order
            layout.append(next(n for n, _ in DIHEDRAL if n in ok_names))
    fns = dict(DIHEDRAL)

    def f(g, layout=tuple(layout)):
        h, w = g.shape
        out = np.zeros((a * h, b * w), dtype=g.dtype)
        for idx, n in enumerate(layout):
            t = fns[n](g)
            if t.shape != (h, w):
                return None
            i, j = divmod(idx, b)
            out[i * h:(i + 1) * h, j * w:(j + 1) * w] = t
        return out

    yield f"tile{a}x{b}[" + ",".join(layout) + "]", 3, f


def fam_fractal(pairs):
    """out = kron(mask(in), in): the classic self-similar tiling."""
    ok = all(b.shape == (a.shape[0] ** 2, a.shape[1] ** 2) for a, b in pairs)
    if not ok:
        return

    def mk(inv):
        def f(g):
            if g.shape[0] ** 2 > MAX_SIDE or g.shape[1] ** 2 > MAX_SIDE:
                return None
            bg = 0
            m = (g != bg) if not inv else (g == bg)
            return np.kron(m.astype(g.dtype), g)
        return f

    yield "fractal", 3, mk(False)
    yield "fractal_inv", 3, mk(True)


def _crop(g, bb):
    if bb is None:
        return None
    y0, y1, x0, x1 = bb
    return g[y0:y1, x0:x1]


def fam_crop(pairs):
    if all(a.shape == b.shape for a, b in pairs):
        return
    for bgmode in ("zero", "common"):
        def f(g, bgmode=bgmode):
            bg = 0 if bgmode == "zero" else most_common_color(g)
            return _crop(g, bbox(g != bg))
        yield f"crop_nonbg_{bgmode}", 2, f
    for c in range(10):
        if not all((a == c).any() for a, _ in pairs):
            continue

        def fc(g, c=c):
            return _crop(g, bbox(g == c))

        def fci(g, c=c):
            bb = bbox(g == c)
            if bb is None:
                return None
            y0, y1, x0, x1 = bb
            if y1 - y0 < 3 or x1 - x0 < 3:
                return None
            return g[y0 + 1:y1 - 1, x0 + 1:x1 - 1]

        yield f"crop_color{c}", 3, fc
        yield f"crop_color{c}_inner", 3, fci


_OBJ_KEYS = {
    "largest": lambda g, m: m.sum(),
    "smallest": lambda g, m: -m.sum(),
    "bbox_largest": lambda g, m: _bbarea(m),
    "bbox_smallest": lambda g, m: -_bbarea(m),
    "most_colors": lambda g, m: len(np.unique(g[m])),
    "fewest_colors": lambda g, m: -len(np.unique(g[m])),
    "topmost": lambda g, m: -bbox(m)[0],
    "bottommost": lambda g, m: bbox(m)[1],
    "leftmost": lambda g, m: -bbox(m)[2],
    "rightmost": lambda g, m: bbox(m)[3],
}


def _bbarea(m):
    y0, y1, x0, x1 = bbox(m)
    return (y1 - y0) * (x1 - x0)


def fam_objects(pairs):
    if all(a.shape == b.shape for a, b in pairs):
        return
    for bgmode, diag, multi in itertools.product(("zero", "common"), (False, True), (True, False)):
        for kname, kfn in _OBJ_KEYS.items():
            for outmode in ("bbox", "masked"):
                def f(g, bgmode=bgmode, diag=diag, multi=multi, kfn=kfn, outmode=outmode):
                    bg = 0 if bgmode == "zero" else most_common_color(g)
                    comps = components(g, bg, diag, multi)
                    if len(comps) < 2 or len(comps) > 60:
                        return None
                    vals = [kfn(g, m) for m in comps]
                    best = max(vals)
                    if sum(1 for v in vals if v == best) != 1:
                        return None  # ambiguous: refuse rather than guess
                    m = comps[vals.index(best)]
                    y0, y1, x0, x1 = bbox(m)
                    if outmode == "bbox":
                        return g[y0:y1, x0:x1].copy()
                    out = np.full((y1 - y0, x1 - x0), bg, dtype=g.dtype)
                    sub = m[y0:y1, x0:x1]
                    out[sub] = g[y0:y1, x0:x1][sub]
                    return out
                yield f"object[{bgmode},{'8' if diag else '4'}conn,{'multi' if multi else 'mono'},{kname},{outmode}]", 4, f
        # object whose colour is unique among objects (e.g. the odd one out)
        def fu(g, bgmode=bgmode, diag=diag, multi=multi):
            bg = 0 if bgmode == "zero" else most_common_color(g)
            comps = components(g, bg, diag, multi)
            if len(comps) < 3 or len(comps) > 60:
                return None
            sigs = [tuple(sorted(np.unique(g[m]).tolist())) for m in comps]
            cnt = {s: sigs.count(s) for s in sigs}
            uniq = [i for i, s in enumerate(sigs) if cnt[s] == 1]
            if len(uniq) != 1:
                return None
            y0, y1, x0, x1 = bbox(comps[uniq[0]])
            return g[y0:y1, x0:x1].copy()
        yield f"object[{bgmode},{'8' if diag else '4'}conn,{'multi' if multi else 'mono'},unique_colors,bbox]", 4, fu

        def fs(g, bgmode=bgmode, diag=diag, multi=multi):
            bg = 0 if bgmode == "zero" else most_common_color(g)
            comps = components(g, bg, diag, multi)
            if len(comps) < 3 or len(comps) > 60:
                return None
            crops = []
            for m in comps:
                y0, y1, x0, x1 = bbox(m)
                crops.append((m[y0:y1, x0:x1].tobytes(), m[y0:y1, x0:x1].shape))
            cnt = {s: crops.count(s) for s in crops}
            uniq = [i for i, s in enumerate(crops) if cnt[s] == 1]
            if len(uniq) != 1:
                return None
            y0, y1, x0, x1 = bbox(comps[uniq[0]])
            return g[y0:y1, x0:x1].copy()
        yield f"object[{bgmode},{'8' if diag else '4'}conn,{'multi' if multi else 'mono'},unique_shape,bbox]", 4, fs


def _split_parts(g, n, axis):
    """Split g into n equal parts along axis, allowing 1-wide uniform separator lines."""
    size = g.shape[axis]
    for sep in (0, 1):
        tot = size - sep * (n - 1)
        if tot <= 0 or tot % n:
            continue
        p = tot // n
        parts = []
        seps = []
        for i in range(n):
            s = i * (p + sep)
            parts.append(g[s:s + p] if axis == 0 else g[:, s:s + p])
            if sep and i < n - 1:
                line = g[s + p] if axis == 0 else g[:, s + p]
                seps.append(line)
        if sep:
            # separators must be uniform lines of one colour
            if not all((ln == ln[0]).all() for ln in seps):
                continue
        return parts
    return None


_BOOL_OPS = {
    "and": lambda a, b: a & b,
    "or": lambda a, b: a | b,
    "xor": lambda a, b: a ^ b,
    "nor": lambda a, b: ~(a | b),
    "a_not_b": lambda a, b: a & ~b,
    "b_not_a": lambda a, b: b & ~a,
    "nand": lambda a, b: ~(a & b),
}


def fam_parts(pairs):
    # Candidate split: n parts along an axis such that part shape == output shape
    for n, axis in itertools.product((2, 3, 4), (0, 1)):
        good = True
        for a, b in pairs:
            ps = _split_parts(a, n, axis)
            if ps is None or ps[0].shape != b.shape:
                good = False
                break
        if not good:
            continue
        # boolean ops on two parts: learn (true_colour, false_colour)
        if n == 2:
            for opname, op in _BOOL_OPS.items():
                tc = fc = None
                ok = True
                for a, b in pairs:
                    pa, pb = _split_parts(a, 2, axis)
                    bga = 0
                    m = op(pa != bga, pb != bga)
                    tv = np.unique(b[m])
                    fv = np.unique(b[~m])
                    if len(tv) > 1 or len(fv) > 1:
                        ok = False
                        break
                    if len(tv):
                        if tc is not None and tc != tv[0]:
                            ok = False
                            break
                        tc = int(tv[0])
                    if len(fv):
                        if fc is not None and fc != fv[0]:
                            ok = False
                            break
                        fc = int(fv[0])
                if not ok or tc is None or fc is None or tc == fc:
                    continue

                def f(g, op=op, tc=tc, fc=fc, axis=axis):
                    ps = _split_parts(g, 2, axis)
                    if ps is None:
                        return None
                    m = op(ps[0] != 0, ps[1] != 0)
                    return np.where(m, tc, fc).astype(g.dtype)
                yield f"parts2_axis{axis}_{opname}_t{tc}_f{fc}", 3, f
        # overlay of parts with a priority order (non-zero cells of earlier parts win)
        for order in itertools.permutations(range(n)):
            def fo(g, order=order, n=n, axis=axis):
                ps = _split_parts(g, n, axis)
                if ps is None:
                    return None
                out = np.zeros_like(ps[0])
                for i in reversed(order):
                    out = np.where(ps[i] != 0, ps[i], out)
                return out
            yield f"parts{n}_axis{axis}_overlay{''.join(map(str, order))}", 4, fo


def fam_symmetry_complete(pairs):
    """Fill cells of a 'hole' colour using the grid's own mirror/rotational symmetry."""
    if not all(a.shape == b.shape for a, b in pairs):
        return
    diffs = set()
    for a, b in pairs:
        d = a != b
        if not d.any():
            return
        vals = np.unique(a[d])
        if len(vals) != 1:
            return
        diffs.add(int(vals[0]))
    if len(diffs) != 1:
        return
    hole = diffs.pop()

    def f(g, hole=hole):
        out = g.copy()
        m = out == hole
        if not m.any():
            return None
        cands = [g[::-1, :], g[:, ::-1], np.rot90(g, 2)]
        if g.shape[0] == g.shape[1]:
            cands += [g.T, np.rot90(g, 2).T, np.rot90(g, 1), np.rot90(g, 3)]
        for _ in range(3):
            for c in cands:
                fill = m & (c != hole)
                out[fill] = c[fill]
                m = out == hole
            cands = [out[::-1, :], out[:, ::-1], np.rot90(out, 2)] + (
                [out.T, np.rot90(out, 2).T, np.rot90(out, 1), np.rot90(out, 3)] if g.shape[0] == g.shape[1] else [])
        if m.any():
            return None
        return out

    yield f"symfill{hole}", 4, f


FAMILIES = [
    fam_dihedral,
    fam_upscale,
    fam_downscale,
    fam_tile_dihedral,
    fam_fractal,
    fam_crop,
    fam_objects,
    fam_parts,
    fam_symmetry_complete,
]


# ----------------------------------------------------------------------------- composition + verification

def _learn_cmap(preds, outs):
    """Colour mapping m with m[pred] == out, consistent over all pairs. None if impossible."""
    m = {}
    for p, o in zip(preds, outs):
        if p.shape != o.shape:
            return None
        for pc, oc in zip(p.ravel().tolist(), o.ravel().tolist()):
            if m.setdefault(pc, oc) != oc:
                return None
    return m


def _apply_cmap(g, m):
    vals = np.unique(g).tolist()
    if any(v not in m for v in vals):
        return None  # unseen colour: refuse
    lut = np.arange(10)
    for k, v in m.items():
        lut[k] = v
    return lut[g]


def _safe(fn, g):
    try:
        r = fn(g)
    except Exception:
        return None
    if r is None:
        return None
    r = np.asarray(r)
    if r.dtype.kind not in "iu":
        return None
    r = r.astype(np.int64)
    return r if _valid(r) else None


def find_programs(task, deadline=None):
    """Return list of (name, complexity, fn) reproducing all train pairs, simplest first."""
    pairs = [(np.array(p["input"], dtype=np.int64), np.array(p["output"], dtype=np.int64)) for p in task["train"]]
    found = []
    for fam in FAMILIES:
        try:
            gen = list(fam(pairs))
        except Exception:
            continue
        for name, cx, fn in gen:
            if deadline is not None and time.time() > deadline:
                return sorted(found, key=lambda x: x[1])
            preds = [_safe(fn, a) for a, _ in pairs]
            if any(p is None for p in preds):
                continue
            if all(p.shape == b.shape and np.array_equal(p, b) for p, (_, b) in zip(preds, pairs)):
                found.append((name, cx, fn))
                continue
            # geometric post-transform and/or colour map on top of the base program
            for gname, gf in DIHEDRAL:
                gp = [gf(p) for p in preds]
                if gname != "id" and all(p.shape == b.shape and np.array_equal(p, b) for p, (_, b) in zip(gp, pairs)):
                    found.append((f"{name}|{gname}", cx + 1, (lambda g, fn=fn, gf=gf: (lambda r: None if r is None else gf(r))(_safe(fn, g)))))
                    continue
                m = _learn_cmap(gp, [b for _, b in pairs])
                if m is None or all(k == v for k, v in m.items()):
                    continue
                found.append((f"{name}|{gname}|cmap{sorted(m.items())}", cx + 2,
                              (lambda g, fn=fn, gf=gf, m=m: (lambda r: None if r is None else _apply_cmap(gf(r), m))(_safe(fn, g)))))
    found.sort(key=lambda x: x[1])  # stable: family order then complexity
    return found


def solve_task(task, max_candidates=2, per_task_seconds=10.0):
    """Returns (predictions, program_names): predictions[t] is a list (<= max_candidates)
    of distinct grids (lists) for test input t, possibly empty."""
    deadline = time.time() + per_task_seconds
    progs = find_programs(task, deadline=deadline)
    preds, names = [], []
    for t in task["test"]:
        g = np.array(t["input"], dtype=np.int64)
        seen, cands, cn = set(), [], []
        for name, _, fn in progs:
            r = _safe(fn, g)
            if r is None:
                continue
            key = (r.shape, r.tobytes())
            if key in seen:
                continue
            seen.add(key)
            cands.append(r.tolist())
            cn.append(name)
            if len(cands) >= max_candidates:
                break
        preds.append(cands)
        names.append(cn)
    return preds, names


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def solve_all(challenges, time_limit=1800.0, per_task=10.0, verbose=False):
    t_end = time.time() + time_limit
    out, progs = {}, {}
    use_alarm = hasattr(signal, "SIGALRM")
    if use_alarm:
        signal.signal(signal.SIGALRM, _alarm)
    for tid in sorted(challenges):
        if time.time() > t_end:
            break
        try:
            if use_alarm:
                signal.alarm(int(per_task * 2) + 1)
            p, n = solve_task(challenges[tid], per_task_seconds=per_task)
        except BaseException as e:  # includes _Timeout
            if isinstance(e, KeyboardInterrupt):
                raise
            p, n = [[] for _ in challenges[tid]["test"]], [[] for _ in challenges[tid]["test"]]
        finally:
            if use_alarm:
                signal.alarm(0)
        if any(len(x) for x in p):
            out[tid] = p
            progs[tid] = n
            if verbose:
                print(f"[dsl] {tid}: {n}", flush=True)
    return out, progs


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("challenges")
    ap.add_argument("out")
    ap.add_argument("--time-limit", type=float, default=1800.0)
    ap.add_argument("--per-task", type=float, default=10.0)
    args = ap.parse_args()
    with open(args.challenges) as f:
        ch = json.load(f)
    t0 = time.time()
    preds, progs = solve_all(ch, time_limit=args.time_limit, per_task=args.per_task, verbose=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"predictions": preds, "programs": progs}, f)
    import os
    os.replace(tmp, args.out)
    print(f"[dsl] wrote {len(preds)} tasks with candidates in {time.time() - t0:.1f}s", flush=True)
