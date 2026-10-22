# -*- coding: utf-8 -*-
"""
Created on Sat Oct 3 17:29:36 2026

@author: sabbi002
"""

import glob
import io
import os
import re
import shutil
import sys
import time
from collections import Counter
from itertools import islice
from operator import itemgetter

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

MODE = "merge"   # here I have "scan", "merge" or "check" Change it to merge before running
#scan mode check only the scan of your file
#check funciton will not work in your case because here I check everything against numpy array formate
#the numpy array formate is easy for me to built graphs, but for you tecplot combination is ok as you are using Tecplot 10

TEC_DIR     = r"d:\Mekong Delta models\25year_models\Variable_density_model"               #here input your directory
TEC_PATTERN = "CONCVELO.TEC.p*"                                        #change the name according to your tecplot file name, but the last part p* is important
OUT_TEC     = os.path.join(TEC_DIR, "CONCVELO_merged.TEC")

PART_TABLE = r"""
p000 :  811   454 9204850     1   811     1   454 |   811   456 9245400     1   811     1   456
p001 :  811   454 9204850     1   811   455   908 |   811   456 9245400     1   811   453   908
"""

TIME_EVERY = 1                   # All the time steps
MAX_ZONES  = None                # For example use 2 for a quick run

CHECK_NPZ   = os.path.join(TEC_DIR, "VDF_2089_tec.npz")
CHECK_ZONES = (1, 2, -1)
NODATA_VALUES = (-9999.99, -9999.0, -9998.0)
NODATA_TOL    = 1.0
NODATA_BELOW  = -9000.0


_ZONE_RE = re.compile(rb'T\s*=\s*"([^"]*)".*?\bI\s*=\s*(\d+).*?\bJ\s*=\s*(\d+)'
                      rb'.*?\bK\s*=\s*(\d+)', re.IGNORECASE)
_NUM_RE = re.compile(rb'[-+]?[0-9]*\.?[0-9]+(?:[eEdD][-+]?[0-9]+)?')
_SHARE_RE = re.compile(rb'\bD\s*=\s*\(([^)]*)\)', re.IGNORECASE)
_VARS_RE = re.compile(rb'"([^"]*)"')


def read_names(line):
    return [s.strip().upper().decode("ascii", "replace") for s in _VARS_RE.findall(line)]


def parse_zone_header(line):
    m = _ZONE_RE.search(line)
    if not m:
        return None
    num = _NUM_RE.search(m.group(1))
    t = float(num.group(0).replace(b"D", b"E").replace(b"d", b"e")) if num else np.nan
    s = _SHARE_RE.search(line)
    shared = [int(x) - 1 for x in re.findall(rb"\d+", s.group(1))] if s else []
    return t, int(m.group(2)), int(m.group(3)), int(m.group(4)), shared


class TecStream:

    def __init__(self, path):
        self.path = path
        self.fh = open(path, "rb", buffering=1 << 22)
        self._back = []
        self.preamble, self.between = [], []
        self.header, self.shared, self.dims, self.t = None, [], None, np.nan
        self.stray = 0
        while True:
            line = self.fh.readline()
            if not line:
                raise ValueError("%s: no VARIABLES line" % path)
            if line.lstrip()[:4].upper() == b"ZONE":
                raise ValueError("%s: ZONE before VARIABLES" % path)
            self.preamble.append(line)
            if b"VARIABLES" in line.upper():
                self.names = read_names(line)
                break

    def _readline(self):
        return self._back.pop(0) if self._back else self.fh.readline()

    def _take(self, n):
        out, self._back = self._back[:n], self._back[n:]
        if len(out) < n:
            out.extend(islice(self.fh, n - len(out)))
        return out

    def next_zone(self):
        self.between = []
        while True:
            line = self._readline()
            if not line:
                return None
            s = line.lstrip()
            if s[:4].upper() == b"ZONE":
                h = parse_zone_header(line)
                if h is None:
                    raise ValueError("%s: unreadable zone header:\n%r" % (self.path, line))
                self.t, ni, nj, nk, self.shared = h
                self.header, self.dims = line, (ni, nj, nk)
                return self.t, ni, nj, nk
            if s[:1].isalpha():
                self.between.append(line)
            elif s.strip():
                self.stray += 1

    def has_xyz(self):
        return not ({0, 1, 2} & set(self.shared))

    def read_block(self, n):
        lines = self._take(n)
        if not lines:
            return lines
        if (self.has_xyz() and len(lines) == n
                and lines[-1].startswith(b"%d,%d,%d," % self.dims)):
            return lines
        txt = b"".join(lines)
        hits = [p for p in (txt.find(b"ZONE"), txt.find(b"TEXT")) if p >= 0]
        if hits:
            z = txt.count(b"\n", 0, min(hits))
            self._back = lines[z:] + self._back
            lines = lines[:z]
        return lines

    def close(self):
        try:
            self.fh.close()
        except Exception:
            pass


def prefix_lengths(ni, nj, nk):
    def digits(n):
        return np.array([len(str(v)) for v in range(1, n + 1)])
    return (digits(nk)[:, None, None] + digits(nj)[None, :, None]
            + digits(ni)[None, None, :] + 3).ravel()


def fast_rests(lines, ni, nj, nk, plen):
    n = ni * nj * nk
    if (len(lines) != n or not lines[0].startswith(b"1,1,1,")
            or not lines[-1].startswith(b"%d,%d,%d," % (ni, nj, nk))):
        return None
    step = ni * nj
    for k in range(1, nk):
        if not lines[k * step].startswith(b"1,1,%d," % (k + 1)):
            return None
    r = np.fromiter(map(len, lines), np.int64, n) - plen
    r0 = int(r[0])
    if r0 < 2 or (r != r0).any():
        return None
    return list(map(itemgetter(slice(-r0, None)), lines))


def nodata_rest(names, eol):
    return b",".join(b"-.9999E+04" if v in ("VX", "VY", "VZ") else b"-.999999E+04"
                     for v in names) + eol


def slow_rests(lines, ni, nj, nk, has_xyz, rest_names, eol, nf_hint=None):
    n = ni * nj * nk
    parsed, bad = [], 0
    for pos, line in enumerate(lines):
        if has_xyz:
            f = line.split(b",", 3)
            try:
                i, j, k = int(f[0]), int(f[1]), int(f[2])
                rest = f[3]
            except (ValueError, IndexError):
                bad += 1
                continue
            if not (0 < i <= ni and 0 < j <= nj and 0 < k <= nk):
                bad += 1
                continue
            idx = ((k - 1) * nj + (j - 1)) * ni + (i - 1)
        else:
            idx, rest = pos, line
        if not rest.endswith(b"\n"):
            rest = rest.rstrip() + eol
        parsed.append((idx, rest))
    common = Counter((r.count(b","), len(r)) for _, r in parsed).most_common(1)
    if common:
        (ncomma, length), _ = common[0]
        nf = ncomma + 1
    else:
        nf, length = (nf_hint or len(rest_names)), None
    rests = [nodata_rest(rest_names[:nf], eol)] * n
    have = np.zeros(n, dtype=bool)
    for idx, rest in parsed:
        if len(rest) == length and rest.count(b",") == nf - 1:
            rests[idx] = rest
            have[idx] = True
        else:
            bad += 1
    return rests, have, bad, nf


def describe_gap(pn, iz, t, nlines, n, bad, have, src):
    j0, j1, i0, i1 = src
    own = ~have[:, j0:j1, i0:i1]
    msg = "p%03d zone %d (%.3f yr): %d of %d lines" % (pn, iz, t, nlines, n)
    if bad:
        msg += ", %d unusable" % bad
    msg += ", %d cells without a line -> " % int((~have).sum())
    n_own = int(own.sum())
    if n_own == 0:
        return msg + "none inside its own box, nothing lost"
    lay = np.nonzero(own.any(axis=(1, 2)))[0] + 1
    return msg + "%d inside its own box, written as nodata (layer %s)" % (
        n_own, lay[0] if len(lay) == 1 else "%d..%d" % (lay[0], lay[-1]))


def count_zones(path):
    n, tail = 0, b""
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 22)
            if not chunk:
                break
            buf = tail + chunk
            n += buf.count(b"ZONE")
            tail = buf[-3:]
    return n


def part_number(path):
    m = re.search(r"[.\-_]p(\d+)$", path, re.IGNORECASE)
    return int(m.group(1)) if m else -1


def find_parts(tec_dir, pattern):
    hits = glob.glob(os.path.join(tec_dir, pattern))
    paths = sorted((f for f in hits if part_number(f) >= 0), key=part_number)
    if len(paths) < len(hits):
        print("   (skipped %d other file(s) matching %s)" % (len(hits) - len(paths), pattern))
    if not paths:
        raise SystemExit("no files matching %s in %s" % (pattern, tec_dir))
    return paths


_PART_RE = re.compile(
    r"^\s*p(\d+)\s*:\s*"
    r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
    r"\s*\|\s*"
    r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
    re.MULTILINE)


def parse_part_table(text):
    table = {}
    for m in _PART_RE.finditer(text):
        g = [int(x) for x in m.groups()]
        p = g[0]
        own_nc, own_nr, _n, o_c0, o_c1, o_r0, o_r1 = g[1:8]
        ext_nc, ext_nr, _n, e_c0, e_c1, e_r0, e_r1 = g[8:15]
        for lab, nc, nr, c0, c1, r0, r1 in (("own", own_nc, own_nr, o_c0, o_c1, o_r0, o_r1),
                                            ("ext", ext_nc, ext_nr, e_c0, e_c1, e_r0, e_r1)):
            if c1 - c0 + 1 != nc or r1 - r0 + 1 != nr:
                raise ValueError("p%03d %s box %d..%d x %d..%d does not match %d x %d"
                                 % (p, lab, c0, c1, r0, r1, nc, nr))
        table[p] = dict(own=(o_c0, o_c1, o_r0, o_r1), ext=(e_c0, e_c1, e_r0, e_r1),
                        ni=ext_nc, nj=ext_nr)
    if not table:
        raise SystemExit("PART_TABLE has no readable 'p### : ... | ...' lines")
    return table


def load_plan(part_table):
    table = parse_part_table(part_table)
    ncol = max(t["own"][1] for t in table.values())
    nrow = max(t["own"][3] for t in table.values())
    plan = {}
    for p, t in table.items():
        oc0, oc1, or0, or1 = t["own"]
        ec0, ec1, er0, er1 = t["ext"]
        if oc0 < ec0 or oc1 > ec1 or or0 < er0 or or1 > er1:
            raise SystemExit("p%03d: own box is not inside the extended box" % p)
        plan[p] = dict(shape=(t["ni"], t["nj"]),
                       src=(or0 - er0, or1 - er0 + 1, oc0 - ec0, oc1 - ec0 + 1),
                       dst=(or0 - 1, oc0 - 1))
    hits = np.zeros((nrow, ncol), np.int16)
    for p in plan:
        j0, j1, i0, i1 = plan[p]["src"]
        r0, c0 = plan[p]["dst"]
        hits[r0:r0 + j1 - j0, c0:c0 + i1 - i0] += 1
    print("   PART_TABLE: %d partitions, grid %d rows x %d cols; coverage: %d cells "
          "once, %d never, %d more than once" % (len(plan), nrow, ncol, (hits == 1).sum(),
                                                 (hits == 0).sum(), (hits > 1).sum()))
    if (hits != 1).any():
        raise SystemExit("   !! the own boxes do not tile the grid, check PART_TABLE")
    return plan, nrow, ncol


_T_RE = re.compile(rb'T\s*=\s*"[^"]*"', re.IGNORECASE)
_DIM_RE = {key: re.compile(rb"(\b" + key + rb"\s*=)(\s*)(\d+)", re.IGNORECASE)
           for key in (b"I", b"J", b"K")}


def merged_header(line, ni, nj, nk):
    m = _T_RE.search(line)
    cut = m.end() if m else 0
    head, tail = line[:cut], line[cut:]
    for key, val in ((b"I", ni), (b"J", nj), (b"K", nk)):
        tail = _DIM_RE[key].sub(
            lambda g, v=val: g.group(1) + b"%*d" % (len(g.group(2)) + len(g.group(3)), v),
            tail, count=1)
    return head + tail


def disk_need(paths, nz, n_out, out_tec):
    total = sum(os.path.getsize(p) for p in paths)
    need = total * n_out / max(nz, 1)
    folder = os.path.dirname(os.path.abspath(out_tec))
    free = shutil.disk_usage(folder).free
    print("   partition files %.1f GB together -> merged file about %.1f GB; "
          "%.1f GB free in %s" % (total / 1e9, need / 1e9, free / 1e9, folder))
    return need, free


def scan(tec_dir=TEC_DIR, pattern=TEC_PATTERN, out_tec=OUT_TEC,
         part_table=PART_TABLE, time_every=TIME_EVERY, max_zones=MAX_ZONES):
    paths = find_parts(tec_dir, pattern)
    print("\n%d partition files in %s" % (len(paths), tec_dir))
    plan, nrow, ncol = load_plan(part_table)
    streams = [TecStream(p) for p in paths]
    trouble = 0
    try:
        names = streams[0].names
        print("   variables: %s\n" % ", ".join(names))
        for s in streams:
            pn = part_number(s.path)
            h = s.next_zone()
            if h is None:
                raise SystemExit("%s: no zones" % s.path)
            t, ni, nj, nk = h
            if pn not in plan:
                print("   !! p%03d is not in PART_TABLE" % pn)
                trouble += 1
                continue
            first = s._readline()
            j0, j1, i0, i1 = plan[pn]["src"]
            r0, c0 = plan[pn]["dst"]
            note = ""
            if plan[pn]["shape"] != (ni, nj):
                note += "   !! the table says I=%d J=%d" % plan[pn]["shape"]
            if not first.startswith(b"1,1,1,"):
                note += "   ! first line is not '1,1,1,...' (slow reading): %r" % first[:40]
            if s.names != names:
                note += "   !! other variables: %s" % s.names
            trouble += bool(note)
            print("   p%03d  I=%3d J=%3d K=%3d  t=%8.3f yr  own cells -> rows %3d..%3d, "
                  "cols %3d..%3d%s" % (pn, ni, nj, nk, t, r0 + 1, r0 + j1 - j0,
                                       c0 + 1, c0 + i1 - i0, note))
        missing = sorted(set(plan) - {part_number(p) for p in paths})
        if missing:
            print("   !! no file for partition(s) %s" % missing)
            trouble += 1
        print("\n   merged zone header:\n   %s"
              % merged_header(streams[0].header, ncol, nrow, nk).decode("ascii", "replace").rstrip())
    finally:
        for s in streams:
            s.close()
    nz = count_zones(paths[0])
    n_out = len(range(0, nz, time_every))
    if max_zones:
        n_out = min(n_out, int(max_zones))
    print("   zones in p000: %d, to be written: %d" % (nz, n_out))
    need, free = disk_need(paths, nz, n_out, out_tec)
    if need > 0.95 * free:
        print("   !! not enough free disk space -- use TIME_EVERY / MAX_ZONES or "
              "another OUT_TEC drive")
        trouble += 1
    print("\n   %s" % ("everything matches" if not trouble else
                       "%d problem(s) above, fix them before merging" % trouble))


def merge(tec_dir=TEC_DIR, pattern=TEC_PATTERN, out_tec=OUT_TEC,
          part_table=PART_TABLE, time_every=TIME_EVERY, max_zones=MAX_ZONES):
    t_start = time.time()
    paths = find_parts(tec_dir, pattern)
    plan, nrow, ncol = load_plan(part_table)
    pnums = [part_number(p) for p in paths]
    if sorted(pnums) != sorted(plan):
        raise SystemExit("partition files %s, PART_TABLE %s -- they must match"
                         % (pnums, sorted(plan)))
    nz = count_zones(paths[0])
    n_out = len(range(0, nz, time_every))
    if max_zones:
        n_out = min(n_out, int(max_zones))
    print("   %d partitions, %d zones, writing %d" % (len(paths), nz, n_out))
    need, free = disk_need(paths, nz, n_out, out_tec)
    if need > 0.95 * free:
        raise SystemExit("   not enough free disk space -- use TIME_EVERY / MAX_ZONES "
                         "or put OUT_TEC on another drive")

    streams = [TecStream(p) for p in paths]
    names = streams[0].names
    if names[:3] != ["X", "Y", "Z"]:
        raise SystemExit("expected X, Y, Z as the first variables, found %s" % names)
    for s in streams[1:]:
        if s.names != names:
            raise SystemExit("%s has variables %s, p%03d has %s"
                             % (s.path, s.names, pnums[0], names))

    pre_ij = [b"%d,%d," % (i, j) for j in range(1, nrow + 1) for i in range(1, ncol + 1)]
    m = len(pre_ij)
    buf = [None] * (3 * m)
    buf[0::3] = pre_ij

    grid, plens = None, {}
    index, problems, no_vel = [], [], []
    slow = written = iz = nk = 0
    tmp = out_tec + ".tmp"
    done = False
    try:
        with open(tmp, "wb", buffering=1 << 22) as fo:
            fo.writelines(streams[0].preamble)
            while not (max_zones and written >= max_zones):
                hdrs = [s.next_zone() for s in streams]
                if all(h is None for h in hdrs):
                    break
                if any(h is None for h in hdrs):
                    print("   ! %s ran out of zones after zone %d, stopping here"
                          % (", ".join("p%03d" % p for p, h in zip(pnums, hdrs) if h is None), iz))
                    break
                iz += 1
                t0, nk = hdrs[0][0], hdrs[0][3]
                for s, pn, (t, ni, nj, nkp) in zip(streams, pnums, hdrs):
                    if (ni, nj) != plan[pn]["shape"] or nkp != nk:
                        raise SystemExit("p%03d zone %d: I=%d J=%d K=%d, but the table says "
                                         "I=%d J=%d (and p%03d has K=%d) -- wrong PART_TABLE?"
                                         % ((pn, iz, ni, nj, nkp) + plan[pn]["shape"]
                                            + (pnums[0], nk)))
                    if s.shared != streams[0].shared:
                        raise SystemExit("zone %d: p%03d shares %s, p%03d shares %s"
                                         % (iz, pn, s.shared, pnums[0], streams[0].shared))
                    if abs(t - t0) > 1e-6:
                        print("   ! zone %d: p%03d at %.4f yr, p%03d at %.4f yr"
                              % (iz, pn, t, pnums[0], t0))
                if (iz - 1) % time_every:
                    for s, (t, ni, nj, nkp) in zip(streams, hdrs):
                        s.read_block(ni * nj * nkp)
                    continue

                if grid is None or grid.shape[0] != nk:
                    grid = np.empty((nk, nrow, ncol), dtype=object)
                xyz = streams[0].has_xyz()
                rest_names = [v for c, v in enumerate(names)
                              if c >= 3 and c not in streams[0].shared]
                nf_zone = None
                for s, pn, (t, ni, nj, _) in zip(streams, pnums, hdrs):
                    n = ni * nj * nk
                    eol = b"\r\n" if s.header.endswith(b"\r\n") else b"\n"
                    lines = s.read_block(n)
                    rests = None
                    if xyz:
                        if (ni, nj, nk) not in plens:
                            plens[(ni, nj, nk)] = prefix_lengths(ni, nj, nk)
                        rests = fast_rests(lines, ni, nj, nk, plens[(ni, nj, nk)])
                    if rests is not None:
                        nf = rests[0].count(b",") + 1
                    else:
                        slow += 1
                        rests, have, bad, nf = slow_rests(lines, ni, nj, nk, xyz,
                                                          rest_names, eol, nf_zone)
                        if bad or not have.all():
                            msg = describe_gap(pn, iz, t, len(lines), n, bad,
                                               have.reshape(nk, nj, ni), plan[pn]["src"])
                            print("   ! " + msg)
                            problems.append(msg)
                    if nf_zone is None:
                        nf_zone = nf
                    elif nf != nf_zone:
                        raise SystemExit("zone %d: p%03d has %d values per line after X, Y, Z,"
                                         " p%03d has %d -- the partition files disagree"
                                         % (iz, pn, nf, pnums[0], nf_zone))
                    block = np.empty(n, dtype=object)
                    block[:] = rests
                    block = block.reshape(nk, nj, ni)
                    j0, j1, i0, i1 = plan[pn]["src"]
                    r0, c0 = plan[pn]["dst"]
                    grid[:, r0:r0 + j1 - j0, c0:c0 + i1 - i0] = block[:, j0:j1, i0:i1]
                    del lines, rests, block
                if nf_zone < len(rest_names):
                    no_vel.append(iz)
                    if len(no_vel) == 1:
                        print("   zone %d has no %s in the partition files (not saved at that"
                              " time): written the same way, reported once"
                              % (iz, ", ".join(rest_names[nf_zone:])))

                index.append((written + 1, iz, t0, fo.tell()))
                fo.writelines(streams[0].between)
                fo.write(merged_header(streams[0].header, ncol, nrow, nk))
                for k in range(nk):
                    cells = grid[k].ravel().tolist()
                    if xyz:
                        buf[1::3] = [b"%d," % (k + 1)] * m
                        buf[2::3] = cells
                        fo.write(b"".join(buf))
                    else:
                        fo.write(b"".join(cells))
                written += 1

                el = time.time() - t_start
                if written == 1:
                    one = fo.tell() - index[0][3]
                    print("   zone 1 done: %.0f MB per zone -> merged file %.1f GB"
                          % (one / 1e6, one * n_out / 1e9))
                if written % 10 == 0 or written == n_out:
                    print("      %d / %d zones  (%.0f s, about %.0f min left)"
                          % (written, n_out, el, el / written * (n_out - written) / 60))
        done = True
    finally:
        for s in streams:
            if s.stray:
                print("   ! %s: %d data line(s) outside any zone were skipped"
                      % (os.path.basename(s.path), s.stray))
            s.close()
        if not done and os.path.exists(tmp):
            os.remove(tmp)
            print("   (unfinished %s removed)" % os.path.basename(tmp))

    if not written:
        os.remove(tmp)
        raise SystemExit("no zone written")
    try:
        os.replace(tmp, out_tec)
    except OSError as e:
        raise SystemExit("could not rename %s to %s (%s) -- is the old file open in "
                         "another program?" % (tmp, out_tec, e))
    with open(out_tec + ".zones.csv", "w") as f:
        f.write("zone,source_zone,time_years,byte_offset\n")
        for row in index:
            f.write("%d,%d,%.10g,%d\n" % row)

    print("\n   saved %s  (%.2f GB, %d zones, %.0f s)"
          % (out_tec, os.path.getsize(out_tec) / 1e9, written, time.time() - t_start))
    print("   every zone: I=%d J=%d K=%d, X Y Z = global column, row, layer"
          % (ncol, nrow, nk))
    print("   zone index: %s.zones.csv" % os.path.basename(out_tec))
    if no_vel:
        print("   %d of %d zones without velocities, as in the partition files"
              % (len(no_vel), written))
    if problems:
        print("\n   %d incomplete block(s):" % len(problems))
        for msg in problems:
            print("      " + msg)
    else:
        print("   every block complete%s" % (" (%d read line by line)" % slow if slow else ""))


def _to_nan(a):
    a = a.astype(np.float32)
    for nd in NODATA_VALUES:
        a[np.abs(a - nd) < NODATA_TOL] = np.nan
    if NODATA_BELOW is not None:
        a[a <= NODATA_BELOW] = np.nan
    return a


def _parse(txt, cols):
    if b"D" in txt or b"d" in txt:
        txt = txt.replace(b"D", b"E").replace(b"d", b"e")
    if pd is not None:
        return pd.read_csv(io.BytesIO(txt), header=None, usecols=cols, engine="c",
                           dtype=np.float64, na_filter=False)[cols].to_numpy()
    return np.loadtxt(io.StringIO(txt.decode("ascii")), delimiter=",", usecols=cols,
                      ndmin=2)


def check(out_tec=OUT_TEC, npz=CHECK_NPZ, zones=CHECK_ZONES):
    idx_file = out_tec + ".zones.csv"
    if not (os.path.isfile(out_tec) and os.path.isfile(idx_file)):
        raise SystemExit("no %s with its .zones.csv -- run merge first" % out_tec)
    with open(idx_file) as f:
        next(f)
        index = [(int(a), float(c), int(o)) for a, _b, c, o in
                 (r.strip().split(",") for r in f if r.strip())]
    picks = []
    for z in zones:
        i = z - 1 if z > 0 else len(index) + z
        if 0 <= i < len(index) and index[i] not in picks:
            picks.append(index[i])

    d = np.load(npz)
    t_npz, layers = d["times_years"], d["layers"].astype(int)
    print("   NPZ %s\n      made from %s, %d times, layers %d..%d"
          % (npz, d["source"] if "source" in d.files else "?", len(t_npz),
             layers.min(), layers.max()))
    if "flip_rows" in d.files and bool(d["flip_rows"]):
        print("   ! the NPZ was made with FLIP_ROWS=True; this check expects rows unflipped")

    zone_vals = {}
    with open(out_tec, "rb") as f:
        names = None
        while names is None:
            line = f.readline()
            if not line:
                raise SystemExit("no VARIABLES line in %s" % out_tec)
            if b"VARIABLES" in line.upper():
                names = read_names(line)
        for z, t_idx, off in picks:
            f.seek(off)
            line = f.readline()
            while line and line.lstrip()[:4].upper() != b"ZONE":
                line = f.readline()
            t, ni, nj, nk, shared = parse_zone_header(line)
            if {0, 1, 2} & set(shared):
                print("   zone %d shares X, Y, Z (D=...), not checked" % z)
                continue
            n = ni * nj * nk
            txt = b"".join(islice(f, n))
            w = txt[:txt.find(b"\n")].count(b",") + 1
            vs = [v for v in names[3:w] if v in d.files]
            a = _parse(txt, [0, 1, 2] + [names.index(v) for v in vs])
            del txt
            c = np.arange(n)
            xyz_ok = (len(a) == n and np.array_equal(a[:, 0], c % ni + 1)
                      and np.array_equal(a[:, 1], c // ni % nj + 1)
                      and np.array_equal(a[:, 2], c // (ni * nj) + 1))
            print("   zone %d (%.3f yr): %d lines, X Y Z %s"
                  % (z, t, len(a), "run 1..%d, 1..%d, 1..%d in F=POINT order"
                     % (ni, nj, nk) if xyz_ok else "NOT in the expected order !!"))
            if len(a) == n:
                zone_vals[z] = (t, ni, nj, nk, {v: _to_nan(a[:, 3 + q].reshape(nk, nj, ni))
                                                for q, v in enumerate(vs)})

    n_same = n_diff = 0
    for v in ("HEAD", "CONC", "VX", "VY", "VZ"):
        if v not in d.files:
            continue
        ref = d[v]
        for z, (t, ni, nj, nk, vals) in zone_vals.items():
            it = int(np.argmin(np.abs(t_npz - t)))
            if v not in vals or abs(t_npz[it] - t) > 1e-6:
                print("      zone %d %-4s: not in the NPZ, skipped" % (z, v))
                continue
            if ref.shape[2:] != (nj, ni) or layers.max() > nk:
                print("      zone %d %-4s: the NPZ grid %s differs from %d x %d x %d"
                      % (z, v, ref.shape[1:], nk, nj, ni))
                n_diff += 1
                continue
            mine, theirs = vals[v][layers - 1], ref[it]
            if np.array_equal(mine, theirs, equal_nan=True):
                print("      zone %d %-4s: identical to NPZ time %d" % (z, v, it + 1))
                n_same += 1
                continue
            n_diff += 1
            both = np.isfinite(mine) & np.isfinite(theirs)
            diff = np.abs(mine[both].astype(np.float64) - theirs[both])
            print("      zone %d %-4s: !! %d cells differ (max %.3g), %d NaN on one side only"
                  % (z, v, int((diff > 0).sum()), diff.max() if diff.size else 0,
                     int((np.isnan(mine) != np.isnan(theirs)).sum())))
        del ref
    print("\n   %d comparison(s) identical, %d different" % (n_same, n_diff))


if __name__ == "__main__":
    what = (sys.argv[1] if len(sys.argv) > 1 else MODE).strip().lower()
    print("   MODE: %s" % what)
    if what.startswith("s"):
        scan()
        print('\n   looks right?  ->  MODE = "merge" (or MAX_ZONES = 2 first for a trial)\n')
    elif what.startswith("m"):
        merge()
    elif what.startswith("c"):
        check()
    else:
        print(__doc__)