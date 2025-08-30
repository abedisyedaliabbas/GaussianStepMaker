#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gaussian Step Maker — GUI
- Generate SINGLE step or FULL (1–7) with flexible geometry handling
- Inline vs linked (%%oldchk + geom=check) control for full mode
- Charge/Multiplicity override
- Vacuum or SCRF (SMD/PCM/IEFPCM/CPCM)
- Scheduler script: PBS / SLURM / local
"""

from __future__ import annotations
import re, glob, os
from pathlib import Path
from typing import List, Sequence, Tuple, Dict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ============================= core generation logic =============================

def natural_key(s: str):
    parts = re.split(r'(\d+)', s.lower())
    return tuple(int(p) if p.isdigit() else p for p in parts)

def read_lines(p: Path) -> List[str]:
    return p.read_text(errors="ignore").splitlines()

def write_lines(p: Path, lines: Sequence[str]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="\n") as f:
        for L in lines:
            f.write(str(L).rstrip("\r\n") + "\n")

def find_geoms(pattern: str) -> List[Path]:
    path = Path(pattern)
    if path.exists() and path.is_dir():
        files = sorted(path.glob("*.com"), key=lambda x: natural_key(x.name))
    else:
        matches = glob.glob(pattern)
        files = sorted([Path(m) for m in matches if Path(m).is_file() and m.lower().endswith(".com")],
                       key=lambda x: natural_key(x.name))
    return files

def extract_cm_coords(lines: List[str]) -> Tuple[str, List[str]]:
    empties = [i for i, L in enumerate(lines) if not L.strip()]
    coords = lines[empties[1]+1:] if len(empties) >= 2 else lines[:]
    atom_pat = re.compile(r"^[A-Za-z]{1,2}\s+[-\d]")
    cm = "0 1"
    for i, L in enumerate(coords):
        t = L.strip()
        if not t: continue
        if not atom_pat.match(t):
            if re.match(r"^-?\d+\s+-?\d+$", t):
                cm = t
                coords = coords[i+1:]
            break
    while coords and not coords[-1].strip():
        coords.pop()
    return cm, coords

def build_scrf(model: str, solvent: str, tail: str = "") -> str:
    if model.lower() in ("", "none", "vac", "vacuum"): return ""
    parts = [model]
    if solvent: parts.append(f"solvent={solvent}")
    if tail:    parts.append(tail)
    return ", ".join(parts)

def route_line(method: str, basis: str, td: str = "", scrf: str = "", extras: str = "") -> str:
    td_part   = f" TD=({td})" if td else ""
    scrf_part = f" SCRF=({scrf})" if scrf else ""
    base = f"# {method}/{basis}{td_part}{scrf_part}"
    if extras: base += f" {extras}"
    return re.sub(r"\s+", " ", base).strip()

def cm_override(parsed_cm: str, charge, mult) -> str:
    if charge is None or mult is None or charge=="" or mult=="":
        return parsed_cm
    return f"{int(charge)} {int(mult)}"

def make_com_inline(job: str, nproc: int, mem: str, route: str, title: str, cm: str, coords: List[str]) -> List[str]:
    return [f"%nprocshared={nproc}", f"%mem={mem}", f"%chk={job}.chk",
            route, "", title, "", cm, *coords, "", ""]

def make_com_linked(job: str, nproc: int, mem: str, oldchk: str, route: str, title: str, cm: str) -> List[str]:
    # linked uses oldchk + geom=check; only include charge/mult line
    return [f"%nprocshared={nproc}", f"%mem={mem}", f"%oldchk={oldchk}", f"%chk={job}.chk",
            route, "", title, "", cm, "", ""]

def pbs_script(job: str, queue: str, nproc: int, mem: str, wall: str, proj: str, gauss_bin: str) -> List[str]:
    return [
        "#!/bin/bash",
        f"#PBS -q {queue}",
        f"#PBS -N {job}",
        f"#PBS -l select=1:ncpus={nproc}:mpiprocs={nproc}:mem={mem}",
        (f"#PBS -l walltime={wall}" if wall else "").strip(),
        (f"#PBS -P {proj}" if proj else "").strip(),
        f"#PBS -o {job}.o", f"#PBS -e {job}.e",
        "cd $PBS_O_WORKDIR",
        f"{gauss_bin} < {job}.com > {job}.log",
    ]

def slurm_script(job: str, queue: str, nproc: int, mem: str, wall: str, acct: str, gauss_bin: str) -> List[str]:
    return [
        "#!/bin/bash",
        f"#SBATCH -J {job}",
        f"#SBATCH -p {queue}",
        "#SBATCH -N 1",
        f"#SBATCH --ntasks={nproc}",
        f"#SBATCH --mem={mem}",
        (f"#SBATCH -t {wall}" if wall else "").strip(),
        (f"#SBATCH -A {acct}" if acct else "").strip(),
        f"#SBATCH -o {job}.out", f"#SBATCH -e {job}.err",
        f"{gauss_bin} < {job}.com > {job}.log",
    ]

def local_script(job: str, gauss_bin: str) -> List[str]:
    return ["#!/bin/bash", f"{gauss_bin} < {job}.com > {job}.log &"]

def write_sh(job: str, sched: str, queue: str, nproc: int, mem: str, wall: str, proj: str, acct: str, gauss_bin: str):
    if sched == "PBS":   return pbs_script(job, queue, nproc, mem, wall, proj, gauss_bin)
    if sched == "SLURM": return slurm_script(job, queue, nproc, mem, wall, acct, gauss_bin)
    return local_script(job, gauss_bin)

def routes(FUNCTIONAL, BASIS, TD_BLOCK, POP_KW, DISP_KW, SCRF, SCRF_CLR):
    def r1(): return route_line(FUNCTIONAL, BASIS, scrf=SCRF, extras=f"Opt Freq{POP_KW}{DISP_KW}")
    def r2(): return route_line(FUNCTIONAL, BASIS, td=TD_BLOCK, scrf=SCRF, extras=f"{POP_KW}{DISP_KW}")
    def r3(): return route_line(FUNCTIONAL, BASIS, td=TD_BLOCK, scrf=SCRF_CLR, extras=f"{POP_KW}{DISP_KW}")
    def r4(): return route_line(FUNCTIONAL, BASIS, td=TD_BLOCK, scrf=SCRF, extras=f"Opt=CalcFC Freq{DISP_KW}")
    def r5(): return route_line(FUNCTIONAL, BASIS, scrf=SCRF, extras=f"density{POP_KW}{DISP_KW}")
    def r6(): return route_line(FUNCTIONAL, BASIS, td=TD_BLOCK, scrf=SCRF_CLR, extras=f"{DISP_KW}")
    def r7(): return route_line(FUNCTIONAL, BASIS, td=TD_BLOCK, scrf=SCRF, extras=f"{POP_KW}{DISP_KW}")
    return {1:r1,2:r2,3:r3,4:r4,5:r5,6:r6,7:r7}

def generate_single(base, cm, coords, cfg) -> List[str]:
    step = int(cfg["STEP"])
    prefix = f"{step:02d}"
    solv_tag = "vac" if cfg["SOLVENT_MODEL"].lower() in ("none","vac","vacuum") else cfg["SOLVENT_NAME"].lower()
    job = f"{prefix}{base}_{cfg['FUNCTIONAL']}_{cfg['BASIS']}_{solv_tag}"
    TD_BLOCK = f"NStates={cfg['TD_NSTATES']}, Root={cfg['TD_ROOT']}"
    POP_KW   = " pop=(full,orbitals=2,threshorbitals=1)" if cfg["POP_FULL"] else ""
    DISP_KW  = " EmpiricalDispersion=GD3BJ" if cfg["DISPERSION"] else ""
    SCRF     = build_scrf(cfg["SOLVENT_MODEL"], cfg["SOLVENT_NAME"])
    SCRF_CLR = build_scrf(cfg["SOLVENT_MODEL"], cfg["SOLVENT_NAME"], "CorrectedLR")
    R = routes(cfg["FUNCTIONAL"], cfg["BASIS"], TD_BLOCK, POP_KW, DISP_KW, SCRF, SCRF_CLR)
    title = {
        1: f"Step1 GS Opt {cfg['FUNCTIONAL']}/{cfg['BASIS']}",
        2: f"Step2 Abs {cfg['FUNCTIONAL']}/{cfg['BASIS']}",
        3: f"Step3 Abs cLR {cfg['FUNCTIONAL']}/{cfg['BASIS']}",
        4: f"Step4 ES Opt {cfg['FUNCTIONAL']}/{cfg['BASIS']}",
        5: f"Step5 Density {cfg['FUNCTIONAL']}/{cfg['BASIS']}",
        6: f"Step6 ES cLR {cfg['FUNCTIONAL']}/{cfg['BASIS']}",
        7: f"Step7 De-excitation {cfg['FUNCTIONAL']}/{cfg['BASIS']}",
    }[step]
    com = make_com_inline(job, cfg["NPROC"], cfg["MEM"], R[step](), title, cm, coords)
    write_lines(cfg["OUT_DIR"] / f"{job}.com", com)
    sh = write_sh(job, cfg["SCHED"], cfg["QUEUE"], cfg["NPROC"], cfg["MEM"], cfg["WALLTIME"], cfg["PROJECT"], cfg["ACCOUNT"], cfg["GAUSS_BIN"])
    write_lines(cfg["OUT_DIR"] / f"{job}.sh", sh)
    return [job]

def generate_full(base, cm_in, coords_in, cfg) -> List[str]:
    jobs = []
    solv_tag = "vac" if cfg["SOLVENT_MODEL"].lower() in ("none","vac","vacuum") else cfg["SOLVENT_NAME"].lower()
    J = lambda k: f"{k:02d}{base}_{cfg['FUNCTIONAL']}_{cfg['BASIS']}_{solv_tag}"
    j01,j02,j03,j04,j05,j06,j07 = (J(i) for i in range(1,8))

    TD_BLOCK = f"NStates={cfg['TD_NSTATES']}, Root={cfg['TD_ROOT']}"
    POP_KW   = " pop=(full,orbitals=2,threshorbitals=1)" if cfg["POP_FULL"] else ""
    DISP_KW  = " EmpiricalDispersion=GD3BJ" if cfg["DISPERSION"] else ""
    SCRF     = build_scrf(cfg["SOLVENT_MODEL"], cfg["SOLVENT_NAME"])
    SCRF_CLR = build_scrf(cfg["SOLVENT_MODEL"], cfg["SOLVENT_NAME"], "CorrectedLR")
    R = routes(cfg["FUNCTIONAL"], cfg["BASIS"], TD_BLOCK, POP_KW, DISP_KW, SCRF, SCRF_CLR)

    inline_set = set(cfg["INLINE_STEPS"])  # steps to inline

    def _write(job, lines):
        write_lines(cfg["OUT_DIR"] / f"{job}.com", lines)
        sh = write_sh(job, cfg["SCHED"], cfg["QUEUE"], cfg["NPROC"], cfg["MEM"], cfg["WALLTIME"], cfg["PROJECT"], cfg["ACCOUNT"], cfg["GAUSS_BIN"])
        write_lines(cfg["OUT_DIR"] / f"{job}.sh", sh)
        jobs.append(job)

    # Step 1 (always inline)
    com01 = make_com_inline(j01, cfg["NPROC"], cfg["MEM"], R[1](), f"Step1 GS Opt {cfg['FUNCTIONAL']}/{cfg['BASIS']}", cm_in, coords_in)
    _write(j01, com01)

    # Steps 2–3 (read 01 if linked)
    if 2 in inline_set:
        com02 = make_com_inline(j02, cfg["NPROC"], cfg["MEM"], R[2](), f"Step2 Abs {cfg['FUNCTIONAL']}/{cfg['BASIS']}", cm_in, coords_in)
    else:
        com02 = make_com_linked(j02, cfg["NPROC"], cfg["MEM"], f"{j01}.chk", R[2]()+" geom=check guess=read", f"Step2 Abs {cfg['FUNCTIONAL']}/{cfg['BASIS']}", cm_in)
    _write(j02, com02)

    if 3 in inline_set:
        com03 = make_com_inline(j03, cfg["NPROC"], cfg["MEM"], R[3](), f"Step3 Abs cLR {cfg['FUNCTIONAL']}/{cfg['BASIS']}", cm_in, coords_in)
    else:
        com03 = make_com_linked(j03, cfg["NPROC"], cfg["MEM"], f"{j01}.chk", R[3]()+" geom=check guess=read", f"Step3 Abs cLR {cfg['FUNCTIONAL']}/{cfg['BASIS']}", cm_in)
    _write(j03, com03)

    # Step 4 (read 01 if linked)
    if 4 in inline_set:
        com04 = make_com_inline(j04, cfg["NPROC"], cfg["MEM"], R[4](), f"Step4 ES Opt {cfg['FUNCTIONAL']}/{cfg['BASIS']}", cm_in, coords_in)
    else:
        com04 = make_com_linked(j04, cfg["NPROC"], cfg["MEM"], f"{j01}.chk", R[4]()+" geom=check guess=read", f"Step4 ES Opt {cfg['FUNCTIONAL']}/{cfg['BASIS']}", cm_in)
    _write(j04, com04)

    # Steps 5–7 (read 04 if linked; inline uses coords from chosen source — we only have input coords at build time)
    for k, title in [(5,"Step5 Density"), (6,"Step6 ES cLR"), (7,"Step7 De-excitation")]:
        if k in inline_set:
            # source setting is advisory; we copy input coords as a valid starting structure
            com = make_com_inline({5:j05,6:j06,7:j07}[k], cfg["NPROC"], cfg["MEM"], R[k](), f"{title} {cfg['FUNCTIONAL']}/{cfg['BASIS']}", cm_in, coords_in)
        else:
            com = make_com_linked({5:j05,6:j06,7:j07}[k], cfg["NPROC"], cfg["MEM"], f"{j04}.chk", R[k]()+" geom=check guess=read", f"{title} {cfg['FUNCTIONAL']}/{cfg['BASIS']}", cm_in)
        _write({5:j05,6:j06,7:j07}[k], com)

    return jobs

# ============================= GUI =============================

class App(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=10)
        self.master.title("Gaussian Step Maker")
        self.grid(sticky="nsew")
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)

        # ---------- Variables ----------
        self.mode = tk.StringVar(value="FULL")
        self.step = tk.IntVar(value=4)
        self.inputs = tk.StringVar(value="./*.com")
        self.outdir = tk.StringVar(value="Jobs")

        self.functional = tk.StringVar(value="m062x")
        self.basis = tk.StringVar(value="def2tzvp")

        self.solv_model = tk.StringVar(value="SMD")  # none/SMD/PCM/IEFPCM/CPCM
        self.solv_name  = tk.StringVar(value="DMSO")

        self.td_n = tk.IntVar(value=3)
        self.td_root = tk.IntVar(value=1)

        self.pop_full = tk.BooleanVar(value=False)
        self.disp = tk.BooleanVar(value=False)

        self.nproc = tk.IntVar(value=32)
        self.mem   = tk.StringVar(value="64GB")
        self.gauss_bin = tk.StringVar(value="g16")

        self.sched = tk.StringVar(value="PBS")  # PBS/SLURM/local
        self.queue = tk.StringVar(value="normal")
        self.wall  = tk.StringVar(value="24:00:00")
        self.project = tk.StringVar(value="15002108")
        self.account = tk.StringVar(value="")

        # FULL options
        self.inline_2 = tk.BooleanVar(value=False)
        self.inline_3 = tk.BooleanVar(value=False)
        self.inline_4 = tk.BooleanVar(value=False)
        self.inline_5 = tk.BooleanVar(value=False)
        self.inline_6 = tk.BooleanVar(value=False)
        self.inline_7 = tk.BooleanVar(value=False)
        self.inline_src_5to7 = tk.IntVar(value=4)  # 1 or 4

        # CM override
        self.charge = tk.StringVar(value="")
        self.mult   = tk.StringVar(value="")

        # ---------- Layout ----------
        self._build_ui()
        self._wire_events()
        self._update_enable_states()

    def _build_ui(self):
        # Top row: Mode & Step
        mframe = ttk.LabelFrame(self, text="Mode")
        mframe.grid(row=0, column=0, sticky="ew", pady=(0,8))
        for i in range(4): mframe.columnconfigure(i, weight=1)

        ttk.Radiobutton(mframe, text="Full (1–7)", variable=self.mode, value="FULL", command=self._update_enable_states).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(mframe, text="Single", variable=self.mode, value="SINGLE", command=self._update_enable_states).grid(row=0, column=1, sticky="w")
        ttk.Label(mframe, text="Step:").grid(row=0, column=2, sticky="e")
        ttk.Spinbox(mframe, from_=1, to=7, textvariable=self.step, width=5).grid(row=0, column=3, sticky="w")

        # Paths
        pframe = ttk.LabelFrame(self, text="Paths")
        pframe.grid(row=1, column=0, sticky="ew", pady=(0,8))
        pframe.columnconfigure(1, weight=1)

        ttk.Label(pframe, text="Input (*.com glob or folder):").grid(row=0, column=0, sticky="w")
        ttk.Entry(pframe, textvariable=self.inputs).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(pframe, text="Browse…", command=self._browse_input).grid(row=0, column=2)

        ttk.Label(pframe, text="Output folder:").grid(row=1, column=0, sticky="w")
        ttk.Entry(pframe, textvariable=self.outdir).grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Button(pframe, text="Choose…", command=self._browse_outdir).grid(row=1, column=2)

        # Method / basis / TD / pop / disp
        m2 = ttk.LabelFrame(self, text="Method / TD-DFT / Extras")
        m2.grid(row=2, column=0, sticky="ew", pady=(0,8))
        for i in range(8): m2.columnconfigure(i, weight=1)

        ttk.Label(m2, text="Functional").grid(row=0, column=0, sticky="w")
        ttk.Entry(m2, textvariable=self.functional, width=12).grid(row=0, column=1, sticky="w")
        ttk.Label(m2, text="Basis").grid(row=0, column=2, sticky="w")
        ttk.Entry(m2, textvariable=self.basis, width=12).grid(row=0, column=3, sticky="w")

        ttk.Label(m2, text="TD NStates").grid(row=0, column=4, sticky="e")
        ttk.Spinbox(m2, from_=1, to=50, textvariable=self.td_n, width=6).grid(row=0, column=5, sticky="w")
        ttk.Label(m2, text="TD Root").grid(row=0, column=6, sticky="e")
        ttk.Spinbox(m2, from_=1, to=50, textvariable=self.td_root, width=6).grid(row=0, column=7, sticky="w")

        ttk.Checkbutton(m2, text="pop=(full,orbitals=2,threshorbitals=1)", variable=self.pop_full).grid(row=1, column=0, columnspan=4, sticky="w", pady=(4,0))
        ttk.Checkbutton(m2, text="EmpiricalDispersion=GD3BJ", variable=self.disp).grid(row=1, column=4, columnspan=4, sticky="w", pady=(4,0))

        # Solvent
        sframe = ttk.LabelFrame(self, text="Environment")
        sframe.grid(row=3, column=0, sticky="ew", pady=(0,8))
        for i in range(6): sframe.columnconfigure(i, weight=1)

        ttk.Label(sframe, text="SCRF model").grid(row=0, column=0, sticky="w")
        ttk.Combobox(sframe, values=["none","SMD","PCM","IEFPCM","CPCM"], textvariable=self.solv_model, state="readonly", width=10).grid(row=0, column=1, sticky="w")
        ttk.Label(sframe, text="Solvent name").grid(row=0, column=2, sticky="e")
        self.solv_entry = ttk.Entry(sframe, textvariable=self.solv_name, width=12)
        self.solv_entry.grid(row=0, column=3, sticky="w")

        # Resources / scheduler
        rframe = ttk.LabelFrame(self, text="Resources / Scheduler")
        rframe.grid(row=4, column=0, sticky="ew", pady=(0,8))
        for i in range(10): rframe.columnconfigure(i, weight=1)

        ttk.Label(rframe, text="nproc").grid(row=0, column=0, sticky="e")
        ttk.Spinbox(rframe, from_=1, to=256, textvariable=self.nproc, width=6).grid(row=0, column=1, sticky="w")
        ttk.Label(rframe, text="mem").grid(row=0, column=2, sticky="e")
        ttk.Entry(rframe, textvariable=self.mem, width=8).grid(row=0, column=3, sticky="w")
        ttk.Label(rframe, text="Gaussian bin").grid(row=0, column=4, sticky="e")
        ttk.Entry(rframe, textvariable=self.gauss_bin, width=8).grid(row=0, column=5, sticky="w")

        ttk.Label(rframe, text="Scheduler").grid(row=1, column=0, sticky="e")
        ttk.Combobox(rframe, values=["PBS","SLURM","local"], textvariable=self.sched, state="readonly", width=8).grid(row=1, column=1, sticky="w")
        ttk.Label(rframe, text="Queue/Partition").grid(row=1, column=2, sticky="e")
        ttk.Entry(rframe, textvariable=self.queue, width=10).grid(row=1, column=3, sticky="w")
        ttk.Label(rframe, text="Walltime").grid(row=1, column=4, sticky="e")
        ttk.Entry(rframe, textvariable=self.wall, width=10).grid(row=1, column=5, sticky="w")
        ttk.Label(rframe, text="PBS Project").grid(row=1, column=6, sticky="e")
        ttk.Entry(rframe, textvariable=self.project, width=10).grid(row=1, column=7, sticky="w")
        ttk.Label(rframe, text="SLURM Account").grid(row=1, column=8, sticky="e")
        ttk.Entry(rframe, textvariable=self.account, width=10).grid(row=1, column=9, sticky="w")

        # Full-mode geometry controls
        gframe = ttk.LabelFrame(self, text="FULL mode: inline geometry per step")
        gframe.grid(row=5, column=0, sticky="ew", pady=(0,8))
        for i in range(10): gframe.columnconfigure(i, weight=1)

        self.cb2 = ttk.Checkbutton(gframe, text="Step 2", variable=self.inline_2)
        self.cb3 = ttk.Checkbutton(gframe, text="Step 3", variable=self.inline_3)
        self.cb4 = ttk.Checkbutton(gframe, text="Step 4", variable=self.inline_4)
        self.cb5 = ttk.Checkbutton(gframe, text="Step 5", variable=self.inline_5)
        self.cb6 = ttk.Checkbutton(gframe, text="Step 6", variable=self.inline_6)
        self.cb7 = ttk.Checkbutton(gframe, text="Step 7", variable=self.inline_7)
        self.cb2.grid(row=0, column=0, sticky="w"); self.cb3.grid(row=0, column=1, sticky="w")
        self.cb4.grid(row=0, column=2, sticky="w"); self.cb5.grid(row=0, column=3, sticky="w")
        self.cb6.grid(row=0, column=4, sticky="w"); self.cb7.grid(row=0, column=5, sticky="w")
        ttk.Label(gframe, text="(else steps link via %oldchk + geom=check)").grid(row=0, column=6, columnspan=4, sticky="w")

        ttk.Label(gframe, text="Inline source for steps 5–7:").grid(row=1, column=0, sticky="e", pady=(6,0))
        ttk.Combobox(gframe, values=[1,4], textvariable=self.inline_src_5to7, state="readonly", width=4).grid(row=1, column=1, sticky="w", pady=(6,0))
        ttk.Label(gframe, text="(we use input coords as starting structure at build time)").grid(row=1, column=2, columnspan=8, sticky="w", pady=(6,0))

        # Charge/multiplicity override
        cmf = ttk.LabelFrame(self, text="Charge / Multiplicity override (optional)")
        cmf.grid(row=6, column=0, sticky="ew", pady=(0,8))
        for i in range(6): cmf.columnconfigure(i, weight=1)

        ttk.Label(cmf, text="Charge").grid(row=0, column=0, sticky="e")
        ttk.Entry(cmf, textvariable=self.charge, width=6).grid(row=0, column=1, sticky="w")
        ttk.Label(cmf, text="Multiplicity").grid(row=0, column=2, sticky="e")
        ttk.Entry(cmf, textvariable=self.mult, width=6).grid(row=0, column=3, sticky="w")
        ttk.Label(cmf, text="(leave empty to use value from input .com)").grid(row=0, column=4, columnspan=2, sticky="w")

        # Action buttons
        bframe = ttk.Frame(self)
        bframe.grid(row=7, column=0, sticky="ew")
        bframe.columnconfigure(0, weight=1)
        ttk.Button(bframe, text="Generate", command=self._on_generate).grid(row=0, column=0, sticky="ew")

    def _wire_events(self):
        self.solv_model.trace_add("write", lambda *_: self._update_enable_states())
        self.mode.trace_add("write", lambda *_: self._update_enable_states())
        self.sched.trace_add("write", lambda *_: self._update_enable_states())

    def _update_enable_states(self):
        # enable/disable step spin for SINGLE mode
        single = (self.mode.get() == "SINGLE")
        state_step = "normal" if single else "disabled"
        for w in self.winfo_children():
            # find the spinbox in Mode frame
            pass
        # simpler: directly set the widget by looking up through children
        # Find spinbox:
        # (we created it earlier, but not kept a handle — locate by class)
        for child in self.winfo_children():
            for c2 in child.winfo_children():
                if isinstance(c2, ttk.Spinbox) and str(c2.cget("textvariable")) == str(self.step):
                    c2.configure(state=state_step)

        # full-mode inline controls
        for cb in [self.cb2, self.cb3, self.cb4, self.cb5, self.cb6, self.cb7]:
            cb.configure(state="normal" if not single else "disabled")

        # solvent name enabled only when model != none
        if self.solv_model.get().lower() in ("none","vac","vacuum"):
            self.solv_entry.configure(state="disabled")
        else:
            self.solv_entry.configure(state="normal")

        # scheduler-specific fields
        is_pbs   = self.sched.get() == "PBS"
        is_slurm = self.sched.get() == "SLURM"
        # Enable/disable project/account entries
        # Find entries by their textvariable names:
        def set_entry_state(var, enabled: bool):
            for child in self.winfo_children():
                for c2 in child.winfo_children():
                    for c3 in c2.winfo_children():
                        try:
                            if str(c3.cget("textvariable")) == str(var):
                                c3.configure(state="normal" if enabled else "disabled")
                        except Exception:
                            pass
        set_entry_state(self.project, is_pbs)
        set_entry_state(self.account, is_slurm)

    def _browse_input(self):
        # allow choosing a folder; else the user can type a glob
        d = filedialog.askdirectory(title="Choose folder containing .com")
        if d:
            self.inputs.set(d)

    def _browse_outdir(self):
        d = filedialog.askdirectory(title="Choose output folder")
        if d:
            self.outdir.set(d)

    def _collect_cfg(self) -> Dict:
        # sanitize OUT_DIR
        outdir = Path(self.outdir.get() or "Jobs")
        # try cast charge/mult
        ch = self.charge.get().strip()
        mu = self.mult.get().strip()
        ch_val = int(ch) if ch != "" else None
        mu_val = int(mu) if mu != "" else None

        return {
            "MODE": self.mode.get(),
            "STEP": int(self.step.get()),
            "INPUTS": self.inputs.get(),
            "OUT_DIR": outdir,
            "FUNCTIONAL": self.functional.get().strip(),
            "BASIS": self.basis.get().strip(),
            "SOLVENT_MODEL": self.solv_model.get().strip(),
            "SOLVENT_NAME": self.solv_name.get().strip(),
            "TD_NSTATES": int(self.td_n.get()),
            "TD_ROOT": int(self.td_root.get()),
            "POP_FULL": bool(self.pop_full.get()),
            "DISPERSION": bool(self.disp.get()),
            "NPROC": int(self.nproc.get()),
            "MEM": self.mem.get().strip(),
            "SCHED": self.sched.get().strip(),
            "QUEUE": self.queue.get().strip(),
            "WALLTIME": self.wall.get().strip(),
            "PROJECT": self.project.get().strip(),
            "ACCOUNT": self.account.get().strip(),
            "GAUSS_BIN": self.gauss_bin.get().strip(),
            "INLINE_STEPS": [k for k,bv in zip([2,3,4,5,6,7],[self.inline_2.get(), self.inline_3.get(), self.inline_4.get(), self.inline_5.get(), self.inline_6.get(), self.inline_7.get()]) if bv],
            "INLINE_SOURCE_5TO7": int(self.inline_src_5to7.get()),
            "CHARGE": ch_val,
            "MULT": mu_val,
        }

    def _on_generate(self):
        try:
            cfg = self._collect_cfg()
            files = find_geoms(cfg["INPUTS"])
            if not files:
                raise RuntimeError("No .com files found (check your folder/glob).")

            cfg["OUT_DIR"].mkdir(parents=True, exist_ok=True)

            submit_lines = []
            for p in files:
                base = p.stem
                parsed_cm, coords = extract_cm_coords(read_lines(p))
                cm_use = cm_override(parsed_cm, cfg["CHARGE"], cfg["MULT"])
                if cfg["MODE"] == "SINGLE":
                    jobs = generate_single(base, cm_use, coords, cfg)
                else:
                    jobs = generate_full(base, cm_use, coords, cfg)
                # submission helper
                for j in jobs:
                    if cfg["SCHED"] == "PBS":
                        submit_lines.append(f"qsub {j}.sh")
                    elif cfg["SCHED"] == "SLURM":
                        submit_lines.append(f"sbatch {j}.sh")
                    else:
                        submit_lines.append(f"bash {j}.sh")

            write_lines(cfg["OUT_DIR"] / "submit_all.sh", submit_lines)
            messagebox.showinfo("Done", f"Generated {'FULL (1–7)' if cfg['MODE']=='FULL' else f'Step {cfg['STEP']}'} for {len(files)} molecule(s).\nOutput: {cfg['OUT_DIR'].resolve()}")
        except Exception as e:
            messagebox.showerror("Error", str(e))


def main():
    root = tk.Tk()
    # nice-ish scaling for high DPI
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)  # type: ignore
    except Exception:
        pass
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
