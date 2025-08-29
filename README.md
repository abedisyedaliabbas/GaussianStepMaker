# GaussianStepMaker

A Python utility to automatically generate Gaussian input (`.com`) and submission (`.sh`) files for multi-step photochemistry workflows.

Supports:

- ✅ Ground-state optimizations  
- ✅ Excited-state optimizations  
- ✅ Absorption and emission TDDFT steps  
- ✅ Density calculations  
- ✅ cLR corrections  
- ✅ Full workflow (steps 1–7) in one go, with geometry chaining or inline options  
- ✅ PBS, SLURM, or local submission scripts  
- ✅ Custom charge/multiplicity overrides  

---

## 🚀 Features
- Generate **single-step** or **full 1–7 step workflows**.  
- Choose whether geometries are **linked via `%oldchk`** or **inlined directly**.  
- Flexible file naming with method, basis, and solvent tags.  
- Automatic `.sh` job submission scripts (PBS, SLURM, or local).  
- Configurable processors, memory, walltime, queue, etc.  
- Optional GUI for easy setup.

---

## 🔑 Entry Points

### 1. Command-line version (cluster-friendly)
Run in your terminal (edit the `CONFIG` section inside the script):

```bash
git clone https://github.com/abedisyedaliabbas/GaussianStepMaker.git
python GaussianStepMaker.py


build a standalone app (Windows)
python -m PyInstaller --onefile --windowed --name GaussianStepMaker GaussianStepMaker_GUI.py
