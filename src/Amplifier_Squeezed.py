# ---- LIMIT THREADS BEFORE importing numpy/scipy/matplotlib ----
from __future__ import annotations

import os

THREADS = os.environ.get("QB_THREADS", "1")
os.environ["OMP_NUM_THREADS"] = THREADS
os.environ["OPENBLAS_NUM_THREADS"] = THREADS
os.environ["MKL_NUM_THREADS"] = THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = THREADS
os.environ["NUMEXPR_NUM_THREADS"] = THREADS
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["OMP_DYNAMIC"] = "FALSE"

# Non-interactive backend for clusters.
import matplotlib as mpl
mpl.use("Agg")

import dataclasses
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.special import gammaln

from pathlib import Path
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

"""
Quantum-battery-powered nondegenerate parametric amplification
Option B: cascaded output-collector temporal-mode calculation

This is the rigorous output-filter/collector version of the open-system calculation.
Instead of reconstructing temporal modes from two-time output correlations, we append
auxiliary collector modes A and B downstream of the signal and idler output ports.
The collector modes absorb filtered temporal modes of the emitted fields.

Source modes:
    a = signal resonator mode
    b = idler resonator mode
    c = finite pump / quantum-battery mode

Collector modes:
    A = signal-output temporal-mode collector
    B = idler-output temporal-mode collector

Source Hamiltonian:
    H_I = i g ( c a^dag b^dag - c^dag a b )

Source master equation without collectors:
    d rho/dt = -i[H_I,rho]
              + kappa_a D[a] rho
              + kappa_b D[b] rho
              + kappa_c D[c] rho

Cascaded collector construction:
    L_s,a = sqrt(kappa_a) a,      L_A = sqrt(gamma_A) A
    L_s,b = sqrt(kappa_b) b,      L_B = sqrt(gamma_B) B

For the unidirectional cascade source -> collector, each channel is represented by
    L_tot = L_s + L_col,
    H_cas = (i/2) (L_s^dag L_col - L_col^dag L_s).

The final collector moments give a canonical temporal-mode output witness:
    N_A = <A^dag A>,
    N_B = <B^dag B>,
    M_AB = <A B>,
    I_min = N_A + N_B + 1 - 2 |M_AB|,
    V_int = 2 |M_AB|/(N_A + N_B + 1),
    C_amp = |M_AB|/sqrt(n_pair(n_pair+1)), n_pair=(N_A+N_B)/2.

This is more rigorous than the time-integrated equal-time proxy because A and B are
actual canonical quantum modes. The price is a much larger Hilbert space.

The script saves PDF only.

Recommended cluster workflow:
    python qb_paramp_optionB_cascaded_collectors.py --mode basic
    python qb_paramp_optionB_cascaded_collectors.py --mode threshold
    python qb_paramp_optionB_cascaded_collectors.py --mode gain
    python qb_paramp_optionB_cascaded_collectors.py --mode nbar

For high-resolution production, increase Na,Nb,Nc,NAf,NBf,Nt after checking the
printed edge populations and the physical bounds I_min >= 0 and C_amp <= 1.
"""


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

@dataclass
class Params:
    # Mean pump/battery energy
    nbar: float = 5.0

    # Source-mode Hilbert-space cutoffs
    Na: int = 6
    Nb: int = 6
    Nc: int = 14

    # Collector-mode Hilbert-space cutoffs.
    # The collector stores emitted photons; choose enough states to cover N_A,N_B.
    NAf: int = 4
    NBf: int = 4

    # Nominal classical pump strength lambda = g sqrt(nbar)
    lam: float = 1.0

    # Source output/leakage rates
    kappa_a: float = 2.5
    kappa_b: float = 2.5
    kappa_c: float = 0.02

    # Collector/filter bandwidths.  gamma_f ~ lambda is a sensible first choice.
    # The collector implements an exponential temporal filter ending at the final time.
    gamma_A: float = 1.0
    gamma_B: float = 1.0

    # Partially dephased coherent pump:
    # rho_c = w |alpha><alpha| + (1-w) rho_phase_randomized, eta_partial=w^2.
    eta_partial: float = 0.25

    # Time grid in units tau=lambda t
    tau_max: float = 4.0
    Nt: int = 81

    # Solver tolerances
    nsteps: int = 50000
    atol: float = 1e-8
    rtol: float = 1e-7

    # Plot/output
    outdir: str = "qb_paramp_outputs_optionB"

    # Safety guard for exact density-matrix mesolve.  QuTiP must build a
    # Liouville-space object of dimension D^2; above this value, exact mesolve
    # can fail from sparse-index overflows even on very large-RAM machines.
    # Use --max_mesolve_dim 0 to disable the guard.
    max_mesolve_dim: int = 150000

    @property
    def g(self) -> float:
        return self.lam / np.sqrt(self.nbar)

    @property
    def rho_reduced(self) -> float:
        return 2.0 * self.lam / np.sqrt(self.kappa_a * self.kappa_b)

    @property
    def stiff_pump_gain_estimate(self) -> float:
        rho = self.rho_reduced
        if rho >= 1.0:
            return np.inf
        return ((1.0 + rho**2) / (1.0 - rho**2)) ** 2

    @property
    def hilbert_dim(self) -> int:
        return self.Na * self.Nb * self.Nc * self.NAf * self.NBf


# -----------------------------------------------------------------------------
# QuTiP solver-options compatibility
# -----------------------------------------------------------------------------

def make_mesolve_options(p: Params):
    """QuTiP 5 solver options.

    Important: this script uses density-matrix mesolve.  For large Hilbert spaces,
    the Liouvillian is enormous.  We therefore keep states out of memory and force
    sparse CSR operators below, so QuTiP does not try to build gigantic DIA-format
    superoperators.
    """
    return {
        "nsteps": p.nsteps,
        "atol": p.atol,
        "rtol": p.rtol,
        "store_states": False,
        "progress_bar": "",
    }


def to_csr_qobj(x: qt.Qobj) -> qt.Qobj:
    """Return a Qobj stored as CSR sparse data.

    QuTiP 5 may create identity/tensor products in DIA format.  For very large
    Liouville dimensions, DIA addition can request an impossible allocation such
    as a (D^2,D^2) Dia array.  Converting all Hamiltonians, collapse operators,
    states, and e_ops to CSR before mesolve avoids that failure mode.
    """
    try:
        return x.to("csr")
    except Exception:
        return x


# -----------------------------------------------------------------------------
# Pump/battery states
# -----------------------------------------------------------------------------

def poisson_probs(nmax: int, nbar: float) -> np.ndarray:
    probs = np.empty(nmax, dtype=float)
    probs[0] = np.exp(-nbar)
    for n in range(1, nmax):
        probs[n] = probs[n - 1] * nbar / n
    total = probs.sum()
    if total <= 0:
        raise ValueError("Poisson probabilities underflowed.")
    return probs / total


def coherent_dm(N: int, nbar: float) -> qt.Qobj:
    ket = qt.coherent(N, np.sqrt(nbar))
    return to_csr_qobj(qt.ket2dm(ket))


def phase_randomized_coherent_dm(N: int, nbar: float) -> qt.Qobj:
    probs = poisson_probs(N, nbar)
    return to_csr_qobj(qt.Qobj(np.diag(probs), dims=[[N], [N]]))


def fock_dm(N: int, n: int) -> qt.Qobj:
    if n >= N:
        raise ValueError(
            f"Fock index n={n} does not fit in cutoff N={N}. Increase Nc or lower nbar."
        )
    return to_csr_qobj(qt.ket2dm(qt.basis(N, n)))


def single_mode_stats(rho: qt.Qobj) -> dict[str, float | complex]:
    """Single-mode pump diagnostics for an already-truncated density matrix."""
    N = rho.shape[0]
    c = qt.destroy(N)
    n_op = c.dag() * c
    n2_op = n_op * n_op
    n_mean = float(np.real(qt.expect(n_op, rho)))
    n2_mean = float(np.real(qt.expect(n2_op, rho)))
    var_n = max(0.0, n2_mean - n_mean**2)
    c_mean = complex(qt.expect(c, rho))
    eta = 0.0 if n_mean <= 1e-14 else abs(c_mean) ** 2 / n_mean
    P_edge = qt.basis(N, N - 1) * qt.basis(N, N - 1).dag()
    edge = float(np.real(qt.expect(P_edge, rho)))
    return {
        "n": n_mean,
        "var": var_n,
        "eta": float(eta),
        "c_mean": c_mean,
        "edge": edge,
    }


def single_mode_eta(rho: qt.Qobj) -> float:
    return float(single_mode_stats(rho)["eta"])


def displaced_squeezed_dm(
    N: int,
    nbar: float,
    r: float,
    angle_kind: str = "amp",
    phi_alpha: float = 0.0,
) -> qt.Qobj:
    """Return D(alpha) S(zeta)|0><0| S^dag D^dag at fixed total energy.

    The fixed-energy convention is
        nbar = |alpha|^2 + sinh^2(r).

    For real alpha, angle_kind='amp' uses zeta phase 0, while
    angle_kind='phase' uses zeta phase pi.  This matches the convention used
    in the accompanying Supplementary Note: amplitude squeezing can reduce
    photon-number fluctuations, whereas phase squeezing increases them.
    """
    r = float(r)
    if r < -1e-14:
        raise ValueError("Squeezing parameter r must be non-negative.")
    squeezed_energy = float(np.sinh(r) ** 2)
    if squeezed_energy > nbar + 1e-12:
        raise ValueError(
            f"Requested squeezed state has sinh(r)^2={squeezed_energy:.6g} "
            f"> nbar={nbar:.6g}. Lower r or increase nbar."
        )
    alpha_abs = np.sqrt(max(float(nbar) - squeezed_energy, 0.0))
    alpha = alpha_abs * np.exp(1j * phi_alpha)

    key = angle_kind.lower().strip()
    if key in {"amp", "amplitude", "amplitude_squeezed"}:
        theta = 2.0 * phi_alpha
    elif key in {"phase", "phase_squeezed"}:
        theta = 2.0 * phi_alpha + np.pi
    else:
        raise ValueError(f"Unknown squeezed angle kind: {angle_kind}")

    zeta = r * np.exp(1j * theta)
    ket = qt.displace(N, alpha) * qt.squeeze(N, zeta) * qt.basis(N, 0)
    ket = ket.unit()
    return to_csr_qobj(qt.ket2dm(ket))


def squeezed_vacuum_dm(N: int, nbar: float, phi_squeeze: float = 0.0) -> qt.Qobj:
    """Squeezed vacuum with mean photon number nbar.

    This state has finite stored energy but <c>=0 and no adjacent-number
    coherence. It is useful as an analytic stress test, but may require a larger
    pump cutoff than the displaced squeezed states.
    """
    if nbar < 0:
        raise ValueError("nbar must be non-negative.")
    r = float(np.arcsinh(np.sqrt(nbar)))
    zeta = r * np.exp(1j * phi_squeeze)
    ket = qt.squeeze(N, zeta) * qt.basis(N, 0)
    ket = ket.unit()
    return to_csr_qobj(qt.ket2dm(ket))


def parse_squeezed_kind(kind: str) -> tuple[str, float] | None:
    """Parse strings such as 'amp_squeezed:0.5' or 'phase_squeezed:0.5'."""
    if ":" not in kind:
        return None
    prefix, value = kind.split(":", 1)
    prefix = prefix.strip().lower()
    if prefix in {"amp_squeezed", "amplitude_squeezed", "squeezed_amp"}:
        return "amp", float(value)
    if prefix in {"phase_squeezed", "squeezed_phase"}:
        return "phase", float(value)
    return None


def pump_state_dm(kind: str, p: Params) -> tuple[qt.Qobj, float]:
    squeezed = parse_squeezed_kind(kind)
    if kind == "coherent":
        rho = coherent_dm(p.Nc, p.nbar)
    elif kind == "phase_randomized":
        rho = phase_randomized_coherent_dm(p.Nc, p.nbar)
    elif kind == "partial":
        w = np.sqrt(p.eta_partial)
        rho_coh = coherent_dm(p.Nc, p.nbar)
        rho_deph = phase_randomized_coherent_dm(p.Nc, p.nbar)
        rho = w * rho_coh + (1.0 - w) * rho_deph
        rho = rho / rho.tr()
    elif kind == "fock":
        rho = fock_dm(p.Nc, int(round(p.nbar)))
    elif squeezed is not None:
        angle_kind, r = squeezed
        rho = displaced_squeezed_dm(p.Nc, p.nbar, r, angle_kind=angle_kind)
    elif kind == "squeezed_vacuum":
        rho = squeezed_vacuum_dm(p.Nc, p.nbar)
    else:
        raise ValueError(f"Unknown pump kind: {kind}")
    return rho, single_mode_eta(rho)


# -----------------------------------------------------------------------------
# Tensor-mode construction
# -----------------------------------------------------------------------------

def tensor_op(single_op: qt.Qobj, slot: int, dims: list[int]) -> qt.Qobj:
    ops = [to_csr_qobj(qt.qeye(d)) for d in dims]
    ops[slot] = to_csr_qobj(single_op)
    return to_csr_qobj(qt.tensor(*ops))


@dataclass
class Operators:
    a: qt.Qobj
    b: qt.Qobj
    c: qt.Qobj
    A: qt.Qobj
    B: qt.Qobj


def build_operators(p: Params) -> Operators:
    dims = [p.Na, p.Nb, p.Nc, p.NAf, p.NBf]
    a = tensor_op(qt.destroy(p.Na), 0, dims)
    b = tensor_op(qt.destroy(p.Nb), 1, dims)
    c = tensor_op(qt.destroy(p.Nc), 2, dims)
    A = tensor_op(qt.destroy(p.NAf), 3, dims)
    B = tensor_op(qt.destroy(p.NBf), 4, dims)
    return Operators(a=a, b=b, c=c, A=A, B=B)


def build_cascaded_model(p: Params) -> tuple[qt.Qobj, list[qt.Qobj], Operators]:
    op = build_operators(p)
    a, b, c, A, B = op.a, op.b, op.c, op.A, op.B

    H_source = 1j * p.g * (c * a.dag() * b.dag() - c.dag() * a * b)

    # Cascaded signal-output channel: source a -> collector A.
    Ls_a = np.sqrt(p.kappa_a) * a
    Lc_A = np.sqrt(p.gamma_A) * A
    H_cas_a = 0.5j * (Ls_a.dag() * Lc_A - Lc_A.dag() * Ls_a)
    Ltot_a = Ls_a + Lc_A

    # Cascaded idler-output channel: source b -> collector B.
    Ls_b = np.sqrt(p.kappa_b) * b
    Lc_B = np.sqrt(p.gamma_B) * B
    H_cas_b = 0.5j * (Ls_b.dag() * Lc_B - Lc_B.dag() * Ls_b)
    Ltot_b = Ls_b + Lc_B

    H = to_csr_qobj(H_source + H_cas_a + H_cas_b)

    c_ops: list[qt.Qobj] = [to_csr_qobj(Ltot_a), to_csr_qobj(Ltot_b)]
    if p.kappa_c > 0:
        c_ops.append(to_csr_qobj(np.sqrt(p.kappa_c) * c))

    return H, c_ops, op


def initial_state(kind: str, p: Params) -> tuple[qt.Qobj, float]:
    rho_a = to_csr_qobj(qt.ket2dm(qt.basis(p.Na, 0)))
    rho_b = to_csr_qobj(qt.ket2dm(qt.basis(p.Nb, 0)))
    rho_c, eta = pump_state_dm(kind, p)
    rho_A = to_csr_qobj(qt.ket2dm(qt.basis(p.NAf, 0)))
    rho_B = to_csr_qobj(qt.ket2dm(qt.basis(p.NBf, 0)))
    return to_csr_qobj(qt.tensor(rho_a, rho_b, rho_c, rho_A, rho_B)), eta


# -----------------------------------------------------------------------------
# Collector diagnostics
# -----------------------------------------------------------------------------

def temporal_certifiers_from_collectors(
    N_A: np.ndarray,
    N_B: np.ndarray,
    M_AB: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N_A = np.real(np.asarray(N_A, dtype=complex))
    N_B = np.real(np.asarray(N_B, dtype=complex))
    M_AB = np.asarray(M_AB, dtype=complex)

    denom = N_A + N_B + 1.0
    Imin = denom - 2.0 * np.abs(M_AB)
    Imax = denom + 2.0 * np.abs(M_AB)

    V = np.zeros_like(Imin, dtype=float)
    mask = denom > 1e-14
    V[mask] = 2.0 * np.abs(M_AB[mask]) / denom[mask]

    n_pair = 0.5 * (N_A + N_B)
    C = np.zeros_like(n_pair, dtype=float)
    mask_pair = n_pair > 1e-14
    C[mask_pair] = np.abs(M_AB[mask_pair]) / np.sqrt(
        n_pair[mask_pair] * (n_pair[mask_pair] + 1.0)
    )

    return V, Imin, Imax, C, n_pair


def physicality_report(label: str, Imin: np.ndarray, C: np.ndarray) -> None:
    min_I = float(np.nanmin(Imin))
    max_C = float(np.nanmax(C))
    print(f"    physicality check for {label}: min Imin={min_I:.6g}, max C={max_C:.6g}")
    if min_I < -1e-6:
        print("    WARNING: Imin < 0. Increase cutoffs or check cascade convention.")
    if max_C > 1.0 + 1e-5:
        print("    WARNING: C_amp > 1. Increase cutoffs or check cascade convention.")


@dataclass
class CollectorResult:
    label: str
    eta_in: float
    tlist: np.ndarray
    tau_grid: np.ndarray
    N_A: np.ndarray
    N_B: np.ndarray
    M_AB: np.ndarray
    V: np.ndarray
    Imin: np.ndarray
    Imax: np.ndarray
    C: np.ndarray
    n_pair: np.ndarray
    eta_c: np.ndarray
    source_n_a: np.ndarray
    source_n_b: np.ndarray
    source_n_c: np.ndarray
    edge_source_a: np.ndarray
    edge_source_b: np.ndarray
    edge_pump_c: np.ndarray
    edge_col_A: np.ndarray
    edge_col_B: np.ndarray
    phi: np.ndarray
    I_fringe_final: np.ndarray


def solve_collector_case(kind: str, p: Params) -> CollectorResult:
    if p.max_mesolve_dim and p.hilbert_dim > p.max_mesolve_dim:
        raise RuntimeError(
            f"Exact density-matrix mesolve is too large for this run: "
            f"D={p.hilbert_dim} exceeds max_mesolve_dim={p.max_mesolve_dim}.\n"
            f"For exact Option B, reduce collector/source cutoffs, e.g. "
            f"--NAf 4 --NBf 4, or raise the guard with --max_mesolve_dim 0 "
            f"only if you know your QuTiP build supports this Liouville size.\n"
            f"Current Liouville dimension is D^2={p.hilbert_dim**2:.3e}."
        )

    H, c_ops, op = build_cascaded_model(p)
    rho0, eta_in = initial_state(kind, p)

    tmax = p.tau_max / p.lam
    tlist = np.linspace(0.0, tmax, p.Nt)
    tau_grid = p.lam * tlist

    a, b, c, A, B = op.a, op.b, op.c, op.A, op.B

    n_a_op = a.dag() * a
    n_b_op = b.dag() * b
    n_c_op = c.dag() * c
    n_A_op = A.dag() * A
    n_B_op = B.dag() * B
    AB_op = A * B
    c_mean_op = c

    # Edge-population projectors to diagnose cutoffs.
    P_a_edge = tensor_op(qt.basis(p.Na, p.Na - 1) * qt.basis(p.Na, p.Na - 1).dag(), 0, [p.Na, p.Nb, p.Nc, p.NAf, p.NBf])
    P_b_edge = tensor_op(qt.basis(p.Nb, p.Nb - 1) * qt.basis(p.Nb, p.Nb - 1).dag(), 1, [p.Na, p.Nb, p.Nc, p.NAf, p.NBf])
    P_c_edge = tensor_op(qt.basis(p.Nc, p.Nc - 1) * qt.basis(p.Nc, p.Nc - 1).dag(), 2, [p.Na, p.Nb, p.Nc, p.NAf, p.NBf])
    P_A_edge = tensor_op(qt.basis(p.NAf, p.NAf - 1) * qt.basis(p.NAf, p.NAf - 1).dag(), 3, [p.Na, p.Nb, p.Nc, p.NAf, p.NBf])
    P_B_edge = tensor_op(qt.basis(p.NBf, p.NBf - 1) * qt.basis(p.NBf, p.NBf - 1).dag(), 4, [p.Na, p.Nb, p.Nc, p.NAf, p.NBf])

    e_ops = [
        n_A_op, n_B_op, AB_op,
        n_a_op, n_b_op, n_c_op, c_mean_op,
        P_a_edge, P_b_edge, P_c_edge, P_A_edge, P_B_edge,
    ]
    e_ops = [to_csr_qobj(op_) for op_ in e_ops]

    # Extra safety: force all objects passed to mesolve into CSR.  This is
    # essential for larger collector cutoffs in QuTiP 5.
    H = to_csr_qobj(H)
    rho0 = to_csr_qobj(rho0)
    c_ops = [to_csr_qobj(L) for L in c_ops]

    options = make_mesolve_options(p)

    print(
        f"  mesolve {kind}: Hilbert D={p.hilbert_dim}, "
        f"Liouville size~{p.hilbert_dim**2:.3e}, Nt={p.Nt}"
    )
    tic = time.time()
    sol = qt.mesolve(H, rho0, tlist, c_ops, e_ops=e_ops, options=options)
    toc = time.time()
    print(f"  finished {kind} in {toc - tic:.1f} s")

    N_A = np.real(np.asarray(sol.expect[0], dtype=complex))
    N_B = np.real(np.asarray(sol.expect[1], dtype=complex))
    M_AB = np.asarray(sol.expect[2], dtype=complex)
    n_a = np.real(np.asarray(sol.expect[3], dtype=complex))
    n_b = np.real(np.asarray(sol.expect[4], dtype=complex))
    n_c = np.real(np.asarray(sol.expect[5], dtype=complex))
    c_mean = np.asarray(sol.expect[6], dtype=complex)

    edge_source_a = np.real(np.asarray(sol.expect[7], dtype=complex))
    edge_source_b = np.real(np.asarray(sol.expect[8], dtype=complex))
    edge_pump_c = np.real(np.asarray(sol.expect[9], dtype=complex))
    edge_col_A = np.real(np.asarray(sol.expect[10], dtype=complex))
    edge_col_B = np.real(np.asarray(sol.expect[11], dtype=complex))

    V, Imin, Imax, C, n_pair = temporal_certifiers_from_collectors(N_A, N_B, M_AB)

    eta_c = np.zeros_like(n_c, dtype=float)
    mask = n_c > 1e-14
    eta_c[mask] = np.abs(c_mean[mask]) ** 2 / n_c[mask]

    phi = np.linspace(0.0, 2.0 * np.pi, 401)
    denom_final = N_A[-1] + N_B[-1] + 1.0
    M_final = M_AB[-1]
    I_fringe_final = denom_final + 2.0 * np.real(np.exp(-1j * phi) * M_final)

    physicality_report(kind, Imin, C)
    print(
        f"    final: eta_in={eta_in:.4f}, V={V[-1]:.4f}, Imin={Imin[-1]:.4f}, "
        f"C={C[-1]:.4f}, n_pair={n_pair[-1]:.4f}, "
        f"edge max source/collector="
        f"{max(edge_source_a.max(), edge_source_b.max(), edge_pump_c.max()):.2e}/"
        f"{max(edge_col_A.max(), edge_col_B.max()):.2e}"
    )

    return CollectorResult(
        label=kind,
        eta_in=eta_in,
        tlist=tlist,
        tau_grid=tau_grid,
        N_A=N_A,
        N_B=N_B,
        M_AB=M_AB,
        V=V,
        Imin=Imin,
        Imax=Imax,
        C=C,
        n_pair=n_pair,
        eta_c=eta_c,
        source_n_a=n_a,
        source_n_b=n_b,
        source_n_c=n_c,
        edge_source_a=edge_source_a,
        edge_source_b=edge_source_b,
        edge_pump_c=edge_pump_c,
        edge_col_A=edge_col_A,
        edge_col_B=edge_col_B,
        phi=phi,
        I_fringe_final=I_fringe_final,
    )


# -----------------------------------------------------------------------------
# Linear mixing for the partially dephased family
# -----------------------------------------------------------------------------

@dataclass
class RawCollectorMoments:
    eta_in: float
    tlist: np.ndarray
    tau_grid: np.ndarray
    N_A: np.ndarray
    N_B: np.ndarray
    M_AB: np.ndarray
    n_pair: np.ndarray


def to_raw_collector(r: CollectorResult) -> RawCollectorMoments:
    return RawCollectorMoments(
        eta_in=r.eta_in,
        tlist=r.tlist,
        tau_grid=r.tau_grid,
        N_A=r.N_A,
        N_B=r.N_B,
        M_AB=r.M_AB,
        n_pair=r.n_pair,
    )


def mixed_collector_certifiers(
    coh: RawCollectorMoments,
    deph: RawCollectorMoments,
    eta_grid: np.ndarray,
) -> dict[str, np.ndarray]:
    Nt = len(coh.tau_grid)
    Imin_map = np.zeros((len(eta_grid), Nt))
    V_map = np.zeros((len(eta_grid), Nt))
    C_map = np.zeros((len(eta_grid), Nt))
    n_pair_map = np.zeros((len(eta_grid), Nt))

    for i, eta in enumerate(eta_grid):
        w = np.sqrt(eta)
        N_A = w * coh.N_A + (1.0 - w) * deph.N_A
        N_B = w * coh.N_B + (1.0 - w) * deph.N_B
        M_AB = w * coh.M_AB + (1.0 - w) * deph.M_AB
        V, Imin, Imax, C, n_pair = temporal_certifiers_from_collectors(N_A, N_B, M_AB)
        Imin_map[i, :] = Imin
        V_map[i, :] = V
        C_map[i, :] = C
        n_pair_map[i, :] = n_pair

    eta_threshold = np.full(Nt, np.nan)
    for j in range(Nt):
        good = Imin_map[:, j] < 1.0
        if np.any(good):
            idx = int(np.argmax(good))
            if idx == 0:
                eta_threshold[j] = eta_grid[0]
            else:
                # Linear interpolation in eta for a smoother threshold.
                x0, x1 = eta_grid[idx - 1], eta_grid[idx]
                y0, y1 = Imin_map[idx - 1, j], Imin_map[idx, j]
                if abs(y1 - y0) > 1e-14:
                    eta_threshold[j] = x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0)
                else:
                    eta_threshold[j] = x1

    return {
        "eta_grid": eta_grid,
        "tau_grid": coh.tau_grid,
        "Imin_map": Imin_map,
        "V_map": V_map,
        "C_map": C_map,
        "n_pair_map": n_pair_map,
        "eta_threshold": eta_threshold,
    }


def crossing_threshold(eta_grid: np.ndarray, y_values: np.ndarray, target: float = 1.0) -> float:
    y = np.asarray(y_values, dtype=float)
    good = y < target
    if not np.any(good):
        return np.nan
    idx = int(np.argmax(good))
    if idx == 0:
        return float(eta_grid[0])
    x0, x1 = eta_grid[idx - 1], eta_grid[idx]
    y0, y1 = y[idx - 1], y[idx]
    if abs(y1 - y0) < 1e-14:
        return float(x1)
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))


def stiff_pump_gain_from_rho(rho: np.ndarray | float) -> np.ndarray | float:
    rho_arr = np.asarray(rho)
    out = np.full_like(rho_arr, np.inf, dtype=float)
    mask = rho_arr < 1.0
    out[mask] = ((1.0 + rho_arr[mask] ** 2) / (1.0 - rho_arr[mask] ** 2)) ** 2
    if np.ndim(rho) == 0:
        return float(out)
    return out


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def ensure_outdir(p: Params) -> Path:
    outdir = Path.cwd() / p.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def label_for_kind(kind: str, p: Params) -> str:
    squeezed = parse_squeezed_kind(kind)
    if squeezed is not None:
        angle_kind, r = squeezed
        if angle_kind == "amp":
            return rf"amplitude squeezed, $r={r:.2f}$"
        return rf"phase squeezed, $r={r:.2f}$"
    return {
        "coherent": r"coherent, $\eta_c^{\rm in}=1$",
        "partial": rf"partially dephased, $\eta_c^{{\rm in}}\simeq {p.eta_partial:.2f}$",
        "phase_randomized": r"phase randomized, $\eta_c^{\rm in}=0$",
        "fock": r"Fock",
        "squeezed_vacuum": r"squeezed vacuum",
    }.get(kind, kind)


def set_plot_style() -> None:
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "legend.fontsize": 9,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })


def plot_basic_collectors(results: dict[str, CollectorResult], p: Params) -> None:
    """Plot time traces plus final-time and optimal-time collector fringes.

    The final-time fringe is useful as a diagnostic, but the physically relevant
    collected temporal mode may occur at an earlier readout time. We therefore
    also plot the fringe at the coherent-pump optimum

        tau_star = argmin_tau I_min^{(f)}(tau)

    and use the same time index for all pump states.
    """
    set_plot_style()
    fig, axs = plt.subplots(2, 2, figsize=(9.4, 6.8))

    for kind, r in results.items():
        label = label_for_kind(kind, p)
        tau = r.tau_grid
        axs[0, 0].plot(tau, r.V, label=label)
        axs[0, 1].plot(tau, r.Imin, label=label)
        axs[1, 0].plot(tau, r.C, label=label)
        axs[1, 1].plot(tau, r.n_pair, label=label)

    axs[0, 0].set_title(r"collector-mode phase locking")
    axs[0, 0].set_xlabel(r"readout time $\tau=\lambda T$")
    axs[0, 0].set_ylabel(r"${\cal V}_{f}$")
    axs[0, 0].set_ylim(-0.04, 1.04)

    axs[0, 1].axhline(1.0, linestyle="-.", label=r"$I_{\min}^{(f)}=1$")
    axs[0, 1].set_title(r"collector-mode squeezing certifier")
    axs[0, 1].set_xlabel(r"readout time $\tau=\lambda T$")
    axs[0, 1].set_ylabel(r"$I_{\min}^{(f)}$")
    axs[0, 1].set_yscale("log")

    axs[1, 0].axhline(1.0, linestyle=":", label="ideal TMSV")
    axs[1, 0].set_title(r"collector-mode pair coherence")
    axs[1, 0].set_xlabel(r"readout time $\tau=\lambda T$")
    axs[1, 0].set_ylabel(r"$C_f$")
    axs[1, 0].set_ylim(-0.04, 1.08)

    axs[1, 1].set_title(r"photons in collected output modes")
    axs[1, 1].set_xlabel(r"readout time $\tau=\lambda T$")
    axs[1, 1].set_ylabel(r"$n_f=(N_A+N_B)/2$")

    # Mark the coherent-pump optimum, if present.
    if "coherent" in results:
        r_opt = results["coherent"]
    else:
        r_opt = next(iter(results.values()))
    idx_star = int(np.nanargmin(r_opt.Imin))
    tau_star = float(r_opt.tau_grid[idx_star])
    I_star = float(r_opt.Imin[idx_star])

    # Cutoff diagnostics at the physically relevant readout time.
    # The usual "edge max" printed by solve_collector_case is a maximum over
    # the whole simulated interval.  For the paper figure, the relevant
    # diagnostic is often the edge population at the coherent optimal readout
    # time tau_star.
    print("\nCutoff diagnostics at coherent optimal readout:")
    print(f"  reference: tau_star={tau_star:.4f}, Imin_star={I_star:.6f}")
    for kind, r in results.items():
        edge_source_star = max(
            float(r.edge_source_a[idx_star]),
            float(r.edge_source_b[idx_star]),
            float(r.edge_pump_c[idx_star]),
        )
        edge_col_star = max(
            float(r.edge_col_A[idx_star]),
            float(r.edge_col_B[idx_star]),
        )
        edge_source_max = max(
            float(np.nanmax(r.edge_source_a)),
            float(np.nanmax(r.edge_source_b)),
            float(np.nanmax(r.edge_pump_c)),
        )
        edge_col_max = max(
            float(np.nanmax(r.edge_col_A)),
            float(np.nanmax(r.edge_col_B)),
        )
        print(
            f"  {kind:18s}: "
            f"Imin(tau_star)={float(r.Imin[idx_star]):.6f}, "
            f"V(tau_star)={float(r.V[idx_star]):.6f}, "
            f"C(tau_star)={float(r.C[idx_star]):.6f}, "
            f"n_pair(tau_star)={float(r.n_pair[idx_star]):.6f}, "
            f"edge source/collector at tau_star="
            f"{edge_source_star:.2e}/{edge_col_star:.2e}, "
            f"edge source/collector max="
            f"{edge_source_max:.2e}/{edge_col_max:.2e}"
        )

    for ax in axs.flat:
        ax.axvline(tau_star, color="0.5", linestyle=":", linewidth=1.0)

    for ax in axs.flat:
        ax.tick_params(direction="in")
        ax.legend(frameon=False)

    fig.tight_layout()
    pdf_path = ensure_outdir(p) / "qb_paramp_optionB_collectors_basic.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved collector basic figure to:\n  {pdf_path}")
    print(f"Coherent optimum for collector readout: tau_star={tau_star:.4f}, Imin_star={I_star:.6f}")

    # Final-time fringe. This is a diagnostic, not necessarily the best collected mode.
    fig2, ax = plt.subplots(figsize=(7.2, 4.2))
    for kind, r in results.items():
        ax.plot(r.phi / np.pi, r.I_fringe_final, label=label_for_kind(kind, p))
    ax.axhline(1.0, linestyle="-.", label=r"$I=1$")
    ax.set_xlabel(r"analysis phase $\phi/\pi$")
    ax.set_ylabel(r"$I_f(\phi)$")
    ax.set_title(r"collector fringe at final time")
    ax.tick_params(direction="in")
    ax.legend(frameon=False)
    fig2.tight_layout()
    pdf_path2 = ensure_outdir(p) / "qb_paramp_optionB_collectors_fringe_final.pdf"
    fig2.savefig(pdf_path2, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved final-time collector fringe figure to:\n  {pdf_path2}")

    # Optimal-time fringe: use the coherent-pump optimum readout time for all states.
    fig3, ax = plt.subplots(figsize=(7.2, 4.2))
    for kind, r in results.items():
        denom = r.N_A[idx_star] + r.N_B[idx_star] + 1.0
        M = r.M_AB[idx_star]
        I_phi = denom + 2.0 * np.real(np.exp(-1j * r.phi) * M)
        ax.plot(r.phi / np.pi, I_phi, label=label_for_kind(kind, p))
    ax.axhline(1.0, linestyle="-.", label=r"$I=1$")
    ax.set_xlabel(r"analysis phase $\phi/\pi$")
    ax.set_ylabel(r"$I_f(\phi)$")
    ax.set_title(rf"collector fringe at optimal readout $\tau_\star={tau_star:.2f}$")
    ax.tick_params(direction="in")
    ax.legend(frameon=False)
    fig3.tight_layout()
    pdf_path3 = ensure_outdir(p) / "qb_paramp_optionB_collectors_fringe_optimal.pdf"
    fig3.savefig(pdf_path3, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved optimal-time collector fringe figure to:\n  {pdf_path3}")


def plot_threshold_map(scan: dict[str, np.ndarray], p: Params, filename: str) -> None:
    set_plot_style()
    tau = scan["tau_grid"]
    eta = scan["eta_grid"]
    T, E = np.meshgrid(tau, eta)
    Imin = scan["Imin_map"]
    V = scan["V_map"]
    eta_thr = scan["eta_threshold"]
    closed = np.tanh(tau) ** 2

    fig, axs = plt.subplots(1, 3, figsize=(13.8, 4.0))
    im0 = axs[0].pcolormesh(T, E, Imin, shading="auto")
    cbar0 = fig.colorbar(im0, ax=axs[0])
    cbar0.set_label(r"$I_{\min}^{(f)}$")
    axs[0].contour(T, E, Imin, levels=[1.0], linewidths=1.8)
    axs[0].plot(tau, closed, linestyle="--", label=r"closed guide $\tanh^2\tau$")
    axs[0].set_xlabel(r"final time $\tau=\lambda T$")
    axs[0].set_ylabel(r"initial phase coherence $\eta_c^{\rm in}$")
    axs[0].set_title(r"collector-mode squeezing threshold")
    axs[0].legend(frameon=False)

    im1 = axs[1].pcolormesh(T, E, V, shading="auto", vmin=0.0, vmax=1.0)
    cbar1 = fig.colorbar(im1, ax=axs[1])
    cbar1.set_label(r"${\cal V}_{f}$")
    axs[1].contour(T, E, Imin, levels=[1.0], linewidths=1.8)
    axs[1].set_xlabel(r"final time $\tau=\lambda T$")
    axs[1].set_ylabel(r"initial phase coherence $\eta_c^{\rm in}$")
    axs[1].set_title(r"phase locking of collected modes")

    axs[2].plot(tau, eta_thr, "o-", ms=3, label=r"collector $I_{\min}^{(f)}=1$")
    axs[2].plot(tau, closed, linestyle="--", label=r"closed guide $\tanh^2\tau$")
    axs[2].set_xlabel(r"final time $\tau=\lambda T$")
    axs[2].set_ylabel(r"critical $\eta_c^{\rm in}$")
    axs[2].set_ylim(-0.04, 1.04)
    axs[2].set_title(r"minimum coherent pump fraction")
    axs[2].legend(frameon=False)

    for ax in axs:
        ax.tick_params(direction="in")

    fig.tight_layout()
    pdf_path = ensure_outdir(p) / filename
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved collector threshold map to:\n  {pdf_path}")


# -----------------------------------------------------------------------------
# Modes/runs
# -----------------------------------------------------------------------------

def print_params(p: Params, title: str) -> None:
    print(title)
    print(f"QuTiP version = {qt.__version__}")
    print(f"nbar = {p.nbar:.3f}")
    print(f"cutoffs: Na={p.Na}, Nb={p.Nb}, Nc={p.Nc}, NAf={p.NAf}, NBf={p.NBf}")
    print(f"Hilbert dimension D = {p.hilbert_dim}; Liouville dimension D^2 ~ {p.hilbert_dim**2:.3e}")
    print(f"lambda = {p.lam:.6f}, g = {p.g:.6f}")
    print(f"kappa_a = {p.kappa_a:.3f}, kappa_b = {p.kappa_b:.3f}, kappa_c = {p.kappa_c:.3f}")
    print(f"gamma_A = {p.gamma_A:.3f}, gamma_B = {p.gamma_B:.3f}")
    print(f"rho = {p.rho_reduced:.3f}, G0 = {p.stiff_pump_gain_estimate:.3f}")
    print(f"tau_max = {p.tau_max:.3f}, Nt = {p.Nt}")
    print()


def run_basic(p: Params, kinds: list[str]) -> None:
    print_params(p, "Option B: cascaded collector basic run")
    results: dict[str, CollectorResult] = {}
    for kind in kinds:
        print(f"Solving collector case: {kind}")
        results[kind] = solve_collector_case(kind, p)
        print()
    plot_basic_collectors(results, p)


def run_threshold(p: Params, n_eta: int = 101) -> None:
    print_params(p, "Option B: collector phase-coherence threshold map")
    coh = solve_collector_case("coherent", p)
    deph = solve_collector_case("phase_randomized", p)
    eta_grid = np.linspace(0.0, 1.0, n_eta)
    scan = mixed_collector_certifiers(to_raw_collector(coh), to_raw_collector(deph), eta_grid)
    print("Selected collector threshold values")
    for idx in np.linspace(0, len(scan["tau_grid"]) - 1, 8, dtype=int):
        tau = scan["tau_grid"][idx]
        et = scan["eta_threshold"][idx]
        print(f"  tau={tau:.3f}, eta_crit={et if np.isfinite(et) else np.nan:.4f}")
    plot_threshold_map(scan, p, "qb_paramp_optionB_collectors_threshold.pdf")


def run_gain_scan(p_base: Params, rho_values: np.ndarray, n_eta: int = 101) -> None:
    print_params(p_base, "Option B: collector gain-threshold scan")
    eta_grid = np.linspace(0.0, 1.0, n_eta)

    rho_list = []
    G0_list = []
    eta_crit_list = []
    n_pair_eta0_list = []
    n_pair_eta1_list = []
    I0_list = []
    I1_list = []

    selected_curves = []

    for rix, rho in enumerate(rho_values):
        lam = 0.5 * float(rho) * np.sqrt(p_base.kappa_a * p_base.kappa_b)
        p = replace(p_base, lam=lam)
        print(f"\nScanning rho={rho:.3f}, lambda={lam:.4f}, G0={stiff_pump_gain_from_rho(float(rho)):.3f}")
        coh = solve_collector_case("coherent", p)
        deph = solve_collector_case("phase_randomized", p)
        mix = mixed_collector_certifiers(to_raw_collector(coh), to_raw_collector(deph), eta_grid)
        I_final = mix["Imin_map"][:, -1]
        eta_crit = crossing_threshold(eta_grid, I_final, 1.0)
        rho_list.append(float(rho))
        G0_list.append(float(stiff_pump_gain_from_rho(float(rho))))
        eta_crit_list.append(eta_crit)
        I0_list.append(float(I_final[0]))
        I1_list.append(float(I_final[-1]))
        n_pair_eta0_list.append(float(mix["n_pair_map"][0, -1]))
        n_pair_eta1_list.append(float(mix["n_pair_map"][-1, -1]))
        print(
            f"  eta_crit={eta_crit if np.isfinite(eta_crit) else np.nan:.4f}, "
            f"Imin eta0/eta1={I_final[0]:.4f}/{I_final[-1]:.4f}, "
            f"n_pair eta0/eta1={mix['n_pair_map'][0,-1]:.4f}/{mix['n_pair_map'][-1,-1]:.4f}"
        )
        if rix in {0, len(rho_values)//3, 2*len(rho_values)//3, len(rho_values)-1}:
            selected_curves.append((float(rho), float(stiff_pump_gain_from_rho(float(rho))), I_final.copy()))

    rho_arr = np.array(rho_list)
    G0_arr = np.array(G0_list)
    eta_arr = np.array(eta_crit_list)
    n0 = np.array(n_pair_eta0_list)
    n1 = np.array(n_pair_eta1_list)

    set_plot_style()
    fig, axs = plt.subplots(1, 3, figsize=(13.8, 4.0))
    axs[0].plot(rho_arr, eta_arr, "o-", label=r"$I_{\min}^{(f)}=1$")
    axs[0].set_xlabel(r"reduced pump strength $\rho=2\lambda/\sqrt{\kappa_a\kappa_b}$")
    axs[0].set_ylabel(r"critical $\eta_c^{\rm in}$")
    axs[0].set_title(r"collector threshold")
    axs[0].set_ylim(-0.04, 1.04)
    axs[0].legend(frameon=False)

    axs[1].plot(G0_arr, eta_arr, "o-", label=r"$I_{\min}^{(f)}=1$")
    axs[1].set_xscale("log")
    axs[1].set_xlabel(r"stiff-pump gain estimate $G_0$")
    axs[1].set_ylabel(r"critical $\eta_c^{\rm in}$")
    axs[1].set_title(r"threshold versus nominal gain")
    axs[1].set_ylim(-0.04, 1.04)
    axs[1].legend(frameon=False)

    axs[2].plot(rho_arr, n1, "o-", label=r"$\eta_c^{\rm in}=1$")
    axs[2].plot(rho_arr, n0, "s--", label=r"$\eta_c^{\rm in}=0$")
    axs[2].set_xlabel(r"reduced pump strength $\rho$")
    axs[2].set_ylabel(r"collected pair number $n_f$")
    axs[2].set_title(r"collected energy is weakly phase-sensitive")
    axs[2].legend(frameon=False)

    for ax in axs:
        ax.tick_params(direction="in")
    fig.tight_layout()
    pdf_path = ensure_outdir(p_base) / "qb_paramp_optionB_collectors_gain_threshold.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved collector gain-threshold figure to:\n  {pdf_path}")

    fig2, ax = plt.subplots(figsize=(7.2, 4.4))
    for rho, G0, Icurve in selected_curves:
        ax.plot(eta_grid, Icurve, label=rf"$\rho={rho:.2f}$, $G_0={G0:.1f}$")
    ax.axhline(1.0, linestyle="-.", label=r"$I_{\min}^{(f)}=1$")
    ax.set_xlabel(r"initial phase coherence $\eta_c^{\rm in}$")
    ax.set_ylabel(r"final $I_{\min}^{(f)}(T)$")
    ax.set_title(r"collector threshold extraction")
    ax.tick_params(direction="in")
    ax.legend(frameon=False)
    fig2.tight_layout()
    pdf_path2 = ensure_outdir(p_base) / "qb_paramp_optionB_collectors_gain_extraction.pdf"
    fig2.savefig(pdf_path2, bbox_inches="tight")
    print(f"Saved collector gain-extraction figure to:\n  {pdf_path2}")


def auto_cutoffs_for_nbar_optionB(
    nbar: float,
    Na_min: int,
    Nb_min: int,
    NAf: int,
    NBf: int,
) -> tuple[int, int, int, int, int]:
    Nc_tail = int(np.ceil(nbar + 5.5 * np.sqrt(nbar + 1.0) + 6.0))
    Nc_fock_safe = int(round(nbar)) + 4
    Nc = max(12, Nc_tail, Nc_fock_safe)
    return Na_min, Nb_min, Nc, NAf, NBf


def run_nbar_scan(p_base: Params, nbar_values: np.ndarray, rho_fixed: float, n_eta: int = 101) -> None:
    print_params(p_base, "Option B: collector nbar-threshold scan")
    eta_grid = np.linspace(0.0, 1.0, n_eta)

    nbar_list = []
    eta_crit_list = []
    dep0_list = []
    dep1_list = []
    n0_list = []
    n1_list = []
    selected_curves = []

    for nix, nbar in enumerate(nbar_values):
        Na, Nb, Nc, NAf, NBf = auto_cutoffs_for_nbar_optionB(
            float(nbar), p_base.Na, p_base.Nb, p_base.NAf, p_base.NBf
        )
        lam = 0.5 * rho_fixed * np.sqrt(p_base.kappa_a * p_base.kappa_b)
        p = replace(p_base, nbar=float(nbar), lam=lam, Na=Na, Nb=Nb, Nc=Nc, NAf=NAf, NBf=NBf)
        print(f"\nScanning nbar={nbar:.3f}, cutoffs Na={Na},Nb={Nb},Nc={Nc},NAf={NAf},NBf={NBf}")
        coh = solve_collector_case("coherent", p)
        deph = solve_collector_case("phase_randomized", p)
        mix = mixed_collector_certifiers(to_raw_collector(coh), to_raw_collector(deph), eta_grid)
        I_final = mix["Imin_map"][:, -1]
        eta_crit = crossing_threshold(eta_grid, I_final, 1.0)
        n0 = float(mix["n_pair_map"][0, -1])
        n1 = float(mix["n_pair_map"][-1, -1])
        nbar_list.append(float(nbar))
        eta_crit_list.append(eta_crit)
        n0_list.append(n0)
        n1_list.append(n1)
        dep0_list.append(n0 / float(nbar))
        dep1_list.append(n1 / float(nbar))
        print(
            f"  eta_crit={eta_crit if np.isfinite(eta_crit) else np.nan:.4f}, "
            f"Imin eta0/eta1={I_final[0]:.4f}/{I_final[-1]:.4f}, n_pair eta0/eta1={n0:.4f}/{n1:.4f}"
        )
        if nix in {0, len(nbar_values)//3, 2*len(nbar_values)//3, len(nbar_values)-1}:
            selected_curves.append((float(nbar), I_final.copy()))

    nbar_arr = np.array(nbar_list)
    eta_arr = np.array(eta_crit_list)
    n0_arr = np.array(n0_list)
    n1_arr = np.array(n1_list)
    dep0_arr = np.array(dep0_list)
    dep1_arr = np.array(dep1_list)

    set_plot_style()
    fig, axs = plt.subplots(1, 3, figsize=(13.8, 4.0))
    axs[0].plot(nbar_arr, eta_arr, "o-", label=rf"$\rho={rho_fixed:.2f}$")
    axs[0].set_xlabel(r"pump/battery energy $\bar n_c$")
    axs[0].set_ylabel(r"critical $\eta_c^{\rm in}$")
    axs[0].set_title(r"collector battery-size dependence")
    axs[0].set_ylim(-0.04, 1.04)
    axs[0].legend(frameon=False)

    axs[1].plot(nbar_arr, dep1_arr, "o-", label=r"$\eta_c^{\rm in}=1$")
    axs[1].plot(nbar_arr, dep0_arr, "s--", label=r"$\eta_c^{\rm in}=0$")
    axs[1].set_xlabel(r"pump/battery energy $\bar n_c$")
    axs[1].set_ylabel(r"collected pair number$/\bar n_c$")
    axs[1].set_title(r"relative battery use")
    axs[1].legend(frameon=False)

    axs[2].plot(nbar_arr, n1_arr, "o-", label=r"$\eta_c^{\rm in}=1$")
    axs[2].plot(nbar_arr, n0_arr, "s--", label=r"$\eta_c^{\rm in}=0$")
    axs[2].set_xlabel(r"pump/battery energy $\bar n_c$")
    axs[2].set_ylabel(r"collected pair number $n_f$")
    axs[2].set_title(r"collected energy at fixed operating point")
    axs[2].legend(frameon=False)

    for ax in axs:
        ax.tick_params(direction="in")
    fig.tight_layout()
    pdf_path = ensure_outdir(p_base) / "qb_paramp_optionB_collectors_nbar_threshold.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved collector nbar-threshold figure to:\n  {pdf_path}")

    fig2, ax = plt.subplots(figsize=(7.2, 4.4))
    for nbar, Icurve in selected_curves:
        ax.plot(eta_grid, Icurve, label=rf"$\bar n_c={nbar:.1f}$")
    ax.axhline(1.0, linestyle="-.", label=r"$I_{\min}^{(f)}=1$")
    ax.set_xlabel(r"initial phase coherence $\eta_c^{\rm in}$")
    ax.set_ylabel(r"final $I_{\min}^{(f)}(T)$")
    ax.set_title(r"collector threshold extraction at fixed $\rho$")
    ax.tick_params(direction="in")
    ax.legend(frameon=False)
    fig2.tight_layout()
    pdf_path2 = ensure_outdir(p_base) / "qb_paramp_optionB_collectors_nbar_extraction.pdf"
    fig2.savefig(pdf_path2, bbox_inches="tight")
    print(f"Saved collector nbar-extraction figure to:\n  {pdf_path2}")



# -----------------------------------------------------------------------------
# Collector bandwidth scan
# -----------------------------------------------------------------------------

def run_gamma_scan(p_base: Params, gamma_values: np.ndarray, n_eta: int = 101) -> None:
    """Scan the collector/filter bandwidth gamma_A=gamma_B=gamma_f.

    For each gamma_f we solve the coherent and phase-randomized endpoints, then
    use linearity to reconstruct the partially dephased family. The threshold is
    extracted at the coherent-pump optimal readout time tau_star(gamma_f), where
    I_min^{(f)} is minimal for the coherent pump.
    """
    print_params(p_base, "Option B: collector bandwidth scan")
    eta_grid = np.linspace(0.0, 1.0, n_eta)

    gamma_list = []
    tau_star_list = []
    I_star_coh_list = []
    I_star_deph_list = []
    eta_crit_list = []
    V_star_list = []
    C_star_list = []
    n_pair_coh_list = []
    n_pair_deph_list = []
    edge_source_list = []
    edge_collector_list = []

    selected_time_curves = []

    selected_set = {0, len(gamma_values)//3, 2*len(gamma_values)//3, len(gamma_values)-1}

    for gix, gamma in enumerate(gamma_values):
        p = replace(p_base, gamma_A=float(gamma), gamma_B=float(gamma))
        print(f"\nScanning gamma_f/lambda = {float(gamma)/p.lam:.3f}  (gamma_f={float(gamma):.4f})")
        coh = solve_collector_case("coherent", p)
        deph = solve_collector_case("phase_randomized", p)

        idx_star = int(np.nanargmin(coh.Imin))
        tau_star = float(coh.tau_grid[idx_star])
        I_star = float(coh.Imin[idx_star])
        I_deph_star = float(deph.Imin[idx_star])

        mix = mixed_collector_certifiers(to_raw_collector(coh), to_raw_collector(deph), eta_grid)
        I_eta_star = mix["Imin_map"][:, idx_star]
        eta_crit = crossing_threshold(eta_grid, I_eta_star, 1.0)

        gamma_list.append(float(gamma))
        tau_star_list.append(tau_star)
        I_star_coh_list.append(I_star)
        I_star_deph_list.append(I_deph_star)
        eta_crit_list.append(eta_crit)
        V_star_list.append(float(coh.V[idx_star]))
        C_star_list.append(float(coh.C[idx_star]))
        n_pair_coh_list.append(float(coh.n_pair[idx_star]))
        n_pair_deph_list.append(float(deph.n_pair[idx_star]))
        edge_source_list.append(float(max(coh.edge_source_a.max(), coh.edge_source_b.max(), coh.edge_pump_c.max())))
        edge_collector_list.append(float(max(coh.edge_col_A.max(), coh.edge_col_B.max())))

        print(
            f"  tau_star={tau_star:.4f}, Imin_coh(tau_star)={I_star:.6f}, "
            f"Imin_deph(tau_star)={I_deph_star:.6f}, eta_crit={eta_crit if np.isfinite(eta_crit) else np.nan:.4f}, "
            f"C_star={coh.C[idx_star]:.4f}, n_pair coh/deph={coh.n_pair[idx_star]:.4f}/{deph.n_pair[idx_star]:.4f}"
        )

        if gix in selected_set:
            selected_time_curves.append((float(gamma), coh.tau_grid.copy(), coh.Imin.copy(), deph.Imin.copy()))

    gamma_arr = np.array(gamma_list)
    tau_star_arr = np.array(tau_star_list)
    I_star_coh_arr = np.array(I_star_coh_list)
    I_star_deph_arr = np.array(I_star_deph_list)
    eta_arr = np.array(eta_crit_list)
    V_star_arr = np.array(V_star_list)
    C_star_arr = np.array(C_star_list)
    n_coh_arr = np.array(n_pair_coh_list)
    n_deph_arr = np.array(n_pair_deph_list)
    edge_source_arr = np.array(edge_source_list)
    edge_collector_arr = np.array(edge_collector_list)

    best_idx = int(np.nanargmin(I_star_coh_arr))
    print("\nBest bandwidth by coherent minimum Imin:")
    print(
        f"  gamma_f={gamma_arr[best_idx]:.6f}, tau_star={tau_star_arr[best_idx]:.6f}, "
        f"Imin_star={I_star_coh_arr[best_idx]:.6f}, eta_crit={eta_arr[best_idx]:.6f}"
    )

    set_plot_style()
    fig, axs = plt.subplots(2, 2, figsize=(10.0, 7.0))

    x = gamma_arr / p_base.lam
    axs[0, 0].plot(x, I_star_coh_arr, "o-", label=r"coherent")
    axs[0, 0].plot(x, I_star_deph_arr, "s--", label=r"phase randomized")
    axs[0, 0].axhline(1.0, linestyle="-.", label=r"$I_{\min}^{(f)}=1$")
    axs[0, 0].set_xlabel(r"collector bandwidth $\gamma_f/\lambda$")
    axs[0, 0].set_ylabel(r"minimum $I_{\min}^{(f)}$")
    axs[0, 0].set_title(r"best collected squeezing")
    axs[0, 0].legend(frameon=False)

    axs[0, 1].plot(x, tau_star_arr, "o-")
    axs[0, 1].set_xlabel(r"collector bandwidth $\gamma_f/\lambda$")
    axs[0, 1].set_ylabel(r"optimal readout time $\tau_\star$")
    axs[0, 1].set_title(r"matched readout time")

    axs[1, 0].plot(x, eta_arr, "o-", label=r"threshold at $\tau_\star$")
    axs[1, 0].set_xlabel(r"collector bandwidth $\gamma_f/\lambda$")
    axs[1, 0].set_ylabel(r"critical $\eta_c^{\rm in}$")
    axs[1, 0].set_ylim(-0.04, 1.04)
    axs[1, 0].set_title(r"phase-coherence threshold")
    axs[1, 0].legend(frameon=False)

    axs[1, 1].plot(x, n_coh_arr, "o-", label=r"$\eta_c^{\rm in}=1$")
    axs[1, 1].plot(x, n_deph_arr, "s--", label=r"$\eta_c^{\rm in}=0$")
    axs[1, 1].set_xlabel(r"collector bandwidth $\gamma_f/\lambda$")
    axs[1, 1].set_ylabel(r"collected pair number at $\tau_\star$")
    axs[1, 1].set_title(r"collected energy at optimal readout")
    axs[1, 1].legend(frameon=False)

    for ax in axs.flat:
        ax.tick_params(direction="in")

    fig.tight_layout()
    pdf_path = ensure_outdir(p_base) / "qb_paramp_optionB_collectors_gamma_scan.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved collector bandwidth scan to:\n  {pdf_path}")

    # Selected time traces for gamma scan.
    fig2, ax = plt.subplots(figsize=(7.4, 4.6))
    for gamma, tau, I_coh, I_deph in selected_time_curves:
        ax.plot(tau, I_coh, label=rf"coh., $\gamma_f/\lambda={gamma/p_base.lam:.2f}$")
        ax.plot(tau, I_deph, linestyle="--", alpha=0.65, label=rf"deph., $\gamma_f/\lambda={gamma/p_base.lam:.2f}$")
    ax.axhline(1.0, linestyle="-.", label=r"$I_{\min}^{(f)}=1$")
    ax.set_xlabel(r"readout time $\tau=\lambda T$")
    ax.set_ylabel(r"$I_{\min}^{(f)}(\tau)$")
    ax.set_title(r"collector-bandwidth comparison")
    ax.tick_params(direction="in")
    ax.legend(frameon=False, ncol=2)
    fig2.tight_layout()
    pdf_path2 = ensure_outdir(p_base) / "qb_paramp_optionB_collectors_gamma_traces.pdf"
    fig2.savefig(pdf_path2, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved collector bandwidth time traces to:\n  {pdf_path2}")

    # Edge-population diagnostic.
    fig3, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.semilogy(x, np.maximum(edge_source_arr, 1e-300), "o-", label="source edge")
    ax.semilogy(x, np.maximum(edge_collector_arr, 1e-300), "s--", label="collector edge")
    ax.set_xlabel(r"collector bandwidth $\gamma_f/\lambda$")
    ax.set_ylabel("maximum edge population")
    ax.set_title("cutoff diagnostic during bandwidth scan")
    ax.tick_params(direction="in")
    ax.legend(frameon=False)
    fig3.tight_layout()
    pdf_path3 = ensure_outdir(p_base) / "qb_paramp_optionB_collectors_gamma_edges.pdf"
    fig3.savefig(pdf_path3, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved collector bandwidth edge diagnostics to:\n  {pdf_path3}")



# -----------------------------------------------------------------------------
# Displaced squeezed pump-battery scan
# -----------------------------------------------------------------------------

@dataclass
class SqueezedScanPoint:
    angle: str
    r: float
    kind: str
    eta_in: float
    n_init: float
    var_init: float
    edge_init: float
    tau_star_ref: float
    I_tau_star: float
    C_tau_star: float
    V_tau_star: float
    n_pair_tau_star: float
    tau_min: float
    I_min_best: float
    C_at_best: float
    n_pair_at_best: float
    edge_source_tau_star: float
    edge_col_tau_star: float


def squeezed_kind(angle: str, r: float) -> str:
    key = angle.lower().strip()
    if key in {"amp", "amplitude", "amplitude_squeezed"}:
        return f"amp_squeezed:{float(r):.12g}"
    if key in {"phase", "phase_squeezed"}:
        return f"phase_squeezed:{float(r):.12g}"
    raise ValueError(f"Unknown squeezed angle: {angle}")


def pump_stats_for_kind(kind: str, p: Params) -> dict[str, float | complex]:
    rho, _ = pump_state_dm(kind, p)
    return single_mode_stats(rho)


def result_at_index(r: CollectorResult, idx: int) -> tuple[float, float, float, float, float, float]:
    edge_source = max(
        float(r.edge_source_a[idx]),
        float(r.edge_source_b[idx]),
        float(r.edge_pump_c[idx]),
    )
    edge_col = max(float(r.edge_col_A[idx]), float(r.edge_col_B[idx]))
    return (
        float(r.Imin[idx]),
        float(r.C[idx]),
        float(r.V[idx]),
        float(r.n_pair[idx]),
        edge_source,
        edge_col,
    )


def run_squeezed_scan(
    p: Params,
    r_values: np.ndarray,
    angles: list[str],
    include_squeezed_vacuum: bool = False,
) -> None:
    """Stress-test the resource interpretation using displaced squeezed pumps.

    For fixed total pump energy nbar, the displaced squeezed state obeys
        nbar = |alpha|^2 + sinh^2(r).
    We compare amplitude-squeezed and phase-squeezed choices.  The values are
    evaluated at the coherent-pump optimal readout time tau_star, and we also
    record the independently optimized minimum for each squeezed state.
    """
    print_params(p, "Option B: displaced squeezed pump-battery stress test")
    r_values = np.array(sorted(set(float(x) for x in r_values)), dtype=float)
    if np.any(r_values < -1e-14):
        raise ValueError("All squeezing values r must be non-negative.")

    r_max_allowed = float(np.arcsinh(np.sqrt(p.nbar)))
    if np.any(r_values > r_max_allowed + 1e-12):
        raise ValueError(
            f"At fixed nbar={p.nbar}, displaced squeezed states require "
            f"sinh(r)^2 <= nbar.  Maximum allowed r is {r_max_allowed:.6f}."
        )

    # Coherent reference; this is also the r=0 displaced squeezed state.
    print("Solving coherent reference for squeezed-state scan")
    coh = solve_collector_case("coherent", p)
    idx_star = int(np.nanargmin(coh.Imin))
    tau_star = float(coh.tau_grid[idx_star])
    print(f"Coherent reference: tau_star={tau_star:.6f}, Imin_star={coh.Imin[idx_star]:.6f}")

    results_by_kind: dict[str, CollectorResult] = {"coherent": coh}
    points: list[SqueezedScanPoint] = []

    # Add coherent r=0 point for every requested squeezed angle.
    for angle in angles:
        angle_key = "amp" if angle.lower().strip().startswith("amp") else "phase"
        stats = pump_stats_for_kind("coherent", p)
        I0, C0, V0, nf0, es0, ec0 = result_at_index(coh, idx_star)
        idx_min = int(np.nanargmin(coh.Imin))
        points.append(SqueezedScanPoint(
            angle=angle_key,
            r=0.0,
            kind="coherent",
            eta_in=float(stats["eta"]),
            n_init=float(stats["n"]),
            var_init=float(stats["var"]),
            edge_init=float(stats["edge"]),
            tau_star_ref=tau_star,
            I_tau_star=I0,
            C_tau_star=C0,
            V_tau_star=V0,
            n_pair_tau_star=nf0,
            tau_min=float(coh.tau_grid[idx_min]),
            I_min_best=float(coh.Imin[idx_min]),
            C_at_best=float(coh.C[idx_min]),
            n_pair_at_best=float(coh.n_pair[idx_min]),
            edge_source_tau_star=es0,
            edge_col_tau_star=ec0,
        ))

    # Solve nonzero squeezed states.  Do not duplicate r=0.
    for angle in angles:
        angle_key = "amp" if angle.lower().strip().startswith("amp") else "phase"
        for r_val in r_values:
            if abs(float(r_val)) < 1e-14:
                continue
            kind = squeezed_kind(angle_key, float(r_val))
            print(f"\nSolving squeezed pump state: angle={angle_key}, r={float(r_val):.4f}")
            stats = pump_stats_for_kind(kind, p)
            print(
                f"  initial pump diagnostics: "
                f"n={float(stats['n']):.6f}, var={float(stats['var']):.6f}, "
                f"var/nbar={float(stats['var'])/p.nbar:.6f}, "
                f"eta={float(stats['eta']):.6f}, edge={float(stats['edge']):.3e}"
            )
            res = solve_collector_case(kind, p)
            results_by_kind[kind] = res
            I_s, C_s, V_s, nf_s, es_s, ec_s = result_at_index(res, idx_star)
            idx_min = int(np.nanargmin(res.Imin))
            points.append(SqueezedScanPoint(
                angle=angle_key,
                r=float(r_val),
                kind=kind,
                eta_in=float(stats["eta"]),
                n_init=float(stats["n"]),
                var_init=float(stats["var"]),
                edge_init=float(stats["edge"]),
                tau_star_ref=tau_star,
                I_tau_star=I_s,
                C_tau_star=C_s,
                V_tau_star=V_s,
                n_pair_tau_star=nf_s,
                tau_min=float(res.tau_grid[idx_min]),
                I_min_best=float(res.Imin[idx_min]),
                C_at_best=float(res.C[idx_min]),
                n_pair_at_best=float(res.n_pair[idx_min]),
                edge_source_tau_star=es_s,
                edge_col_tau_star=ec_s,
            ))

    if include_squeezed_vacuum:
        kind = "squeezed_vacuum"
        r_vac = float(np.arcsinh(np.sqrt(p.nbar)))
        print(f"\nSolving squeezed vacuum pump state: r={r_vac:.4f}")
        stats = pump_stats_for_kind(kind, p)
        print(
            f"  initial pump diagnostics: "
            f"n={float(stats['n']):.6f}, var={float(stats['var']):.6f}, "
            f"var/nbar={float(stats['var'])/p.nbar:.6f}, "
            f"eta={float(stats['eta']):.6f}, edge={float(stats['edge']):.3e}"
        )
        res = solve_collector_case(kind, p)
        results_by_kind[kind] = res
        I_s, C_s, V_s, nf_s, es_s, ec_s = result_at_index(res, idx_star)
        idx_min = int(np.nanargmin(res.Imin))
        points.append(SqueezedScanPoint(
            angle="vacuum",
            r=r_vac,
            kind=kind,
            eta_in=float(stats["eta"]),
            n_init=float(stats["n"]),
            var_init=float(stats["var"]),
            edge_init=float(stats["edge"]),
            tau_star_ref=tau_star,
            I_tau_star=I_s,
            C_tau_star=C_s,
            V_tau_star=V_s,
            n_pair_tau_star=nf_s,
            tau_min=float(res.tau_grid[idx_min]),
            I_min_best=float(res.Imin[idx_min]),
            C_at_best=float(res.C[idx_min]),
            n_pair_at_best=float(res.n_pair[idx_min]),
            edge_source_tau_star=es_s,
            edge_col_tau_star=ec_s,
        ))

    print("\nSqueezed-state summary at coherent optimal readout")
    print("angle   r       eta_in    var/nbar   I(tau*)   C(tau*)   n_f(tau*)   tau_min   I_min_best   edge_src/edge_col")
    for pt in sorted(points, key=lambda x: (x.angle, x.r)):
        print(
            f"{pt.angle:7s} {pt.r:6.3f}  {pt.eta_in:8.4f}  "
            f"{pt.var_init/p.nbar:8.4f}  {pt.I_tau_star:8.4f}  "
            f"{pt.C_tau_star:8.4f}  {pt.n_pair_tau_star:9.4f}  "
            f"{pt.tau_min:7.4f}  {pt.I_min_best:10.4f}  "
            f"{pt.edge_source_tau_star:.2e}/{pt.edge_col_tau_star:.2e}"
        )

    plot_squeezed_scan(points, results_by_kind, p, idx_star, tau_star)


def plot_squeezed_scan(
    points: list[SqueezedScanPoint],
    results_by_kind: dict[str, CollectorResult],
    p: Params,
    idx_star: int,
    tau_star: float,
) -> None:
    set_plot_style()
    angle_styles = {
        "amp": {"marker": "o", "linestyle": "-", "label": "amplitude squeezed"},
        "phase": {"marker": "s", "linestyle": "--", "label": "phase squeezed"},
    }

    fig, axs = plt.subplots(2, 2, figsize=(10.0, 7.0))

    for angle, style in angle_styles.items():
        pts = sorted([pt for pt in points if pt.angle == angle], key=lambda x: x.r)
        if not pts:
            continue
        r = np.array([pt.r for pt in pts])
        I_tau = np.array([pt.I_tau_star for pt in pts])
        I_best = np.array([pt.I_min_best for pt in pts])
        C_tau = np.array([pt.C_tau_star for pt in pts])
        nf_tau = np.array([pt.n_pair_tau_star for pt in pts])
        eta = np.array([pt.eta_in for pt in pts])
        var_norm = np.array([pt.var_init / p.nbar for pt in pts])

        axs[0, 0].plot(r, I_tau, marker=style["marker"], linestyle=style["linestyle"], label=style["label"] + r" at $\tau_\star$")
        axs[0, 0].plot(r, I_best, marker=style["marker"], linestyle=":", alpha=0.75, label=style["label"] + r" optimized")
        axs[0, 1].plot(r, C_tau, marker=style["marker"], linestyle=style["linestyle"], label=style["label"])
        axs[1, 0].plot(r, nf_tau, marker=style["marker"], linestyle=style["linestyle"], label=style["label"])
        axs[1, 1].plot(r, eta, marker=style["marker"], linestyle=style["linestyle"], label=style["label"] + r" $\eta_c^{\rm in}$")
        axs[1, 1].plot(r, var_norm, marker=style["marker"], linestyle=":", alpha=0.75, label=style["label"] + r" ${\rm Var}(n_c)/\bar n_c$")

    vac_pts = [pt for pt in points if pt.angle == "vacuum"]
    if vac_pts:
        pt = vac_pts[0]
        axs[0, 0].plot([pt.r], [pt.I_tau_star], "X", ms=8, label="squeezed vacuum")
        axs[0, 1].plot([pt.r], [pt.C_tau_star], "X", ms=8, label="squeezed vacuum")
        axs[1, 0].plot([pt.r], [pt.n_pair_tau_star], "X", ms=8, label="squeezed vacuum")
        axs[1, 1].plot([pt.r], [pt.eta_in], "X", ms=8, label="squeezed vacuum")

    axs[0, 0].axhline(1.0, linestyle="-.", label=r"$I_{\min}^{(f)}=1$")
    axs[0, 0].set_xlabel(r"squeezing parameter $r$")
    axs[0, 0].set_ylabel(r"$I_{\min}^{(f)}$")
    axs[0, 0].set_title(rf"collector certifier at $\tau_\star={tau_star:.2f}$")
    axs[0, 0].set_yscale("log")

    axs[0, 1].set_xlabel(r"squeezing parameter $r$")
    axs[0, 1].set_ylabel(r"$C_f$")
    axs[0, 1].set_title(r"temporal-mode pair coherence")
    axs[0, 1].set_ylim(-0.04, 1.04)

    axs[1, 0].set_xlabel(r"squeezing parameter $r$")
    axs[1, 0].set_ylabel(r"$n_f=(N_A+N_B)/2$")
    axs[1, 0].set_title(r"collected energy")

    axs[1, 1].set_xlabel(r"squeezing parameter $r$")
    axs[1, 1].set_ylabel(r"initial pump diagnostic")
    axs[1, 1].set_title(r"phase reference versus number noise")

    for ax in axs.flat:
        ax.tick_params(direction="in")
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    pdf_path = ensure_outdir(p) / "qb_paramp_optionB_squeezed_scan.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved squeezed-state scan figure to:\n  {pdf_path}")

    # Time-trace comparison for selected states, including coherent reference.
    fig2, ax = plt.subplots(figsize=(7.4, 4.6))
    if "coherent" in results_by_kind:
        r0 = results_by_kind["coherent"]
        ax.plot(r0.tau_grid, r0.Imin, label="coherent reference")
    for kind, res in results_by_kind.items():
        if kind == "coherent":
            continue
        sq = parse_squeezed_kind(kind)
        if sq is None and kind != "squeezed_vacuum":
            continue
        ax.plot(res.tau_grid, res.Imin, label=label_for_kind(kind, p))
    ax.axhline(1.0, linestyle="-.", label=r"$I_{\min}^{(f)}=1$")
    ax.axvline(tau_star, color="0.5", linestyle=":", linewidth=1.0)
    ax.set_xlabel(r"readout time $\tau=\lambda T$")
    ax.set_ylabel(r"$I_{\min}^{(f)}(\tau)$")
    ax.set_title("squeezed pump temporal-mode certifier")
    ax.tick_params(direction="in")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig2.tight_layout()
    pdf_path2 = ensure_outdir(p) / "qb_paramp_optionB_squeezed_time_traces.pdf"
    fig2.savefig(pdf_path2, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved squeezed-state time traces to:\n  {pdf_path2}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def make_params_from_args(args: argparse.Namespace) -> Params:
    return Params(
        nbar=args.nbar,
        Na=args.Na,
        Nb=args.Nb,
        Nc=args.Nc,
        NAf=args.NAf,
        NBf=args.NBf,
        lam=args.lam,
        kappa_a=args.kappa_a,
        kappa_b=args.kappa_b,
        kappa_c=args.kappa_c,
        gamma_A=args.gamma_A,
        gamma_B=args.gamma_B,
        eta_partial=args.eta_partial,
        tau_max=args.tau_max,
        Nt=args.Nt,
        nsteps=args.nsteps,
        atol=args.atol,
        rtol=args.rtol,
        outdir=args.outdir,
        max_mesolve_dim=args.max_mesolve_dim,
    )


def parse_float_list(s: str) -> np.ndarray:
    return np.array([float(x.strip()) for x in s.split(",") if x.strip()], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Option B cascaded-collector simulation for QB-powered parametric amplifier")
    parser.add_argument("--mode", choices=["basic", "threshold", "gain", "nbar", "gamma", "squeezed", "all"], default="basic")

    parser.add_argument("--nbar", type=float, default=Params.nbar)
    parser.add_argument("--Na", type=int, default=Params.Na)
    parser.add_argument("--Nb", type=int, default=Params.Nb)
    parser.add_argument("--Nc", type=int, default=Params.Nc)
    parser.add_argument("--NAf", type=int, default=Params.NAf)
    parser.add_argument("--NBf", type=int, default=Params.NBf)
    parser.add_argument("--lam", type=float, default=Params.lam)

    parser.add_argument("--kappa_a", type=float, default=Params.kappa_a)
    parser.add_argument("--kappa_b", type=float, default=Params.kappa_b)
    parser.add_argument("--kappa_c", type=float, default=Params.kappa_c)
    parser.add_argument("--gamma_A", type=float, default=Params.gamma_A)
    parser.add_argument("--gamma_B", type=float, default=Params.gamma_B)
    parser.add_argument("--eta_partial", type=float, default=Params.eta_partial)

    parser.add_argument("--tau_max", type=float, default=Params.tau_max)
    parser.add_argument("--Nt", type=int, default=Params.Nt)
    parser.add_argument("--nsteps", type=int, default=Params.nsteps)
    parser.add_argument("--atol", type=float, default=Params.atol)
    parser.add_argument("--rtol", type=float, default=Params.rtol)
    parser.add_argument("--outdir", type=str, default=Params.outdir)
    parser.add_argument("--max_mesolve_dim", type=int, default=Params.max_mesolve_dim)

    parser.add_argument("--kinds", type=str, default="coherent,partial,phase_randomized,fock")
    parser.add_argument("--n_eta", type=int, default=101)
    parser.add_argument("--rho_values", type=str, default="0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.92")
    parser.add_argument("--gamma_values", type=str, default="0.3,0.5,0.75,1.0,1.5,2.0,3.0")
    parser.add_argument("--nbar_values", type=str, default="1.5,2.0,3.0,5.0,8.0,10.0,12.0")
    parser.add_argument("--rho_fixed", type=float, default=0.8)
    parser.add_argument("--squeeze_r_values", type=str, default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--squeeze_angles", type=str, default="amp,phase")
    parser.add_argument("--include_squeezed_vacuum", action="store_true")

    args = parser.parse_args()
    p = make_params_from_args(args)

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    if int(round(p.nbar)) >= p.Nc and "fock" in kinds:
        print("Warning: Fock pump does not fit in Nc cutoff. Removing Fock case.")
        kinds = [k for k in kinds if k != "fock"]

    if args.mode in {"basic", "all"}:
        run_basic(p, kinds)
    if args.mode in {"threshold", "all"}:
        run_threshold(p, n_eta=args.n_eta)
    if args.mode in {"gain", "all"}:
        run_gain_scan(p, parse_float_list(args.rho_values), n_eta=args.n_eta)
    if args.mode in {"gamma", "all"}:
        run_gamma_scan(p, parse_float_list(args.gamma_values), n_eta=args.n_eta)
    if args.mode in {"squeezed", "all"}:
        squeeze_angles = [a.strip() for a in args.squeeze_angles.split(",") if a.strip()]
        run_squeezed_scan(
            p,
            parse_float_list(args.squeeze_r_values),
            squeeze_angles,
            include_squeezed_vacuum=args.include_squeezed_vacuum,
        )
    if args.mode in {"nbar", "all"}:
        run_nbar_scan(p, parse_float_list(args.nbar_values), rho_fixed=args.rho_fixed, n_eta=args.n_eta)


if __name__ == "__main__":
    main()
