"""
Seeded-signal gain and added-noise test for the finite quantum-pump / quantum-battery
nondegenerate parametric amplifier.

This script is a referee-response extension of the Option-B cascaded-collector model.
It keeps the same five quantum modes used for the submitted calculations:

    a : signal resonator
    b : idler resonator
    c : finite pump / quantum-battery mode
    A : downstream signal-output temporal collector
    B : downstream idler-output temporal collector

The source Hamiltonian is

    H_I = i g (c a^dag b^dag - c^dag a b),

and the source -> collector cascade uses

    L_tot = L_source + L_collector,
    H_cas = (i/2)(L_source^dag L_collector - L_collector^dag L_source).

NEW IN THIS SCRIPT
------------------
Memory note: all large operators, states, and collapse operators are explicitly
converted to QuTiP CSR storage before mesolve.  This avoids QuTiP 5.0.x DIA
Liouvillian allocations that can otherwise require tens of gigabytes at D=8750.

A weak coherent traveling input field is injected into the signal input channel.  In
input-output notation its amplitude beta_in has units sqrt(rate).  Because the signal
source and its downstream collector form one cascaded network, the coherent input must
drive the *network input channel*, whose total coupling operator is

    L_tot,a = sqrt(kappa_a) a + sqrt(gamma_A) A.

For the convention a_out = a_in + sqrt(kappa_a) a, the coherent-input displacement is
implemented by

    H_seed = i (beta_in^* L_tot,a - beta_in L_tot,a^dag).

This includes both the direct coherent field reaching the collector and the field
scattered by the signal resonator.  Driving only sqrt(kappa_a) a would omit the direct
traveling-wave contribution and would therefore give the wrong collector reference.

At readout time T, a constant coherent input beta_in over [0,T] has total incident
photon number |beta_in|^2 T.  Because the collector is an exponentially filtered
canonical mode, this top-hat photon number is not itself the correctly mode-matched
reference for a finite-time gain calibration.  We therefore keep

    G_top_hat(T) = |<A>|^2 / (|beta_in|^2 T)

only as a diagnostic of finite-window collection.

For the amplifier characterization used in the referee response, the physically robust
quantity is the pump-induced coherent power gain calibrated against the *same* finite
input pulse, passive source cavity, collector filter, and readout time with the pump
interaction switched off:

    G_coh(T) = |<A>|_on^2 / |<A>|_off^2.

Thus G_coh=1 for the passive network by construction, and any finite-time mode mismatch
or passive collection efficiency cancels from the amplification factor.

The incoherent output population is

    N_inc(T) = <A^dag A> - |<A>|^2.

With X_A=(A+A^dag)/sqrt(2) and P_A=(A-A^dag)/(i sqrt(2)), the mean symmetrized
quadrature variance is

    Vbar_A = [Var(X_A)+Var(P_A)]/2 = N_inc + 1/2.

Using the passive coherent response as the calibrated input reference, we define the
passive-calibrated input-referred added noise

    N_add(T) = Vbar_A(T)/G_coh(T) - 1/2.

For G_coh>1, the corresponding finite-gain phase-preserving quantum limit is

    N_add,QL = (1 - 1/G_coh)/2.

The script also reports

    eta_passive(T) = |<A>|_off^2 / (|beta_in|^2 T),

which quantifies the finite-window passive collection factor.  This factor is *not*
interpreted as amplifier gain; it is retained only as a diagnostic of the selected
input/output temporal-mode pair.

The partially dephased pump state is obtained exactly by linear mixing of raw moments
from coherent and phase-randomized pump runs, avoiding an unnecessary extra mesolve.

Outputs
-------
  seeded_gain_noise_timeseries_<kind>.csv
  seeded_gain_noise_summary.csv
  seeded_linearity.csv
  qb_seeded_gain_noise_vs_time.pdf
  qb_seeded_gain_noise_summary.pdf
  qb_seeded_linearity.pdf

Recommended production parameters matching the submitted main data set:

  nbar=5, Na=5, Nb=5, Nc=14, NAf=5, NBf=5,
  gamma_A=gamma_B=2, tau_max=2, Nt=61, tau_eval=1.3666667.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import argparse
import csv
import time
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

@dataclass
class Params:
    nbar: float = 5.0

    Na: int = 5
    Nb: int = 5
    Nc: int = 14
    NAf: int = 5
    NBf: int = 5

    lam: float = 1.0

    kappa_a: float = 2.5
    kappa_b: float = 2.5
    kappa_c: float = 0.02

    gamma_A: float = 2.0
    gamma_B: float = 2.0

    eta_partial: float = 0.25

    tau_max: float = 2.0
    Nt: int = 61
    tau_eval: float = 1.3666666667

    nsteps: int = 50000
    atol: float = 1e-8
    rtol: float = 1e-7

    max_mesolve_dim: int = 15000
    outdir: str = "qb_seeded_gain_noise"

    @property
    def g(self) -> float:
        return self.lam / np.sqrt(self.nbar)

    @property
    def xi(self) -> float:
        return 2.0 * self.lam / np.sqrt(self.kappa_a * self.kappa_b)

    @property
    def hilbert_dim(self) -> int:
        return self.Na * self.Nb * self.Nc * self.NAf * self.NBf


# -----------------------------------------------------------------------------
# Pump states
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
    # Force CSR: the full five-mode Liouvillian is enormous, and QuTiP 5's
    # DIA representation can allocate one full Liouville-space vector per
    # diagonal.  CSR stores only actual nonzero entries.
    return qt.ket2dm(qt.coherent(N, np.sqrt(nbar))).to("csr")


def phase_randomized_coherent_dm(N: int, nbar: float) -> qt.Qobj:
    probs = poisson_probs(N, nbar)
    return qt.Qobj(np.diag(probs), dims=[[N], [N]]).to("csr")


def fock_dm(N: int, n: int) -> qt.Qobj:
    if n >= N:
        raise ValueError(
            f"Fock index n={n} does not fit in cutoff N={N}. Increase Nc or lower nbar."
        )
    return qt.ket2dm(qt.basis(N, n)).to("csr")


def single_mode_eta(rho: qt.Qobj) -> float:
    N = rho.shape[0]
    if N <= 1:
        return 0.0
    c = qt.destroy(N)
    n_mean = float(np.real(qt.expect(c.dag() * c, rho)))
    if n_mean <= 1e-14:
        return 0.0
    c_mean = complex(qt.expect(c, rho))
    return abs(c_mean) ** 2 / n_mean


def pump_state_dm(kind: str, p: Params) -> tuple[qt.Qobj, float]:
    if kind == "coherent":
        rho = coherent_dm(p.Nc, p.nbar)
    elif kind == "phase_randomized":
        rho = phase_randomized_coherent_dm(p.Nc, p.nbar)
    elif kind == "partial":
        w = np.sqrt(p.eta_partial)
        rho_coh = coherent_dm(p.Nc, p.nbar)
        rho_pr = phase_randomized_coherent_dm(p.Nc, p.nbar)
        rho = w * rho_coh + (1.0 - w) * rho_pr
        rho = rho / rho.tr()
    elif kind == "fock":
        rho = fock_dm(p.Nc, int(round(p.nbar)))
    elif kind == "vacuum":
        rho = qt.ket2dm(qt.basis(p.Nc, 0)).to("csr")
    else:
        raise ValueError(f"Unknown pump kind: {kind}")
    return rho, single_mode_eta(rho)


# -----------------------------------------------------------------------------
# Tensor construction and cascaded model
# -----------------------------------------------------------------------------

def tensor_op(single_op: qt.Qobj, slot: int, dims: list[int]) -> qt.Qobj:
    # Keep all large operators in compressed-sparse-row form.  This is crucial
    # for QuTiP 5.0.x at D~10^4: a DIA Liouvillian with O(10--30) diagonals
    # would try to allocate tens of gigabytes even though the operator is sparse.
    ops = [qt.qeye(d).to("csr") for d in dims]
    ops[slot] = single_op.to("csr")
    return qt.tensor(*ops).to("csr")


@dataclass
class Operators:
    a: qt.Qobj
    b: qt.Qobj
    c: qt.Qobj
    A: qt.Qobj
    B: qt.Qobj


def build_operators(p: Params) -> Operators:
    dims = [p.Na, p.Nb, p.Nc, p.NAf, p.NBf]
    return Operators(
        a=tensor_op(qt.destroy(p.Na), 0, dims),
        b=tensor_op(qt.destroy(p.Nb), 1, dims),
        c=tensor_op(qt.destroy(p.Nc), 2, dims),
        A=tensor_op(qt.destroy(p.NAf), 3, dims),
        B=tensor_op(qt.destroy(p.NBf), 4, dims),
    )


def build_cascaded_model(
    p: Params,
    beta_in: complex,
    g_override: Optional[float] = None,
) -> tuple[qt.Qobj, list[qt.Qobj], Operators]:
    op = build_operators(p)
    a, b, c, A, B = op.a, op.b, op.c, op.A, op.B

    g_eff = p.g if g_override is None else float(g_override)
    H_source = 1j * g_eff * (c * a.dag() * b.dag() - c.dag() * a * b)

    # Signal cascade: source a -> collector A.
    Ls_a = np.sqrt(p.kappa_a) * a
    Lc_A = np.sqrt(p.gamma_A) * A
    H_cas_a = 0.5j * (Ls_a.dag() * Lc_A - Lc_A.dag() * Ls_a)
    Ltot_a = Ls_a + Lc_A

    # Idler cascade: source b -> collector B.
    Ls_b = np.sqrt(p.kappa_b) * b
    Lc_B = np.sqrt(p.gamma_B) * B
    H_cas_b = 0.5j * (Ls_b.dag() * Lc_B - Lc_B.dag() * Ls_b)
    Ltot_b = Ls_b + Lc_B

    H = H_source + H_cas_a + H_cas_b

    # Coherent input displacement of the SIGNAL NETWORK INPUT channel.
    # For a_out = a_in + sqrt(kappa_a) a, this convention gives the correct
    # traveling coherent input.  The use of Ltot_a (not merely Ls_a) is essential:
    # it includes the direct coherent field propagating from the network input to A.
    if abs(beta_in) > 0.0:
        H_seed = 1j * (np.conjugate(beta_in) * Ltot_a - beta_in * Ltot_a.dag())
        H = H + H_seed

    # CRITICAL MEMORY SAFEGUARD FOR QUTIP 5.0.x:
    # Convert the Hamiltonian and collapse operators to CSR *before* mesolve
    # constructs the Liouvillian.  Otherwise the extra seed-drive diagonals can
    # make QuTiP keep the superoperator in DIA format, which allocates a full
    # D^2-length array for every diagonal.
    H = H.to("csr")
    Ltot_a = Ltot_a.to("csr")
    Ltot_b = Ltot_b.to("csr")
    c_ops: list[qt.Qobj] = [Ltot_a, Ltot_b]
    if p.kappa_c > 0.0:
        c_ops.append((np.sqrt(p.kappa_c) * c).to("csr"))

    return H, c_ops, op


def initial_state(kind: str, p: Params) -> tuple[qt.Qobj, float]:
    rho_a = qt.ket2dm(qt.basis(p.Na, 0)).to("csr")
    rho_b = qt.ket2dm(qt.basis(p.Nb, 0)).to("csr")
    rho_c, eta = pump_state_dm(kind, p)
    rho_A = qt.ket2dm(qt.basis(p.NAf, 0)).to("csr")
    rho_B = qt.ket2dm(qt.basis(p.NBf, 0)).to("csr")
    rho0 = qt.tensor(rho_a, rho_b, rho_c.to("csr"), rho_A, rho_B).to("csr")
    return rho0, eta


# -----------------------------------------------------------------------------
# Raw seeded result
# -----------------------------------------------------------------------------

@dataclass
class SeededRawResult:
    label: str
    eta_in: float
    beta_in: complex
    tlist: np.ndarray
    tau_grid: np.ndarray

    A_mean: np.ndarray
    A2: np.ndarray
    N_A: np.ndarray
    B_mean: np.ndarray
    N_B: np.ndarray
    M_AB: np.ndarray

    n_a: np.ndarray
    n_b: np.ndarray
    n_c: np.ndarray

    edge_a: np.ndarray
    edge_b: np.ndarray
    edge_c: np.ndarray
    edge_A: np.ndarray
    edge_B: np.ndarray


def cutoff_projector(N: int, slot: int, dims: list[int]) -> qt.Qobj:
    edge = qt.basis(N, N - 1)
    return tensor_op(edge * edge.dag(), slot, dims)


def solve_seeded_case(
    kind: str,
    p: Params,
    beta_in: complex,
    g_override: Optional[float] = None,
) -> SeededRawResult:
    if p.hilbert_dim > p.max_mesolve_dim:
        raise ValueError(
            f"Hilbert dimension D={p.hilbert_dim} exceeds --max_mesolve_dim={p.max_mesolve_dim}."
        )

    H, c_ops, op = build_cascaded_model(p, beta_in=beta_in, g_override=g_override)
    rho0, eta_in = initial_state(kind, p)

    tmax = p.tau_max / p.lam
    tlist = np.linspace(0.0, tmax, p.Nt)
    tau_grid = p.lam * tlist

    a, b, c, A, B = op.a, op.b, op.c, op.A, op.B
    dims = [p.Na, p.Nb, p.Nc, p.NAf, p.NBf]

    e_ops = [
        A,
        A * A,
        A.dag() * A,
        B,
        B.dag() * B,
        A * B,
        a.dag() * a,
        b.dag() * b,
        c.dag() * c,
        cutoff_projector(p.Na, 0, dims),
        cutoff_projector(p.Nb, 1, dims),
        cutoff_projector(p.Nc, 2, dims),
        cutoff_projector(p.NAf, 3, dims),
        cutoff_projector(p.NBf, 4, dims),
    ]
    e_ops = [e.to("csr") for e in e_ops]

    options = {
        "nsteps": p.nsteps,
        "atol": p.atol,
        "rtol": p.rtol,
        "store_states": False,
    }

    print(
        f"  mesolve {kind}, beta={beta_in:.6g}: D={p.hilbert_dim}, "
        f"Liouville~{p.hilbert_dim**2:.3e}, Nt={p.Nt}"
    )
    print(f"    data layers: H={H.dtype.__name__}, rho0={rho0.dtype.__name__}, "
          f"c_ops={[x.dtype.__name__ for x in c_ops]}")
    tic = time.time()
    sol = qt.mesolve(H, rho0, tlist, c_ops, e_ops=e_ops, options=options)
    toc = time.time()
    print(f"    finished in {toc - tic:.1f} s")

    def carray(i: int) -> np.ndarray:
        return np.asarray(sol.expect[i], dtype=complex)

    def rarray(i: int) -> np.ndarray:
        return np.real(carray(i))

    result = SeededRawResult(
        label=kind,
        eta_in=eta_in,
        beta_in=complex(beta_in),
        tlist=tlist,
        tau_grid=tau_grid,
        A_mean=carray(0),
        A2=carray(1),
        N_A=rarray(2),
        B_mean=carray(3),
        N_B=rarray(4),
        M_AB=carray(5),
        n_a=rarray(6),
        n_b=rarray(7),
        n_c=rarray(8),
        edge_a=rarray(9),
        edge_b=rarray(10),
        edge_c=rarray(11),
        edge_A=rarray(12),
        edge_B=rarray(13),
    )

    max_src = max(result.edge_a.max(), result.edge_b.max(), result.edge_c.max())
    max_col = max(result.edge_A.max(), result.edge_B.max())
    print(f"    edge max source/collector={max_src:.2e}/{max_col:.2e}")

    return result


def mix_raw_results(
    coherent: SeededRawResult,
    phase_randomized: SeededRawResult,
    eta_partial: float,
) -> SeededRawResult:
    """Exact linear mixing at the density-matrix / raw-moment level."""
    if len(coherent.tlist) != len(phase_randomized.tlist):
        raise ValueError("Cannot mix results with different time grids.")
    if abs(coherent.beta_in - phase_randomized.beta_in) > 1e-14:
        raise ValueError("Cannot mix results with different coherent input amplitudes.")

    w = np.sqrt(eta_partial)

    def mix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return w * x + (1.0 - w) * y

    fields = {}
    for name in [
        "A_mean", "A2", "N_A", "B_mean", "N_B", "M_AB",
        "n_a", "n_b", "n_c", "edge_a", "edge_b", "edge_c", "edge_A", "edge_B",
    ]:
        fields[name] = mix(getattr(coherent, name), getattr(phase_randomized, name))

    # The exact eta_in of the finite-cutoff mixed initial state is most safely
    # reconstructed from the same finite-cutoff pump density matrices.
    # Since this helper has no Params object, eta_partial is used as the intended label;
    # the production value differs from it only by the already documented cutoff effect.
    return SeededRawResult(
        label="partial",
        eta_in=float(eta_partial),
        beta_in=coherent.beta_in,
        tlist=coherent.tlist.copy(),
        tau_grid=coherent.tau_grid.copy(),
        **fields,
    )


# -----------------------------------------------------------------------------
# Derived gain / noise metrics
# -----------------------------------------------------------------------------

@dataclass
class SeededMetrics:
    label: str
    eta_in: float
    beta_in: complex
    tlist: np.ndarray
    tau_grid: np.ndarray

    N_in: np.ndarray
    N_sig: np.ndarray
    N_inc: np.ndarray
    signal_fraction: np.ndarray

    Vx: np.ndarray
    Vp: np.ndarray
    Vbar: np.ndarray

    G_mode: np.ndarray
    eta_passive: np.ndarray
    G_over_passive: np.ndarray
    N_add_top_hat: np.ndarray
    N_add: np.ndarray
    N_add_ql: np.ndarray

    Imin: np.ndarray
    C_pair: np.ndarray

    max_edge_source: float
    max_edge_collector: float


def safe_divide(num: np.ndarray, den: np.ndarray, min_den: float = 1e-14) -> np.ndarray:
    out = np.full(np.broadcast_shapes(np.shape(num), np.shape(den)), np.nan, dtype=float)
    num_arr = np.broadcast_to(np.asarray(num, dtype=float), out.shape)
    den_arr = np.broadcast_to(np.asarray(den, dtype=float), out.shape)
    mask = np.abs(den_arr) > min_den
    out[mask] = num_arr[mask] / den_arr[mask]
    return out


def derive_metrics(raw: SeededRawResult, passive: SeededRawResult) -> SeededMetrics:
    if len(raw.tlist) != len(passive.tlist):
        raise ValueError("Raw and passive reference time grids differ.")

    beta2 = abs(raw.beta_in) ** 2
    N_in = beta2 * raw.tlist

    N_sig = np.abs(raw.A_mean) ** 2
    N_inc = np.real(raw.N_A - N_sig)
    # Clip only tiny negative roundoff, while retaining a warning below for real failures.
    N_inc_clean = np.where((N_inc < 0.0) & (N_inc > -1e-10), 0.0, N_inc)

    centered_A2 = raw.A2 - raw.A_mean ** 2
    Vx = N_inc_clean + 0.5 + np.real(centered_A2)
    Vp = N_inc_clean + 0.5 - np.real(centered_A2)
    Vbar = 0.5 * (Vx + Vp)

    G_mode = safe_divide(N_sig, N_in)

    N_sig_off = np.abs(passive.A_mean) ** 2
    eta_passive = safe_divide(N_sig_off, N_in)
    G_over_passive = safe_divide(N_sig, N_sig_off)

    # G_mode uses the total top-hat incident photon number and is retained only as
    # a finite-window collection diagnostic.  The amplifier gain used for noise
    # calibration is G_over_passive: pump-on coherent output divided by the
    # pump-off coherent output for the identical pulse/filter/readout protocol.
    N_add_top_hat = safe_divide(Vbar, G_mode) - 0.5
    N_add = safe_divide(Vbar, G_over_passive) - 0.5
    N_add_ql = np.full_like(G_over_passive, np.nan)
    mask_amp = np.isfinite(G_over_passive) & (G_over_passive > 1.0 + 1e-10)
    N_add_ql[mask_amp] = 0.5 * (1.0 - 1.0 / G_over_passive[mask_amp])

    signal_fraction = safe_divide(N_sig, raw.N_A)

    denom = raw.N_A + raw.N_B + 1.0
    Imin = denom - 2.0 * np.abs(raw.M_AB)
    n_pair = 0.5 * (raw.N_A + raw.N_B)
    C_pair = np.zeros_like(n_pair)
    mask_pair = n_pair > 1e-14
    C_pair[mask_pair] = np.abs(raw.M_AB[mask_pair]) / np.sqrt(
        n_pair[mask_pair] * (n_pair[mask_pair] + 1.0)
    )

    max_edge_source = max(raw.edge_a.max(), raw.edge_b.max(), raw.edge_c.max())
    max_edge_collector = max(raw.edge_A.max(), raw.edge_B.max())

    if np.nanmin(N_inc) < -1e-8:
        print(
            f"    WARNING {raw.label}: N_inc has negative values down to {np.nanmin(N_inc):.3e}; "
            "increase cutoffs or tighten solver tolerances."
        )
    if np.nanmin(Vx) < -1e-8 or np.nanmin(Vp) < -1e-8:
        print(f"    WARNING {raw.label}: negative quadrature variance detected.")

    return SeededMetrics(
        label=raw.label,
        eta_in=raw.eta_in,
        beta_in=raw.beta_in,
        tlist=raw.tlist,
        tau_grid=raw.tau_grid,
        N_in=N_in,
        N_sig=N_sig,
        N_inc=N_inc_clean,
        signal_fraction=signal_fraction,
        Vx=Vx,
        Vp=Vp,
        Vbar=Vbar,
        G_mode=G_mode,
        eta_passive=eta_passive,
        G_over_passive=G_over_passive,
        N_add_top_hat=N_add_top_hat,
        N_add=N_add,
        N_add_ql=N_add_ql,
        Imin=Imin,
        C_pair=C_pair,
        max_edge_source=float(max_edge_source),
        max_edge_collector=float(max_edge_collector),
    )


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def ensure_outdir(p: Params) -> Path:
    outdir = Path.cwd() / p.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def set_plot_style() -> None:
    # Larger fonts than the submitted figures, directly addressing the referee comment.
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 15,
        "legend.fontsize": 12,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })


def label_for_kind(kind: str, p: Params) -> str:
    return {
        "coherent": r"coherent",
        "partial": rf"partially dephased, $\eta_{{\rm c}}^{{\rm in}}\simeq {p.eta_partial:.2f}$",
        "phase_randomized": r"phase randomized",
        "fock": r"Fock",
    }.get(kind, kind)


def eval_index(tau_grid: np.ndarray, tau_eval: float) -> int:
    return int(np.argmin(np.abs(tau_grid - tau_eval)))


def write_timeseries_csv(metrics: SeededMetrics, p: Params) -> Path:
    out = ensure_outdir(p) / f"seeded_gain_noise_timeseries_{metrics.label}.csv"
    fields = [
        "tau", "t", "N_in", "N_sig", "N_inc", "signal_fraction",
        "Vx", "Vp", "Vbar", "G_mode", "eta_passive", "G_over_passive",
        "N_add_top_hat", "N_add", "N_add_ql", "Imin", "C_pair",
    ]
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for i in range(len(metrics.tau_grid)):
            w.writerow([
                metrics.tau_grid[i], metrics.tlist[i], metrics.N_in[i], metrics.N_sig[i],
                metrics.N_inc[i], metrics.signal_fraction[i], metrics.Vx[i], metrics.Vp[i],
                metrics.Vbar[i], metrics.G_mode[i], metrics.eta_passive[i],
                metrics.G_over_passive[i], metrics.N_add_top_hat[i], metrics.N_add[i],
                metrics.N_add_ql[i], metrics.Imin[i], metrics.C_pair[i],
            ])
    return out


def summary_row(metrics: SeededMetrics, p: Params) -> dict[str, Union[float, str]]:
    idx = eval_index(metrics.tau_grid, p.tau_eval)
    return {
        "pump_state": metrics.label,
        "eta_in": metrics.eta_in,
        "beta_abs": abs(metrics.beta_in),
        "tau_eval": metrics.tau_grid[idx],
        "N_in": metrics.N_in[idx],
        "N_sig": metrics.N_sig[idx],
        "N_inc": metrics.N_inc[idx],
        "signal_fraction": metrics.signal_fraction[idx],
        "Vx": metrics.Vx[idx],
        "Vp": metrics.Vp[idx],
        "Vbar": metrics.Vbar[idx],
        "G_mode": metrics.G_mode[idx],
        "eta_passive": metrics.eta_passive[idx],
        "G_over_passive": metrics.G_over_passive[idx],
        "N_add_top_hat": metrics.N_add_top_hat[idx],
        "N_add": metrics.N_add[idx],
        "N_add_ql": metrics.N_add_ql[idx],
        "Imin": metrics.Imin[idx],
        "C_pair": metrics.C_pair[idx],
        "max_edge_source": metrics.max_edge_source,
        "max_edge_collector": metrics.max_edge_collector,
    }


def write_summary_csv(rows: list[dict[str, Union[float, str]]], p: Params) -> Path:
    out = ensure_outdir(p) / "seeded_gain_noise_summary.csv"
    if not rows:
        return out
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return out


def plot_time_traces(metrics_map: dict[str, SeededMetrics], p: Params) -> Path:
    set_plot_style()
    fig, axs = plt.subplots(2, 2, figsize=(10.0, 7.4))

    for kind, m in metrics_map.items():
        label = label_for_kind(kind, p)
        tau = m.tau_grid
        axs[0, 0].plot(tau, m.G_over_passive, label=label)
        axs[0, 1].plot(tau, m.N_add, label=label)
        axs[1, 0].plot(tau, m.N_sig, label=label)
        axs[1, 1].plot(tau, m.N_inc, label=label)

    axs[0, 0].axhline(1.0, linestyle="--", linewidth=1.0)
    axs[0, 0].set_title("passive-calibrated coherent power gain")
    axs[0, 0].set_ylabel(r"$G_{\rm coh}$")

    axs[0, 1].set_title("input-referred added noise")
    axs[0, 1].set_ylabel(r"$N_{\rm add}$")

    axs[1, 0].set_title("coherent output signal")
    axs[1, 0].set_ylabel(r"$|\langle A\rangle|^2$")

    axs[1, 1].set_title("incoherent output population")
    axs[1, 1].set_ylabel(r"$\langle A^\dagger A\rangle-|\langle A\rangle|^2$")

    for ax in axs.flat:
        ax.axvline(p.tau_eval, linestyle=":", linewidth=1.0)
        ax.set_xlabel(r"$\tau=\lambda T$")
        ax.tick_params(direction="in")
        ax.legend(frameon=False)

    fig.tight_layout()
    out = ensure_outdir(p) / "qb_seeded_gain_noise_vs_time.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_summary(metrics_map: dict[str, SeededMetrics], p: Params) -> Path:
    set_plot_style()
    kinds = list(metrics_map.keys())
    labels = [
        {"coherent": "coherent", "partial": "partial", "phase_randomized": "phase\nrandomized", "fock": "Fock"}.get(k, k)
        for k in kinds
    ]

    G = []
    Nadd = []
    Nql = []
    Ninc = []
    for k in kinds:
        m = metrics_map[k]
        idx = eval_index(m.tau_grid, p.tau_eval)
        G.append(m.G_over_passive[idx])
        Nadd.append(m.N_add[idx])
        Nql.append(m.N_add_ql[idx])
        Ninc.append(m.N_inc[idx])

    x = np.arange(len(kinds))
    fig, axs = plt.subplots(1, 3, figsize=(12.4, 4.2))

    axs[0].bar(x, G)
    axs[0].axhline(1.0, linestyle="--", linewidth=1.0)
    axs[0].set_ylabel(r"$G_{\rm coh}$")
    axs[0].set_title("passive-calibrated coherent gain")

    axs[1].bar(x, Nadd)
    # Plot QL points only where the selected temporal-mode gain exceeds unity.
    for i, q in enumerate(Nql):
        if np.isfinite(q):
            axs[1].plot(i, q, marker="o", linestyle="None")
    axs[1].set_ylabel(r"$N_{\rm add}$")
    axs[1].set_title("added noise")

    axs[2].bar(x, Ninc)
    axs[2].set_ylabel(r"$N_{\rm inc}$")
    axs[2].set_title("incoherent output")

    for ax in axs:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.tick_params(direction="in")

    fig.suptitle(rf"seeded amplifier at $\tau={p.tau_eval:.3f}$", y=1.02)
    fig.tight_layout()
    out = ensure_outdir(p) / "qb_seeded_gain_noise_summary.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_linearity(rows: list[dict[str, float]], p: Params) -> Path:
    set_plot_style()
    beta = np.array([r["seed_amp"] for r in rows], dtype=float)
    G = np.array([r["G_over_passive"] for r in rows], dtype=float)
    Nadd = np.array([r["N_add"] for r in rows], dtype=float)
    Ninc = np.array([r["N_inc"] for r in rows], dtype=float)

    fig, axs = plt.subplots(1, 3, figsize=(12.4, 4.0))
    axs[0].plot(beta, G, "o-")
    axs[0].set_xlabel(r"seed amplitude $|\beta_{\rm in}|/\sqrt{\lambda}$")
    axs[0].set_ylabel(r"$G_{\rm coh}$")
    axs[0].set_title("gain linearity")

    axs[1].plot(beta, Nadd, "o-")
    axs[1].set_xlabel(r"seed amplitude $|\beta_{\rm in}|/\sqrt{\lambda}$")
    axs[1].set_ylabel(r"$N_{\rm add}$")
    axs[1].set_title("noise stability")

    axs[2].plot(beta, Ninc, "o-")
    axs[2].set_xlabel(r"seed amplitude $|\beta_{\rm in}|/\sqrt{\lambda}$")
    axs[2].set_ylabel(r"$N_{\rm inc}$")
    axs[2].set_title("background stability")

    for ax in axs:
        ax.tick_params(direction="in")

    fig.tight_layout()
    out = ensure_outdir(p) / "qb_seeded_linearity.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# -----------------------------------------------------------------------------
# Main seeded-noise run
# -----------------------------------------------------------------------------

def print_params(p: Params, seed_amp: float, seed_phase: float) -> None:
    print("Seeded finite-pump amplifier: coherent gain and added-noise test")
    print(f"QuTiP version = {qt.__version__}")
    print(f"nbar = {p.nbar:.6f}")
    print(f"cutoffs: Na={p.Na}, Nb={p.Nb}, Nc={p.Nc}, NAf={p.NAf}, NBf={p.NBf}")
    print(f"Hilbert D = {p.hilbert_dim}; Liouville D^2 ~ {p.hilbert_dim**2:.3e}")
    print(f"lambda = {p.lam:.6f}, g = {p.g:.6f}, xi = {p.xi:.6f}")
    print(f"kappa_a/b/c = {p.kappa_a:.6f}/{p.kappa_b:.6f}/{p.kappa_c:.6f}")
    print(f"gamma_A/B = {p.gamma_A:.6f}/{p.gamma_B:.6f}")
    print(f"tau_max = {p.tau_max:.6f}, Nt = {p.Nt}, tau_eval = {p.tau_eval:.7f}")
    print(f"main seed amplitude = {seed_amp:.6f} sqrt(lambda), phase = {seed_phase:.6f} rad")
    print()


def passive_reference(p: Params, beta_in: complex) -> SeededRawResult:
    # The pump, idler and idler collector are decoupled when g=0.  Reducing those
    # dimensions makes this passive calibration essentially free while preserving the
    # exact signal-source / signal-collector network.
    p_off = replace(p, Nb=2, Nc=2, NBf=2, max_mesolve_dim=max(p.max_mesolve_dim, 1000))
    print("Passive pump-off reference (g=0):")
    raw = solve_seeded_case("vacuum", p_off, beta_in=beta_in, g_override=0.0)
    print()
    return raw


def rescale_passive_beta(passive: SeededRawResult, beta_new: complex) -> SeededRawResult:
    """Use exact linear scaling of the passive coherent response for a new beta.

    Number moments of a passive coherent network scale as |beta|^2 and coherent
    amplitudes as beta.  The *full* A2 includes the coherent square.  Vacuum variance
    remains exactly 1/2; scaling these raw moments is sufficient for the gain reference,
    but derive_metrics only uses passive.A_mean, so the remaining fields are copied.
    """
    if abs(passive.beta_in) <= 0:
        raise ValueError("Passive reference beta is zero; cannot rescale.")
    r = beta_new / passive.beta_in
    rr = abs(r) ** 2
    return SeededRawResult(
        label="vacuum",
        eta_in=0.0,
        beta_in=beta_new,
        tlist=passive.tlist.copy(),
        tau_grid=passive.tau_grid.copy(),
        A_mean=r * passive.A_mean,
        A2=(r ** 2) * passive.A2,
        N_A=rr * passive.N_A,
        B_mean=passive.B_mean.copy(),
        N_B=passive.N_B.copy(),
        M_AB=passive.M_AB.copy(),
        n_a=rr * passive.n_a,
        n_b=passive.n_b.copy(),
        n_c=passive.n_c.copy(),
        edge_a=passive.edge_a.copy(),
        edge_b=passive.edge_b.copy(),
        edge_c=passive.edge_c.copy(),
        edge_A=passive.edge_A.copy(),
        edge_B=passive.edge_B.copy(),
    )


def run_seeded_noise(
    p: Params,
    seed_amp: float,
    seed_phase: float,
    linearity_seeds: np.ndarray,
    kinds: list[str],
) -> None:
    print_params(p, seed_amp, seed_phase)

    if p.hilbert_dim > p.max_mesolve_dim:
        raise ValueError(
            f"Main Hilbert dimension {p.hilbert_dim} exceeds max_mesolve_dim={p.max_mesolve_dim}."
        )

    beta_main = seed_amp * np.sqrt(p.lam) * np.exp(1j * seed_phase)
    passive_main = passive_reference(p, beta_main)

    # Sanity check: a passive network driven by a coherent input must leave the
    # collector in a coherent state (up to numerical/cutoff error), hence N_inc ~ 0.
    passive_metrics = derive_metrics(passive_main, passive_main)
    max_passive_inc = float(np.nanmax(np.abs(passive_metrics.N_inc)))
    max_passive_vdev = float(np.nanmax(np.abs(passive_metrics.Vbar - 0.5)))
    print("Passive coherent-state sanity check:")
    print(f"  max |N_inc| = {max_passive_inc:.3e}")
    print(f"  max |Vbar-1/2| = {max_passive_vdev:.3e}")
    if max_passive_inc > 1e-6 or max_passive_vdev > 1e-6:
        print("  WARNING: passive collector is not numerically coherent to 1e-6.")
    print()

    # Main pump-state solves.  Partial state is obtained exactly by raw-moment mixing.
    requested = [k for k in kinds if k != "partial"]
    if "coherent" not in requested:
        requested.insert(0, "coherent")
    if "phase_randomized" not in requested:
        requested.append("phase_randomized")

    raw_main: dict[str, SeededRawResult] = {}
    for kind in requested:
        print(f"Main seeded solve: {kind}")
        raw_main[kind] = solve_seeded_case(kind, p, beta_in=beta_main)
        print()

    raw_main["partial"] = mix_raw_results(
        raw_main["coherent"], raw_main["phase_randomized"], p.eta_partial
    )

    ordered_kinds = [k for k in ["coherent", "partial", "phase_randomized", "fock"] if k in raw_main]
    metrics_map: dict[str, SeededMetrics] = {}
    rows: list[dict[str, Union[float, str]]] = []

    print("Seeded amplifier summary at requested readout time")
    print(
        "state              eta_in     G_coh     N_add    N_QL     "
        "N_sig      N_inc      Vx/Vp        G_top_hat"
    )
    for kind in ordered_kinds:
        m = derive_metrics(raw_main[kind], passive_main)
        metrics_map[kind] = m
        row = summary_row(m, p)
        rows.append(row)
        idx = eval_index(m.tau_grid, p.tau_eval)
        print(
            f"{kind:18s} {m.eta_in:8.4f}  {m.G_over_passive[idx]:8.4f}  "
            f"{m.N_add[idx]:7.4f}  {m.N_add_ql[idx]:7.4f}  "
            f"{m.N_sig[idx]:9.5f}  {m.N_inc[idx]:9.5f}  "
            f"{m.Vx[idx]:.5f}/{m.Vp[idx]:.5f}  {m.G_mode[idx]:10.4f}"
        )

        ts_path = write_timeseries_csv(m, p)
        print(f"  saved {ts_path}")

    summary_path = write_summary_csv(rows, p)
    print(f"Saved summary CSV: {summary_path}")
    print()

    # ------------------------------------------------------------------
    # Weak-seed linearity test for the coherent pump.
    # Reuse the main coherent solve if its seed appears in the list.
    # ------------------------------------------------------------------
    linearity_rows: list[dict[str, float]] = []
    seed_values = np.array(sorted(set(float(x) for x in linearity_seeds if x > 0.0)))
    if not np.any(np.isclose(seed_values, seed_amp, rtol=0.0, atol=1e-12)):
        seed_values = np.sort(np.append(seed_values, seed_amp))

    print("Coherent-pump weak-seed linearity test")
    for amp in seed_values:
        beta = amp * np.sqrt(p.lam) * np.exp(1j * seed_phase)
        passive_scaled = rescale_passive_beta(passive_main, beta)
        if abs(amp - seed_amp) <= 1e-12:
            raw = raw_main["coherent"]
        else:
            print(f"  solving coherent pump at seed_amp={amp:.6f}")
            raw = solve_seeded_case("coherent", p, beta_in=beta)
        m = derive_metrics(raw, passive_scaled)
        idx = eval_index(m.tau_grid, p.tau_eval)
        line = {
            "seed_amp": float(amp),
            "tau_eval": float(m.tau_grid[idx]),
            "G_mode": float(m.G_mode[idx]),
            "G_over_passive": float(m.G_over_passive[idx]),
            "N_add": float(m.N_add[idx]),
            "N_inc": float(m.N_inc[idx]),
            "Vx": float(m.Vx[idx]),
            "Vp": float(m.Vp[idx]),
        }
        linearity_rows.append(line)
        print(
            f"    seed={amp:.6f}: G_coh={line['G_over_passive']:.6f}, "
            f"G_top_hat={line['G_mode']:.6f}, N_add={line['N_add']:.6f}, "
            f"N_inc={line['N_inc']:.6f}"
        )

    # Quantitative linearity diagnostics.
    Gvals = np.array([r["G_over_passive"] for r in linearity_rows])
    Nvals = np.array([r["N_add"] for r in linearity_rows])
    Ivals = np.array([r["N_inc"] for r in linearity_rows])

    def rel_span(x: np.ndarray) -> float:
        mean = float(np.nanmean(np.abs(x)))
        if mean <= 1e-14:
            return np.nan
        return float((np.nanmax(x) - np.nanmin(x)) / mean)

    print("Linearity diagnostics across seed amplitudes:")
    print(f"  relative span of G_coh  = {rel_span(Gvals):.3e}")
    print(f"  relative span of N_add  = {rel_span(Nvals):.3e}")
    print(f"  relative span of N_inc  = {rel_span(Ivals):.3e}")
    print("  (Small spans indicate operation in the weak-probe linear-response regime.)")
    print()

    linearity_path = ensure_outdir(p) / "seeded_linearity.csv"
    with linearity_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linearity_rows[0].keys()))
        w.writeheader()
        w.writerows(linearity_rows)
    print(f"Saved linearity CSV: {linearity_path}")

    p1 = plot_time_traces(metrics_map, p)
    p2 = plot_summary(metrics_map, p)
    p3 = plot_linearity(linearity_rows, p)
    print(f"Saved figures:\n  {p1}\n  {p2}\n  {p3}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_float_list(s: str) -> np.ndarray:
    return np.array([float(x.strip()) for x in s.split(",") if x.strip()], dtype=float)


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
        tau_eval=args.tau_eval,
        nsteps=args.nsteps,
        atol=args.atol,
        rtol=args.rtol,
        max_mesolve_dim=args.max_mesolve_dim,
        outdir=args.outdir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seeded coherent-gain and added-noise test for the Option-B finite-pump amplifier"
    )

    parser.add_argument("--nbar", type=float, default=5.0)
    parser.add_argument("--Na", type=int, default=5)
    parser.add_argument("--Nb", type=int, default=5)
    parser.add_argument("--Nc", type=int, default=14)
    parser.add_argument("--NAf", type=int, default=5)
    parser.add_argument("--NBf", type=int, default=5)
    parser.add_argument("--lam", type=float, default=1.0)

    parser.add_argument("--kappa_a", type=float, default=2.5)
    parser.add_argument("--kappa_b", type=float, default=2.5)
    parser.add_argument("--kappa_c", type=float, default=0.02)
    parser.add_argument("--gamma_A", type=float, default=2.0)
    parser.add_argument("--gamma_B", type=float, default=2.0)
    parser.add_argument("--eta_partial", type=float, default=0.25)

    parser.add_argument("--tau_max", type=float, default=2.0)
    parser.add_argument("--Nt", type=int, default=61)
    parser.add_argument("--tau_eval", type=float, default=1.3666666667)

    parser.add_argument("--seed_amp", type=float, default=0.05,
                        help="Dimensionless coherent input amplitude in units sqrt(lambda).")
    parser.add_argument("--seed_phase", type=float, default=0.0)
    parser.add_argument("--linearity_seeds", type=str, default="0.025,0.05,0.10")
    parser.add_argument("--kinds", type=str, default="coherent,phase_randomized,fock")

    parser.add_argument("--nsteps", type=int, default=50000)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--max_mesolve_dim", type=int, default=15000)
    parser.add_argument("--outdir", type=str, default="qb_seeded_gain_noise")

    args = parser.parse_args()
    p = make_params_from_args(args)

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    valid = {"coherent", "phase_randomized", "fock"}
    unknown = [k for k in kinds if k not in valid]
    if unknown:
        raise ValueError(f"Unknown --kinds entries: {unknown}; valid entries are {sorted(valid)}")
    if int(round(p.nbar)) >= p.Nc and "fock" in kinds:
        print("Warning: Fock pump does not fit in Nc cutoff. Removing Fock case.")
        kinds = [k for k in kinds if k != "fock"]

    run_seeded_noise(
        p=p,
        seed_amp=float(args.seed_amp),
        seed_phase=float(args.seed_phase),
        linearity_seeds=parse_float_list(args.linearity_seeds),
        kinds=kinds,
    )


if __name__ == "__main__":
    main()
