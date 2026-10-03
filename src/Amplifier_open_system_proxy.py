"""
Quantum-battery-powered nondegenerate parametric amplification
Open-system QuTiP master-equation proxy

Hamiltonian in the resonant interaction picture (hbar=1):
    H_I = i g ( c a^dag b^dag - c^dag a b )

Modes:
    a = signal resonator
    b = idler resonator
    c = finite pump / quantum-battery resonator

Open-system model:
    drho/dt = -i[H_I,rho]
             + kappa_a D[a]rho + kappa_b D[b]rho + kappa_c D[c]rho

The script compares coherent, partially dephased, phase-randomized, and Fock pump batteries.
It computes instantaneous output-mode proxies based on
    Phi_a(t) = kappa_a <a^dag a>
    Phi_b(t) = kappa_b <b^dag b>
    M_out(t) = sqrt(kappa_a kappa_b) <a b>

For kappa_a=kappa_b, the normalized interference proxy reduces to the same structure as
in the closed model:
    I_min/sqrt(kappa_a kappa_b) = n_a+n_b+1-2|<ab>|.

Only PDF outputs are saved.

Prepared for Borhan Ahmadi, 2026-05-21.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

try:
    import qutip as qt
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "This script requires QuTiP. Install it in your environment with, for example,\n"
        "    pip install qutip\n"
        "or run it in the Python environment where you already use QuTiP."
    ) from exc


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

@dataclass
class OpenSystemParams:
    # Battery energy and Hilbert-space cutoffs.
    nbar: float = 10.0
    Na: int = 12
    Nb: int = 12
    Nc: int = 32

    # Nominal classical squeezing rate lambda = g sqrt(nbar).
    # We set lambda = 1, so the plotted time tau equals lambda t.
    lambda_nominal: float = 1.0

    # Output-coupling/loss rates in units of lambda_nominal.
    # For a stiff-pump, below-threshold nondegenerate amplifier requires
    # rho = 2 lambda / sqrt(kappa_a kappa_b) < 1.
    kappa_a: float = 2.5
    kappa_b: float = 2.5
    kappa_c: float = 0.02

    # Time grid in nominal squeezing time tau = lambda t.
    tau_max: float = 4.0
    n_tau: int = 121

    # Partially dephased battery coherence fraction.
    eta_partial: float = 0.25

    # Solver tolerances.
    nsteps: int = 20000
    atol: float = 1e-8
    rtol: float = 1e-6

    # Output directory.
    outdir: Path = Path.cwd() / "qb_paramp_outputs"


# -----------------------------------------------------------------------------
# State constructors
# -----------------------------------------------------------------------------

def poisson_probs(nmax: int, nbar: float) -> np.ndarray:
    """Truncated Poisson distribution over n=0,...,nmax."""
    p = np.empty(nmax + 1, dtype=float)
    p[0] = np.exp(-nbar)
    for n in range(1, nmax + 1):
        p[n] = p[n - 1] * nbar / n
    return p / p.sum()


def phase_randomized_coherent_dm(N: int, nbar: float) -> qt.Qobj:
    """Density matrix diagonal in Fock basis with Poisson weights."""
    probs = poisson_probs(N - 1, nbar)
    rho = qt.Qobj(np.diag(probs), dims=[[N], [N]])
    return rho / rho.tr()


def pump_density_matrix(
    N: int,
    nbar: float,
    kind: str,
    eta_partial: float = 0.25,
) -> tuple[qt.Qobj, float]:
    """Return pump density matrix and nominal initial phase coherence eta_in.

    kind options:
        coherent
        partial
        phase_randomized
        fock

    For the partial state,
        rho = w |alpha><alpha| + (1-w) rho_phase_randomized,
    with w=sqrt(eta_partial).  This keeps the same photon-number distribution
    as the coherent state to very high accuracy, while reducing <c>.
    """
    alpha = np.sqrt(nbar)

    if kind == "coherent":
        return qt.coherent_dm(N, alpha), 1.0

    if kind == "phase_randomized":
        return phase_randomized_coherent_dm(N, nbar), 0.0

    if kind == "fock":
        n = int(round(nbar))
        if n >= N:
            raise ValueError("Fock pump photon number exceeds pump cutoff Nc.")
        return qt.fock_dm(N, n), 0.0

    if kind == "partial":
        w = np.sqrt(eta_partial)
        rho_coh = qt.coherent_dm(N, alpha)
        rho_pr = phase_randomized_coherent_dm(N, nbar)
        rho = w * rho_coh + (1.0 - w) * rho_pr
        return rho / rho.tr(), eta_partial

    raise ValueError(f"Unknown pump kind: {kind}")


# -----------------------------------------------------------------------------
# Operators and solver
# -----------------------------------------------------------------------------

def build_operators(p: OpenSystemParams) -> dict[str, qt.Qobj]:
    Ia = qt.qeye(p.Na)
    Ib = qt.qeye(p.Nb)
    Ic = qt.qeye(p.Nc)

    a = qt.tensor(qt.destroy(p.Na), Ib, Ic)
    b = qt.tensor(Ia, qt.destroy(p.Nb), Ic)
    c = qt.tensor(Ia, Ib, qt.destroy(p.Nc))

    na = a.dag() * a
    nb = b.dag() * b
    nc = c.dag() * c
    ab = a * b

    Pa_edge = qt.tensor(qt.fock_dm(p.Na, p.Na - 1), Ib, Ic)
    Pb_edge = qt.tensor(Ia, qt.fock_dm(p.Nb, p.Nb - 1), Ic)
    Pc_edge = qt.tensor(Ia, Ib, qt.fock_dm(p.Nc, p.Nc - 1))

    return dict(
        a=a,
        b=b,
        c=c,
        na=na,
        nb=nb,
        nc=nc,
        ab=ab,
        Pa_edge=Pa_edge,
        Pb_edge=Pb_edge,
        Pc_edge=Pc_edge,
    )


def solver_options(p: OpenSystemParams):
    """QuTiP 4/5 compatible solver options."""
    try:
        return qt.Options(
            nsteps=p.nsteps,
            atol=p.atol,
            rtol=p.rtol,
            store_states=False,
            progress_bar=None,
        )
    except Exception:
        return {"nsteps": p.nsteps, "atol": p.atol, "rtol": p.rtol, "store_states": False}


def initial_density_matrix(p: OpenSystemParams, pump_kind: str) -> tuple[qt.Qobj, float]:
    rho_a = qt.fock_dm(p.Na, 0)
    rho_b = qt.fock_dm(p.Nb, 0)
    rho_c, eta_in = pump_density_matrix(
        p.Nc,
        p.nbar,
        pump_kind,
        eta_partial=p.eta_partial,
    )
    return qt.tensor(rho_a, rho_b, rho_c), eta_in


def solve_open_system_case(
    p: OpenSystemParams,
    pump_kind: str,
    ops: dict[str, qt.Qobj] | None = None,
) -> dict[str, np.ndarray | float | str]:
    """Solve one pump-battery case and return diagnostics."""
    if ops is None:
        ops = build_operators(p)

    a = ops["a"]
    b = ops["b"]
    c = ops["c"]

    g = p.lambda_nominal / np.sqrt(p.nbar)
    H = 1j * g * (c * a.dag() * b.dag() - c.dag() * a * b)

    c_ops = []
    if p.kappa_a > 0:
        c_ops.append(np.sqrt(p.kappa_a) * a)
    if p.kappa_b > 0:
        c_ops.append(np.sqrt(p.kappa_b) * b)
    if p.kappa_c > 0:
        c_ops.append(np.sqrt(p.kappa_c) * c)

    tau_grid = np.linspace(0.0, p.tau_max, p.n_tau)
    tlist = tau_grid / p.lambda_nominal

    rho0, eta_in = initial_density_matrix(p, pump_kind)

    e_ops = [
        ops["na"],
        ops["nb"],
        ops["nc"],
        ops["ab"],
        ops["c"],
        ops["Pa_edge"],
        ops["Pb_edge"],
        ops["Pc_edge"],
    ]

    result = qt.mesolve(
        H,
        rho0,
        tlist,
        c_ops=c_ops,
        e_ops=e_ops,
        options=solver_options(p),
    )

    n_a = np.real(np.asarray(result.expect[0], dtype=complex))
    n_b = np.real(np.asarray(result.expect[1], dtype=complex))
    n_c = np.real(np.asarray(result.expect[2], dtype=complex))
    ab = np.asarray(result.expect[3], dtype=complex)
    c_mean = np.asarray(result.expect[4], dtype=complex)

    edge_a = np.real(np.asarray(result.expect[5], dtype=complex))
    edge_b = np.real(np.asarray(result.expect[6], dtype=complex))
    edge_c = np.real(np.asarray(result.expect[7], dtype=complex))

    n_pair = 0.5 * (n_a + n_b)
    denom_camp = np.sqrt(np.maximum(n_pair * (n_pair + 1.0), 1e-30))
    C_amp = np.abs(ab) / denom_camp
    C_amp[n_pair < 1e-12] = 0.0

    eta_c = np.abs(c_mean) ** 2 / np.maximum(n_c, 1e-30)
    eta_c[n_c < 1e-12] = 0.0

    V_minus_opt = n_a + n_b + 1.0 - 2.0 * np.abs(ab)
    V_plus_opt = n_a + n_b + 1.0 + 2.0 * np.abs(ab)

    Phi_a = p.kappa_a * n_a
    Phi_b = p.kappa_b * n_b
    M_out = np.sqrt(p.kappa_a * p.kappa_b) * ab
    k_ref = np.sqrt(p.kappa_a * p.kappa_b)

    # Mode-matched output interference proxy.  For kappa_a=kappa_b this is
    # exactly kappa times the closed-model I(phi) structure.
    I_mean = Phi_a + Phi_b + k_ref
    I_min = I_mean - 2.0 * np.abs(M_out)
    I_max = I_mean + 2.0 * np.abs(M_out)
    visibility_out = 2.0 * np.abs(M_out) / np.maximum(I_mean, 1e-30)

    N_pair_out = cumulative_trapezoid(0.5 * (Phi_a + Phi_b), tlist, initial=0.0)
    depletion = p.nbar - n_c

    return dict(
        pump_kind=pump_kind,
        eta_in=eta_in,
        tau=tau_grid,
        t=tlist,
        n_a=n_a,
        n_b=n_b,
        n_c=n_c,
        ab=ab,
        c_mean=c_mean,
        n_pair=n_pair,
        C_amp=C_amp,
        eta_c=eta_c,
        V_minus_opt=V_minus_opt,
        V_plus_opt=V_plus_opt,
        Phi_a=Phi_a,
        Phi_b=Phi_b,
        M_out=M_out,
        I_min=I_min,
        I_max=I_max,
        I_min_norm=I_min / k_ref,
        I_max_norm=I_max / k_ref,
        visibility_out=visibility_out,
        N_pair_out=N_pair_out,
        depletion=depletion,
        depletion_fraction=depletion / p.nbar,
        edge_a=edge_a,
        edge_b=edge_b,
        edge_c=edge_c,
        edge_max=max(float(np.max(edge_a)), float(np.max(edge_b)), float(np.max(edge_c))),
    )


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_open_system_proxy(
    p: OpenSystemParams,
    cases: dict[str, dict[str, np.ndarray | float | str]],
) -> Path:
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "legend.fontsize": 9,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })

    fig, axs = plt.subplots(2, 2, figsize=(9.6, 6.8))

    label_map = {
        "coherent": rf"coherent, $\eta_c^{{\rm in}}=1$",
        "partial": rf"partially dephased, $\eta_c^{{\rm in}}={p.eta_partial:.2f}$",
        "phase_randomized": rf"phase randomized, $\eta_c^{{\rm in}}=0$",
        "fock": "Fock",
    }

    for key, dat in cases.items():
        tau = dat["tau"]
        label = label_map.get(key, key)
        axs[0, 0].plot(tau, dat["visibility_out"], label=label)
        axs[0, 1].plot(tau, dat["I_min_norm"], label=label)
        axs[1, 0].plot(tau, dat["eta_c"], label=label)
        axs[1, 1].plot(tau, dat["N_pair_out"], label=label)

    axs[0, 0].set_xlabel(r"nominal time $\tau=\lambda t$")
    axs[0, 0].set_ylabel(r"output visibility proxy $\mathcal V_{\rm out}$")
    axs[0, 0].set_ylim(-0.04, 1.04)
    axs[0, 0].set_title("output phase locking")

    axs[0, 1].axhline(1.0, linestyle="-.", label=r"$I_{\min}=1$")
    axs[0, 1].set_xlabel(r"nominal time $\tau=\lambda t$")
    axs[0, 1].set_ylabel(r"normalized output dip $I_{\min}/\sqrt{\kappa_a\kappa_b}$")
    axs[0, 1].set_yscale("log")
    axs[0, 1].set_title("output squeezing proxy")

    axs[1, 0].set_xlabel(r"nominal time $\tau=\lambda t$")
    axs[1, 0].set_ylabel(r"battery coherence $\eta_c(t)$")
    axs[1, 0].set_ylim(-0.04, 1.04)
    axs[1, 0].set_title("surviving pump phase reference")

    axs[1, 1].set_xlabel(r"nominal time $\tau=\lambda t$")
    axs[1, 1].set_ylabel(r"cumulative emitted pairs $N_{\rm pair}^{\rm out}$")
    axs[1, 1].set_title("energy transfer to output")

    for ax in axs.flat:
        ax.tick_params(direction="in")
        ax.legend(frameon=False)

    fig.tight_layout()

    p.outdir.mkdir(parents=True, exist_ok=True)
    pdf_path = p.outdir / "qb_paramp_open_system_proxy.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    return pdf_path


def plot_open_system_fringe_snapshot(
    p: OpenSystemParams,
    cases: dict[str, dict[str, np.ndarray | float | str]],
    tau_star: float = 2.0,
) -> Path:
    phi = np.linspace(0.0, 2.0 * np.pi, 401)
    k_ref = np.sqrt(p.kappa_a * p.kappa_b)

    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "legend.fontsize": 10,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    label_map = {
        "coherent": rf"coherent, $\eta_c^{{\rm in}}=1$",
        "partial": rf"partially dephased, $\eta_c^{{\rm in}}={p.eta_partial:.2f}$",
        "phase_randomized": rf"phase randomized, $\eta_c^{{\rm in}}=0$",
        "fock": "Fock",
    }

    for key, dat in cases.items():
        tau = np.asarray(dat["tau"])
        idx = int(np.argmin(np.abs(tau - tau_star)))
        Phi_a = float(np.real(dat["Phi_a"][idx]))
        Phi_b = float(np.real(dat["Phi_b"][idx]))
        M = complex(dat["M_out"][idx])

        I_phi = Phi_a + Phi_b + k_ref + 2.0 * np.real(np.exp(-1j * phi) * M)
        I_phi_norm = I_phi / np.mean(I_phi)
        ax.plot(phi / np.pi, I_phi_norm, label=label_map.get(key, key))

    ax.set_xlabel(r"analysis phase $\phi/\pi$")
    ax.set_ylabel(r"$I_{\rm out}(\phi)/\overline{I}_{\rm out}$")
    ax.set_title(rf"output interference proxy at $\tau\simeq {tau_star:.1f}$")
    ax.tick_params(direction="in")
    ax.legend(frameon=False)
    fig.tight_layout()

    p.outdir.mkdir(parents=True, exist_ok=True)
    pdf_path = p.outdir / "qb_paramp_open_system_fringe_snapshot.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    return pdf_path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    p = OpenSystemParams()

    rho_reduced = 2.0 * p.lambda_nominal / np.sqrt(p.kappa_a * p.kappa_b)
    stiff_gain = ((1.0 + rho_reduced**2) / (1.0 - rho_reduced**2)) ** 2 if rho_reduced < 1 else np.inf

    print("Open-system finite-battery parametric amplifier")
    print(f"nbar = {p.nbar:.3f}")
    print(f"cutoffs: Na={p.Na}, Nb={p.Nb}, Nc={p.Nc}")
    print(f"lambda = g sqrt(nbar) = {p.lambda_nominal:.3f}")
    print(f"g = {p.lambda_nominal / np.sqrt(p.nbar):.6f}")
    print(f"kappa_a = {p.kappa_a:.3f}, kappa_b = {p.kappa_b:.3f}, kappa_c = {p.kappa_c:.3f}")
    print(f"stiff-pump reduced coupling rho = 2 lambda/sqrt(kappa_a kappa_b) = {rho_reduced:.3f}")
    if rho_reduced < 1:
        print(f"stiff-pump below-threshold power gain estimate G0 = {stiff_gain:.3f}")
    else:
        print("WARNING: stiff-pump proxy is above threshold; finite battery will still regularize dynamics.")
    print()

    ops = build_operators(p)

    pump_cases = ["coherent", "partial", "phase_randomized", "fock"]
    cases = {}

    for kind in pump_cases:
        print(f"Solving case: {kind}")
        dat = solve_open_system_case(p, kind, ops=ops)
        cases[kind] = dat

        final_idx = -1
        print(
            f"  eta_in={float(dat['eta_in']):.3f}, "
            f"final Vout={float(dat['visibility_out'][final_idx]):.4f}, "
            f"final Imin_norm={float(dat['I_min_norm'][final_idx]):.4f}, "
            f"final eta_c={float(dat['eta_c'][final_idx]):.4f}, "
            f"Npair_out={float(dat['N_pair_out'][final_idx]):.4f}, "
            f"max edge population={float(dat['edge_max']):.2e}"
        )
        if float(dat["edge_max"]) > 1e-3:
            warnings.warn(
                f"Large edge population for {kind}: {float(dat['edge_max']):.2e}. "
                "Increase Na, Nb, or Nc and rerun."
            )
        print()

    pdf1 = plot_open_system_proxy(p, cases)
    pdf2 = plot_open_system_fringe_snapshot(p, cases, tau_star=2.0)


if __name__ == "__main__":
    main()