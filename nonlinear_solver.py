import numpy as np
from sympy import nsolve
import matplotlib.pyplot as plt
import io, base64

def compute_Wmax(c1_val, c2_val, c3_val, theta_val, f1, f2, f3, initial_guess=(0.5, 0.5, 0.5)):
    """
    İlk denklem seti (f1, f2, f3) üzerinden Wₘₐₓ ve o noktadaki (γ1, γ2, γ3) döner.
    """
    W = 0.0
    step_coarse = 0.1
    prev_W = 0.0
    gamma_at_prev = initial_guess

    while True:
        subs_vals = {
            'c1': c1_val, 'c2': c2_val, 'c3': c3_val,
            'theta': theta_val, 'W': W
        }
        try:
            sol = nsolve(
                [f1.subs(subs_vals), f2.subs(subs_vals), f3.subs(subs_vals)],
                ['gamma1', 'gamma2', 'gamma3'],
                gamma_at_prev
            )
            g1_val = float(sol[0])
            g2_val = float(sol[1])
        except Exception:
            break

        lhs = (g1_val * g2_val) / (g1_val * g2_val + (1 - g1_val) * (1 - g2_val))
        if lhs >= theta_val:
            return prev_W, gamma_at_prev

        prev_W = W
        gamma_at_prev = (g1_val, g2_val, float(sol[2]))
        W += step_coarse

    return prev_W, gamma_at_prev


def compute_WFPs(c1_val, c2_val, c3_val, theta_val, Wmax, gamma_at_Wmax, g1, g2, g3, initial_guess=None):
    """
    İkinci denklem seti (g1, g2, g3) kullanılarak W_FP, W_FP1 ve W_FP_min döner.
    """
    if initial_guess is None:
        gamma_prev = gamma_at_Wmax
    else:
        gamma_prev = initial_guess

    W = Wmax
    step = 0.1
    W_FP = None
    W_FP1 = None

    while True:
        W += step
        subs_vals = {
            'c1': c1_val, 'c2': c2_val, 'c3': c3_val,
            'theta': theta_val, 'W': W
        }
        try:
            sol2 = nsolve(
                [g1.subs(subs_vals), g2.subs(subs_vals), g3.subs(subs_vals)],
                ['gamma1', 'gamma2', 'gamma3'],
                gamma_prev
            )
            g1_v = float(sol2[0])
            g2_v = float(sol2[1])
            g3_v = float(sol2[2])
        except Exception:
            break

        if W_FP1 is None and (g1_v < 0.5 or g1_v >= 1 or g2_v < 0.5 or g2_v >= 1 or g3_v < 0.5 or g3_v >= 1):
            W_FP1 = W

        num1 = g1_v * (1 - g3_v)
        den1 = num1 + (1 - g1_v) * g3_v
        cond1 = num1 / den1 if den1 != 0 else 999
        num2 = g1_v * g2_v
        den2 = num2 + (1 - g1_v) * (1 - g2_v)
        cond2 = num2 / den2 if den2 != 0 else -999

        if W_FP is None and not (cond1 < theta_val < cond2):
            W_FP = W

        if W_FP is not None and W_FP1 is not None:
            break

        gamma_prev = (g1_v, g2_v, g3_v)

    W_FP_min = None
    if W_FP is not None and W_FP1 is not None:
        W_FP_min = min(W_FP, W_FP1)
    elif W_FP is not None:
        W_FP_min = W_FP
    elif W_FP1 is not None:
        W_FP_min = W_FP1

    return W_FP, W_FP1, W_FP_min


def solve_gamma_range(range_start, range_end, N, c1_val, c2_val, c3_val, theta_val, eq_funcs, initial_guess):
    """
    W ∈ [range_start, range_end] arasında N nokta. eq_funcs = (e1, e2, e3) Sympy objeleri.
    """
    W_vals = np.linspace(range_start, range_end, N)
    gamma_vals = [[], [], []]

    for W in W_vals:
        subs_vals = {
            'c1': c1_val, 'c2': c2_val, 'c3': c3_val,
            'theta': theta_val, 'W': W
        }
        try:
            sol = nsolve(
                [eq_funcs[0].subs(subs_vals),
                 eq_funcs[1].subs(subs_vals),
                 eq_funcs[2].subs(subs_vals)],
                ['gamma1', 'gamma2', 'gamma3'],
                initial_guess
            )
            g1_num = float(sol[0])
            g2_num = float(sol[1])
            g3_num = float(sol[2])
            gamma_vals[0].append(g1_num)
            gamma_vals[1].append(g2_num)
            gamma_vals[2].append(g3_num)
            initial_guess = (g1_num, g2_num, g3_num)
        except Exception:
            gamma_vals[0].append(np.nan)
            gamma_vals[1].append(np.nan)
            gamma_vals[2].append(np.nan)

    return W_vals, gamma_vals


def generate_plots(W_vals1, gamma_vals1, W_vals2, gamma_vals2, Wmax, W_FP_min):
    """
    İki ayrı grafik oluşturup base64 döner: (plot1_b64, plot2_b64).
    """
    def fig_to_base64():
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()
        return img_b64

    # Grafik 1: W ∈ [0, Wmax]
    plt.figure()
    plt.plot(W_vals1, gamma_vals1[0], label='γ₁')
    plt.plot(W_vals1, gamma_vals1[1], label='γ₂')
    plt.plot(W_vals1, gamma_vals1[2], label='γ₃')
    plt.axvline(Wmax, color='black', linestyle='--', label='Wₘₐₓ')
    plt.xlabel('W')
    plt.ylabel('γ')
    plt.title('W ∈ [0, Wₘₐₓ]')
    plt.legend()
    plot1_b64 = fig_to_base64()

    # Grafik 2: W ∈ [Wmax, W_FP_min]
    plt.figure()
    plt.plot(W_vals2, gamma_vals2[0], label='γ₁')
    plt.plot(W_vals2, gamma_vals2[1], label='γ₂')
    plt.plot(W_vals2, gamma_vals2[2], label='γ₃')
    if W_FP_min is not None:
        plt.axvline(W_FP_min, color='red', linestyle='--', label='W_FP_min')
    plt.xlabel('W')
    plt.ylabel('γ')
    plt.title('W ∈ [Wₘₐₓ, W_FP_min]')
    plt.legend()
    plot2_b64 = fig_to_base64()

    return plot1_b64, plot2_b64
