"""
Definition of sequences used so far
"""
from mrisd import seq_diagram as sdi
import numpy as np


def se() -> sdi.Diagram:
    # set tr and plot size
    tr = 70
    num_trs_to_plot = 1

    # set flip angle
    alpha = 90
    alpha_ref = 180

    # some timings
    len_rf = 4
    len_grad = 2

    seq = sdi.Diagram(tr_in_ms=tr, num_trs=num_trs_to_plot,
                      annotate_consecutive_dash_points='TE/2')

    # set rf
    seq.add_rf(flip_angle_in_deg=alpha, timing_in_ms=-len_rf / 2, duration_in_ms=len_rf)

    # set slice sel
    seq.add_grad(axis='z', grad_amplitude=1.0, timing_in_ms=-len_rf / 2, duration_in_ms=len_rf)
    seq.add_grad(axis='z', grad_amplitude=-1.0, timing_in_ms=+len_rf / 2, duration_in_ms=len_grad)

    # set refocussing
    # set rf_excitation
    te = 40
    seq.add_rf(flip_angle_in_deg=alpha_ref, timing_in_ms=te / 2 - len_rf / 2, duration_in_ms=len_rf)

    # set slice sel
    seq.add_grad(axis='z', grad_amplitude=1.0, timing_in_ms=te / 2 - len_rf, duration_in_ms=len_grad)
    seq.add_grad(axis='z', grad_amplitude=0.5, timing_in_ms=te / 2 - len_rf / 2, duration_in_ms=len_rf)
    seq.add_grad(axis='z', grad_amplitude=1.0, timing_in_ms=te / 2 + len_rf / 2, duration_in_ms=len_grad)

    # set phase encode
    seq.add_grad(axis='y', grad_amplitude=1.0, timing_in_ms=te / 2 + len_rf, duration_in_ms=len_grad)
    # set read pre
    seq.add_grad(axis='x', grad_amplitude=-1.0, timing_in_ms=te - 2 * len_rf, duration_in_ms=len_grad)

    # set readout
    # add rf for diag
    seq.add_grad(axis='x', grad_amplitude=1.0, timing_in_ms=te - 2*len_grad, duration_in_ms=4*len_grad)
    # set adc
    seq.add_adc(timing_in_ms=te - len_grad, duration_in_ms=2*len_grad)

    # set easy signal plot
    seq.add_easy_signal(timing_in_ms=te - len_rf / 2, duration_in_ms=2*len_grad)

    # define magnetization curves
    _ = seq.set_mag_longitudinal(0, tr, t1_in_ms=1000, start_mag=0)
    _ = seq.set_mag_transverse(0, tr, start_mag=1.0, t2_in_ms=60)
    return seq


def mese(ref_fa: float = 180.0, sparse_sampling: bool = False) -> sdi.Diagram:
    # set tr and plot size
    tr = 160
    etl = 5
    esp = 30
    num_trs_to_plot = 1

    # set flip angle
    alpha = 90
    alpha_ref = int(ref_fa)

    # some timings
    len_rf = 4
    len_grad = 2

    seq = sdi.Diagram(tr_in_ms=tr, num_trs=num_trs_to_plot,
                      annotate_consecutive_dash_points='TE/2')

    # set rf
    seq.add_rf(flip_angle_in_deg=alpha, timing_in_ms=-len_rf / 2, duration_in_ms=len_rf)

    # set slice sel
    seq.add_grad(axis='z', grad_amplitude=1.0, timing_in_ms=-len_rf / 2, duration_in_ms=len_rf)
    seq.add_grad(axis='z', grad_amplitude=-1.0, timing_in_ms=+len_rf / 2, duration_in_ms=len_grad)

    # set read pre
    seq.add_grad(axis='x', grad_amplitude=1.0, timing_in_ms=len_rf,
                 duration_in_ms=len_grad)

    # set refocussing
    # set rf_excitation
    for e_idx in range(etl):
        seq.add_rf(flip_angle_in_deg=alpha_ref, timing_in_ms=(2*e_idx + 1) * esp / 2 - len_rf / 2,
                   duration_in_ms=len_rf)

        # set slice sel
        seq.add_grad(axis='z', grad_amplitude=1.0, timing_in_ms=(2*e_idx + 1) * esp / 2 - len_rf,
                     duration_in_ms=len_grad)
        seq.add_grad(axis='z', grad_amplitude=0.5, timing_in_ms=(2*e_idx + 1) * esp / 2 - len_rf / 2,
                     duration_in_ms=len_rf)
        seq.add_grad(axis='z', grad_amplitude=1.0, timing_in_ms=(2*e_idx + 1) * esp / 2 + len_rf / 2,
                     duration_in_ms=len_grad)

        # set phase encode
        if sparse_sampling:
            grad_amp = (1.0 - 0.5) * np.random.random_sample() + 0.5
        else:
            grad_amp = 1.0
        seq.add_grad(axis='y', grad_amplitude=grad_amp, timing_in_ms=(e_idx + 1) * esp - 3*len_grad,
                     duration_in_ms=len_grad)
        seq.add_grad(axis='y', grad_amplitude=grad_amp, timing_in_ms=(e_idx + 1) * esp + 2*len_grad,
                     duration_in_ms=len_grad)

        # set readout
        # add rf for diag
        seq.add_grad(axis='x', grad_amplitude=1.0, timing_in_ms=(e_idx + 1) * esp - 2*len_grad, duration_in_ms=4*len_grad)
        # set adc
        seq.add_adc(timing_in_ms=(e_idx + 1) * esp - len_grad, duration_in_ms=2*len_grad)

        # set easy signal plot
        seq.add_easy_signal(timing_in_ms=(e_idx + 1) * esp - len_rf / 2, duration_in_ms=2*len_grad)

    # define magnetization curves
    # _ = seq.set_mag_longitudinal(0, tr, t1_in_ms=1000, start_mag=0)
    # _ = seq.set_mag_transverse(0, tr, start_mag=1.0, t2_in_ms=60)
    return seq


def ir() -> sdi.Diagram:
    # set tr and plot size
    tr = 100
    num_trs_to_plot = 1

    # set fli angle
    alpha_i = 180
    alpha_e = 90

    # some timings
    len_rf = 4
    len_grad = 2

    seq = sdi.Diagram(tr_in_ms=tr, num_trs=num_trs_to_plot,
                      annotate_consecutive_dash_points='TI',
                      annotate_full_range='TR')

    # set rf
    seq.add_rf(flip_angle_in_deg=alpha_i, timing_in_ms=-len_rf / 2, duration_in_ms=len_rf)

    # set slice sel
    seq.add_grad(axis='z', grad_amplitude=1.0, timing_in_ms=-len_rf / 2, duration_in_ms=len_rf)
    seq.add_grad(axis='z', grad_amplitude=-1.0, timing_in_ms=+len_rf / 2, duration_in_ms=len_grad)

    # set rf_excitation
    ti = 30
    seq.add_rf(flip_angle_in_deg=alpha_e, timing_in_ms=ti-len_rf / 2, duration_in_ms=len_rf)

    # set slice sel
    seq.add_grad(axis='z', grad_amplitude=1.0, timing_in_ms=ti-len_rf / 2, duration_in_ms=len_rf)
    seq.add_grad(axis='z', grad_amplitude=-1.0, timing_in_ms=ti+len_rf / 2, duration_in_ms=len_grad)

    # set phase encode
    seq.add_grad(axis='y', grad_amplitude=1.0, timing_in_ms=ti + len_rf/2, duration_in_ms=len_grad)
    # set read pre
    seq.add_grad(axis='x', grad_amplitude=-1.0, timing_in_ms=ti + len_rf / 2, duration_in_ms=len_grad)
    # set readout
    seq.add_grad(axis='x', grad_amplitude=1.0, timing_in_ms=ti + len_rf / 2 + len_grad, duration_in_ms=3*len_grad)
    # set adc
    seq.add_adc(timing_in_ms=ti + len_rf / 2 + 1.3 * len_grad, duration_in_ms=2.4*len_grad)
    # set easy signal plot
    seq.add_easy_signal(timing_in_ms=ti + len_rf / 2 + 1.3 * len_grad, duration_in_ms=2.4*len_grad)
    # define magnetization curves
    m_z_ti = seq.set_mag_longitudinal(0, ti, t1_in_ms=ti/1.5)
    _ = seq.set_mag_longitudinal(ti, tr-ti, t1_in_ms=ti/1.5, start_mag=0)
    _ = seq.set_mag_transverse(ti, 200, m_z_ti)
    return seq


def flash_dfa() -> sdi.Diagram:
    # set tr and plot size
    tr = 25  # [ms]
    num_trs_to_plot = 4

    # set flip angles
    alpha_1 = 6  # [°]
    alpha_2 = 21  # [°]

    seq = sdi.Diagram(tr_in_ms=tr, num_trs=num_trs_to_plot)

    # map couple tr with alternating flip angles
    flip_angles = [alpha_1, alpha_2] * int(num_trs_to_plot / 2)

    # set per tr
    for tr_idx in range(flip_angles.__len__()):
        rf_len = 4
        grad_len = 2
        zero_time = tr_idx * tr
        # add rf
        seq.add_rf(flip_angles[tr_idx], zero_time - rf_len / 2, rf_len)
        # add slice grad / might be slab selective - timings always off starting point
        seq.add_grad('z', 1.0, zero_time - rf_len / 2, rf_len)
        seq.add_grad('z', -1.0, zero_time + rf_len / 2, grad_len)
        # add phase encode
        seq.add_grad('y', 1.0, zero_time + rf_len / 2, grad_len)
        # add pre read
        seq.add_grad('x', -1.0, zero_time + rf_len / 2, grad_len)
        # add read
        seq.add_grad('x', 1.0, zero_time + rf_len / 2 + grad_len, 3 * grad_len)
        # add adc
        seq.add_adc(zero_time + rf_len / 2 + 1.3 * grad_len, 2.4 * grad_len)
        # set easy signal display
        seq.add_easy_signal(zero_time + rf_len / 2 + 1.3 * grad_len, 2.4 * grad_len)

    return seq


if __name__ == '__main__':
    seq_d = flash_dfa()
    seq_d.plot(which_parts=['rf', 'gs', 'gr', 'gp', 'adc'])
