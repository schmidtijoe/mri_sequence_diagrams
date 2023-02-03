import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mrisd import utils, relaxation

logModule = logging.getLogger(__name__)


class Diagram:
    def __init__(self, tr_in_ms: int = 200, num_trs: int = 1, annotate_consecutive_dash_points: str = 'TR',
                 annotate_full_range: str = None):
        self.tr_in_ms: int = tr_in_ms
        self.num_trs: int = num_trs
        self.t_total: int = int(tr_in_ms * num_trs)

        # set some adjustable vars - mainly for plotting
        self.prepend_start: int = 10  # [ms] plot before the start so 0 time objects are not cut
        self.prepend_text: float = 0.05 * self.t_total  # annotation size in ms equivalent
        self.larmor_vis_freq: float = 2 * 2 * np.pi  # per x ms

        # set all arrays
        self.time_array = np.linspace(-self.prepend_start, int(tr_in_ms * num_trs),
                                      int(tr_in_ms * num_trs * 1e3))  # sample time array in us
        self.rf_array: np.ndarray = np.full_like(self.time_array, np.nan)
        zero_point, _ = utils.get_start_end_idx(self.time_array, 0, 0)
        self.rf_array[zero_point:] = 0.0
        self.gs_array: np.ndarray = np.copy(self.rf_array)
        self.gr_array: np.ndarray = np.copy(self.rf_array)
        self.gp_array: np.ndarray = np.copy(self.rf_array)
        self.adc_array: np.ndarray = np.copy(self.rf_array)
        self.signal: np.ndarray = np.copy(self.rf_array)

        self.mag_z: np.ndarray = np.copy(self.signal)
        self.mag_xy: np.ndarray = np.copy(self.signal)

        self.dash_points: list = []
        self.dash_annotate: str = annotate_consecutive_dash_points
        self.annotate_full_range: str = annotate_full_range
        self.rf_labels: list = []
        self.echo_points: list = []

    @staticmethod
    def _map_rf_size(fa_in_deg: float):
        # want to visualize different amplitudes but within a size range
        amp_min = 0.4
        amp_max = 1.0
        # normalize 180° to 1
        mapped_fa = amp_min + (amp_max - amp_min) * np.radians(fa_in_deg) / np.pi
        # clip in case different angles are provided
        return np.clip(mapped_fa, amp_min, amp_max)

    def add_rf(self, flip_angle_in_deg: float, timing_in_ms: float,
               duration_in_ms: float = 4.0):
        # normalize size
        amp = self._map_rf_size(flip_angle_in_deg)
        start_idx, end_idx = utils.get_start_end_idx(self.time_array, timing_in_ms, duration_in_ms)
        sinc_x = np.linspace(-(end_idx - start_idx) / 2, (end_idx - start_idx) / 2, int(end_idx - start_idx))
        self.rf_array[start_idx:end_idx] = np.abs(amp * np.sinc(2e-3 * sinc_x))
        time_point = self.time_array[int((end_idx - start_idx) / 2) + start_idx]
        self.rf_labels.append([time_point, flip_angle_in_deg])
        self.dash_points.append(time_point)

    def add_grad(self, axis: str, grad_amplitude: float, timing_in_ms: float, duration_in_ms):
        # get indexes based on timing
        start_idx, end_idx = utils.get_start_end_idx(self.time_array, timing_in_ms, duration_in_ms)
        # choose array based on ax
        if axis == 'z':
            self.gs_array[start_idx:end_idx] = grad_amplitude
        elif axis == 'y':
            self.gp_array[start_idx:end_idx] = grad_amplitude
        elif axis == 'x':
            self.gr_array[start_idx:end_idx] = grad_amplitude
        else:
            err = 'no valid gradient axis given, choose from x, y or z'
            logModule.error(err)
            raise ValueError(err)

    def add_adc(self, timing_in_ms: float, duration_in_ms: float):
        # get indexes based on timing
        start_idx, end_idx = utils.get_start_end_idx(self.time_array, timing_in_ms, duration_in_ms)
        # choose array based on ax
        self.adc_array[start_idx:end_idx] = 1.0
        time_point = self.time_array[utils.array1d_to_value_find_nearest_idx(
            self.time_array, timing_in_ms + duration_in_ms / 2)]
        self.echo_points.append(time_point)

    def add_easy_signal(self, timing_in_ms: float, duration_in_ms: float):
        start_idx, end_idx = utils.get_start_end_idx(self.time_array, timing_in_ms, duration_in_ms)
        mid_idx = utils.array1d_to_value_find_nearest_idx(self.time_array, timing_in_ms + duration_in_ms / 2)

        larmor_sig = np.sin(self.larmor_vis_freq * self.time_array[start_idx:end_idx])
        sigma = 0.3 * duration_in_ms
        envelope = np.divide(
            np.exp(
                -np.square(self.time_array[start_idx:end_idx] - self.time_array[mid_idx]) / (2 * sigma ** 2)
            ),
            np.sqrt(2 * np.pi * sigma ** 2)
        )
        self.signal[start_idx:end_idx] = envelope * larmor_sig

    def set_mag_longitudinal(self, start_time: float, duration: float,
                             start_mag: float = 1.0, t1_in_ms: float = 200.0):
        # get indexes
        start_idx, end_idx = utils.get_start_end_idx(self.time_array, start_time, duration)
        # timing in us!
        relax_time = self.time_array[start_idx:end_idx] - self.time_array[start_idx]
        mag_curve = relaxation.t1_ir_relaxation(t=relax_time, t1_val=t1_in_ms, m_initial=start_mag)
        self.mag_z[start_idx:end_idx] = mag_curve
        return mag_curve[-1]

    def set_mag_transverse(self, start_time: float, duration: float, start_mag: float = 1.0, t2_in_ms: float = 10.0):
        start_idx, end_idx = utils.get_start_end_idx(self.time_array, start_time, duration)
        # timing in us!
        relax_time = self.time_array[start_idx:end_idx] - self.time_array[start_idx]
        mag_curve = relaxation.t2_star_relaxation(relax_time, t2_star_val=t2_in_ms, m_initial=start_mag)
        self.mag_xy[start_idx:end_idx] = mag_curve
        return mag_curve[-1]

    def plot(self, which_parts: list = None, dpi: int = 200, add_magnetization: bool = False, save: str = None):
        default = ['rf', 'gs', 'gr', 'gp', 'sig', 'adc']
        if add_magnetization:
            default.append('mz')
            default.append('mxy')
        if which_parts is None:
            which_parts = default
        for part in which_parts:
            if part not in default:
                err = f"Provide list of the parts to plot. Can only contain one or more of: {default}"
                logModule.error(err)
                raise ValueError(err)

        # setup plotting
        num_rows = which_parts.__len__()
        if 'signal' in which_parts and 'adc' in which_parts:
            num_rows -= 1
        colors = cm.viridis(np.linspace(0, 1, 2 * which_parts.__len__())).reshape((which_parts.__len__(), 2, -1))
        fig = plt.figure(figsize=(12, num_rows), dpi=dpi)
        # hr = np.full(num_rows + 1, 10)
        # hr[-1] = 1
        gs = fig.add_gridspec(num_rows, 1)

        select_idx = 0

        if 'rf' in which_parts:
            ax_rf = fig.add_subplot(gs[select_idx])
            ax_rf.axis(False)
            ax_rf.set_xlim(np.min(self.time_array), np.max(self.time_array))
            ax_rf.text(-self.prepend_text - self.prepend_start, 0, "RF")
            ax_rf.plot(self.time_array, self.rf_array, color=colors[select_idx, 0])
            ax_rf.fill_between(self.time_array, self.rf_array, alpha=0.6, color=colors[select_idx, 1])
            for labels in self.rf_labels:
                ax_rf.text(labels[0], 1.05 * np.nanmax(self.rf_array), f"$\\alpha$: {labels[1]:d} °")
            select_idx += 1

        if 'gs' in which_parts:
            ax_gs = fig.add_subplot(gs[select_idx])
            ax_gs.axis(False)
            ax_gs.text(-self.prepend_text - self.prepend_start, 0, "$G_{\mathrm{slice}}$")
            ax_gs.set_xlim(np.min(self.time_array), np.max(self.time_array))
            ax_gs.plot(self.time_array, self.gs_array, color=colors[select_idx, 0])
            ax_gs.fill_between(self.time_array, self.gs_array, alpha=0.6, color=colors[select_idx, 1])
            select_idx += 1

        if 'gr' in which_parts:
            ax_gx = fig.add_subplot(gs[select_idx])
            ax_gx.axis(False)
            ax_gx.text(-self.prepend_text - self.prepend_start, 0, "$G_{\mathrm{read}}$")
            ax_gx.set_xlim(np.min(self.time_array), np.max(self.time_array))
            ax_gx.plot(self.time_array, self.gr_array, color=colors[select_idx, 0])
            ax_gx.fill_between(self.time_array, self.gr_array, alpha=0.6, color=colors[select_idx, 1])
            select_idx += 1

        if 'gp' in which_parts:
            ax_gp = fig.add_subplot(gs[select_idx])
            ax_gp.axis(False)
            ax_gp.text(-self.prepend_text - self.prepend_start, 0, "$G_{\mathrm{phase}}$")
            ax_gp.set_xlim(np.min(self.time_array), np.max(self.time_array))
            ax_gp.plot(self.time_array, self.gp_array, color=colors[select_idx, 0])
            ax_gp.plot(self.time_array, -self.gp_array, color=colors[select_idx, 0])
            # plot phase encode steps
            for fac_interp in np.linspace(0, 1, 5):
                interpol_gp = fac_interp * self.gp_array
                ax_gp.plot(self.time_array, interpol_gp, color=colors[select_idx, 0])
                ax_gp.plot(self.time_array, -interpol_gp, color=colors[select_idx, 0])

            ax_gp.fill_between(self.time_array, self.gp_array, alpha=0.6, color=colors[select_idx, 1])
            ax_gp.fill_between(self.time_array, -self.gp_array, alpha=0.6, color=colors[select_idx, 1])
            select_idx += 1

        if 'adc' in which_parts or 'sig' in which_parts:
            ax_sig_adc = fig.add_subplot(gs[select_idx])
            ax_sig_adc.axis(False)
            ax_sig_adc.set_xlim(np.min(self.time_array), np.max(self.time_array))
            if 'adc' in which_parts:
                if 'sig' in which_parts:
                    alpha = 0.6
                else:
                    alpha = 1.0
                ax_sig_adc.fill_between(self.time_array, self.adc_array, color=colors[select_idx, 1],
                                        alpha=0.6 * alpha)
                ax_sig_adc.plot(self.time_array, self.adc_array, color=colors[select_idx, 0], alpha=alpha)

            if 'sig' in which_parts:
                ax_sig_adc.plot(self.time_array, self.signal, color='#ff4d4d')
                # ax_sig.fill_between(self.time_array, self.gs_array, alpha=0.6, color=colors[select_idx, 1])

            ax_sig_adc.text(-self.prepend_text - self.prepend_start, 0, 'Signal / ADC')

            for val in self.echo_points:
                ax_sig_adc.scatter(val, 0.05, color='#990000', zorder=3)
            select_idx += 1

        if 'mz' in which_parts:
            ax_mz = fig.add_subplot(gs[select_idx])
            ax_mz.axis(False)
            ax_mz.set_xlim(np.min(self.time_array), np.max(self.time_array))
            ax_mz.set_ylim(-1, 1)
            ax_mz.plot(self.time_array, self.mag_z, color=colors[select_idx, 0])
            null_line = np.zeros_like(self.time_array)
            null_line[np.argwhere(np.isnan(self.mag_z))] = np.nan
            ax_mz.plot(self.time_array, null_line,
                       alpha=0.8, color=colors[select_idx, 1], linestyle='dashed')
            ax_mz.text(-self.prepend_text - self.prepend_start, 0, '$M_z$')
            select_idx += 1

        if 'mxy' in which_parts:
            ax_mxy = fig.add_subplot(gs[select_idx])
            ax_mxy.axis(False)
            ax_mxy.set_xlim(np.min(self.time_array), np.max(self.time_array))
            ax_mxy.set_ylim(-1, 1)
            null_line = np.zeros_like(self.time_array)

            null_line[np.argwhere(np.isnan(self.mag_z))] = np.nan
            ax_mxy.plot(self.time_array, null_line,
                        alpha=0.8, color=colors[select_idx, 1], linestyle='dashed')
            ax_mxy.plot(self.time_array, self.mag_xy, color=colors[select_idx, 0])
            ax_mxy.fill_between(self.time_array[self.adc_array > 0], self.mag_xy[self.adc_array > 0],
                                color=colors[select_idx, 1], alpha=0.7)

            ax_mxy.text(-self.prepend_text - self.prepend_start, 0, '$M_xy$')
            select_idx += 1

        # plot annotation lines
        ax_inter = fig.add_subplot(gs[:])
        ax_inter.axis(False)
        ax_inter.set_xlim(np.min(self.time_array), np.max(self.time_array))
        ax_inter.set_ylim(0, 1)
        for val_idx in range(self.dash_points.__len__()):
            ax_inter.vlines(self.dash_points[val_idx], 0.1, 1, color='#cc0000', alpha=0.7, linestyles='dashed')
            if val_idx < self.dash_points.__len__() - 1:
                ax_inter.annotate(text='', xy=(self.dash_points[val_idx], 0.05),
                                  xytext=(self.dash_points[val_idx + 1], 0.05),
                                  arrowprops=dict(arrowstyle='<->', color='#800000',
                                                  ls='--', alpha=0.7))
                ax_inter.text(
                    (self.dash_points[val_idx + 1] - self.dash_points[val_idx]) / 2 + self.dash_points[val_idx],
                    0.06,
                    self.dash_annotate,
                    dict(color='#800000')
                )
        if self.annotate_full_range is not None:
            ax_inter.annotate(text='', xy=(self.dash_points[0], 0.02),
                                  xytext=(self.t_total, 0.02),
                                  arrowprops=dict(arrowstyle='<->', color='#800000',
                                                  ls='--', alpha=0.7))
            ax_inter.text(
                self.t_total/2 + self.dash_points[0],
                0.03,
                self.annotate_full_range,
                dict(color='#800000')
            )
        # plt.tight_layout()
        if save is not None:
            plt.savefig(save, bbox_inches='tight', dpi=dpi)
            plt.close()
        else:
            plt.show()


if __name__ == '__main__':
    diag = Diagram()
    diag.add_rf(90, 0)

    diag.plot()
