import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mrisd import utils

logModule = logging.getLogger(__name__)


class Diagram:
    def __init__(self, tr_in_ms: int = 200, num_trs: int = 1):
        self.tr_in_ms: int = tr_in_ms
        self.num_trs: int = num_trs
        
        # set some adjustable vars - mainly for plotting
        self.prepend_start: int = 10      # [ms] plot before the start so 0 time objects are not cut
        self.prepend_text: int = 10       # annotation size in ms equivalent
        
        # set all arrays
        self.time_array = np.linspace(0, int(tr_in_ms * num_trs), int(tr_in_ms * num_trs * 1e3))  # sample time array in us
        self.rf_array: np.ndarray = np.full_like(self.time_array, np.nan)
        self.gs_array: np.ndarray = np.full_like(self.time_array, np.nan)
        self.gr_array: np.ndarray = np.full_like(self.time_array, np.nan)
        self.gp_array: np.ndarray = np.full_like(self.time_array, np.nan)
        self.dash_points: list = []

    def set_rf(self, flip_angle_in_deg: float, timing_in_ms: float, duration_in_ms: float = 4.0):
        # normalize size
        amp = np.radians(flip_angle_in_deg) / np.pi
        start_idx, end_idx = utils.get_start_end_idx(self.time_array, timing_in_ms, int(duration_in_ms))
        self.rf_array[start_idx:end_idx] = np.abs(
            amp * np.sinc(
                8 * np.linspace(-(end_idx - start_idx) / 2, (end_idx - start_idx) / 2, int(end_idx - start_idx))
            )
        )
        self.dash_points.append(self.time_array[int((end_idx - start_idx) / 2) + start_idx])

    def plot(self, which_parts: list = None):
        default = ['rf', 'gs', 'gr', 'gp', 'sig']
        if which_parts is None:
            which_parts = default
        for part in which_parts:
            if part not in default:
                err = f"Provide list of the parts to plot. Can only contain one or more of: {default}"
                logModule.error(err)
                raise ValueError(err)

        # setup plotting
        colors = cm.viridis(np.linspace(0, 1, 2*which_parts.__len__())).reshape((which_parts.__len__(), 2, -1))
        fig = plt.figure(figsize=(10, which_parts.__len__()))
        gs = fig.add_gridspec(which_parts.__len__(), 1)
        
        select_idx = 0
        
        if 'rf' in which_parts:
            ax_rf = fig.add_subplot(gs[select_idx])
            ax_rf.axis(False)
            ax_rf.set_xlim(np.min(self.time_array) - self.prepend_text, np.max(self.time_array))
            ax_rf.text(-self.prepend_text - self.prepend_start, 0, "RF")
            ax_rf.plot(self.time_array, self.rf_array, color=colors[select_idx, 0])
            ax_rf.fill_between(self.time_array, self.rf_array, alpha=0.6, color=colors[select_idx, 1])
            select_idx += 1

        if 'gs' in which_parts:
            ax_gs = fig.add_subplot(gs[select_idx])
            ax_gs.axis(False)
            ax_gs.text(-self.prepend_text - self.prepend_start, 0, "$G_{\mathrm{slice}}$")
            ax_gs.set_xlim(np.min(self.time_array) - self.prepend_text, np.max(self.time_array))
            ax_gs.plot(self.time_array, self.gs_array, color=colors[select_idx, 0])
            ax_gs.fill_between(self.time_array, self.gs_array, alpha=0.6, color=colors[select_idx, 1])
            select_idx += 1

        if 'gr' in which_parts:
            ax_gx = fig.add_subplot(gs[select_idx])
            ax_gx.axis(False)
            ax_gx.text(-self.prepend_text - self.prepend_start, 0, "$G_{\mathrm{read}}$")
            ax_gx.set_xlim(np.min(self.time_array) - self.prepend_text, np.max(self.time_array))
            select_idx += 1

        if 'gp' in which_parts:
            ax_gp = fig.add_subplot(gs[select_idx])
            ax_gp.axis(False)
            ax_gp.text(-self.prepend_text - self.prepend_start, 0, "$G_{\mathrm{phase}}$")
            ax_gp.set_xlim(np.min(self.time_array) - self.prepend_text, np.max(self.time_array))
            select_idx += 1

        if 'sig' in which_parts:
            ax_sig = fig.add_subplot(gs[select_idx])
            ax_sig.axis(False)
            ax_sig.text(-self.prepend_text - self.prepend_start, 0, "Signal")
            ax_sig.set_xlim(np.min(self.time_array) - self.prepend_text, np.max(self.time_array))
            select_idx += 1

        # plot annotation lines
        ax_inter = fig.add_subplot(gs[:])
        ax_inter.axis(False)
        ax_inter.set_xlim(np.min(self.time_array) - self.prepend_text, np.max(self.time_array))
        for val in self.dash_points:
            ax_inter.vlines(val, 1, -1, color='maroon', alpha=0.7, linestyles='dashed')

        plt.show()


if __name__ == '__main__':
    diag = Diagram()
    diag.set_rf(90, 0)

    diag.plot()

