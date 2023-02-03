import numpy as np


# define relaxation model
def e_1(t: float, t1_val: float, m_0: float = 1.0):
    # take times in same order of magnitude (s, ms, us)
    # define equilibrium magnetization to be normalized at 1 by default
    return m_0 * (1 - np.exp(-t / t1_val))


def t1_relaxation(t: float, t1_val: float, m_0: float = 1.0):
    return - m_0 * np.exp(-t / t1_val) + e_1(t, t1_val, m_0)


def t1_ir_relaxation(t: float, t1_val: float, rho: float = 1.0, m_initial: float = 1.0):
    # use either both times in seconds or ms
    # inversion flip angle defines starting value,
    # default: 180 degrees
    m_start = m_initial * np.cos(np.radians(rho * 180.0))
    # relaxation curve
    longitudinal_magnetization = m_start * np.exp(-t / t1_val) + e_1(t, t1_val)
    return longitudinal_magnetization


def t2_star_relaxation(t: float, t2_star_val: float, m_initial: float = 1.0):
    return m_initial * np.exp(-t / t2_star_val)
