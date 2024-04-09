########################################################################################################################
# Copyright 2024 the authors (see AUTHORS file for full list).
#
#                                                                                                                    #
# This file is part of OpenCCM.
#
#                                                                                                                    #
# OpenCCM is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
#
# License as published by the Free Software Foundation, either version 2.1 of the License, or (at your option) any  later version.                                                                                                       #
#                                                                                                                    #
# OpenCCM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.                                                                                                             #
#                                                                                                                     #
# You should have received a copy of the GNU Lesser General Public License along with OpenCCM. If not, see             #
# <https://www.gnu.org/licenses/>.                                                                                     #
########################################################################################################################
from typing import Optional, Iterable

import numpy as np
from numpy import pi
from sympy import Function, Piecewise, cos

from openccm import ConfigParser


class H(Function):
    """
    Wrapper class for sympy usage.
    """

    @classmethod
    def eval(cls, t, y_start: float = 0.0, y_stop: float = 1.0, t_start: float = 0.0, dt: float = 1.0):
        """
        Smoothed Heaviside function using a cosine ramp to avoid numerical oscilations.
        Ramping happens over a half-period to provide a zero derivative at both the start and end point of the
        smoothing period.

        Args:
            t:          The time at which to evaluate the function at
            y_start:    The initial value of the function before t = t_start
            y_stop:     The final value of the function at t = t_start + dt
            t_start:    The time at which the function should start going from y_start to y_stop
            dt:         The period of time over which the function goes from y_start to y_stop
        """
        return Piecewise(
            (y_start,                                                     t < t_start),
            (y_stop,                                                            t > t_start + dt),
            ((y_start+y_stop)/2 + (y_start-y_stop)/2 * cos((t-t_start)*pi/dt),  True)
        )


def generate_t_eval(config_parser: ConfigParser) -> Optional[Iterable]:
    """
    Helper function to generate the time evaluation points which are given to solve_ivp.

    4 options are available:
        1. 'all':               DEFAULT. All time steps will be returned.
        2. dt, 'linear':        Linear distribution of points between t0 and tf with a spacing of dt.
                                t0 and tf are guaranteed to be the last and end point, even if the last step
                                is not of size dt.
        3. 'log', num_points:   Logarithmic distribution of points between t0 and tf with num_points points.
                                The exact values of t0 and tf are guaranteed to be the first and last point.
                                If t0 is 0, this is range is approximated by having a log distribution between
                                first_timestep and tf with num_points-1 points and then pre-pending 0 to the list.
        4. 't1, t2, ..., tn':   Arbitrary time points which are sorted. Only requirements are: t1 >= t0 and tn <= tf.

    Args:
        config_parser: The OpenCCM ConfigParser from which to get the t_eval string and the start and end times.

    Returns:
        ~:  Type depends on t_eval form. None is returned if all data points are to be stored,
            otherwise a list of floats or a numpy array is returned.
    """
    t_evel_str = config_parser.get_list(['SIMULATION', 't_eval'], str)
    t0, tf     = config_parser.get_list(['SIMULATION', 't_span'], float)
    assert t0 >= 0

    if len(t_evel_str) == 1 and t_evel_str[0].lower() == 'all':
        return None
    elif len(t_evel_str) == 2 and 'linear' in t_evel_str[1].lower():
        dt = float(t_evel_str[0])

        ts = []
        t = t0
        while t < tf:
            ts.append(t)
            t += dt
        ts.append(tf)
    elif len(t_evel_str) == 2 and 'log' in t_evel_str[0].lower():
        num_samples = int(t_evel_str[1])
        first_timestep = config_parser.get_item(['SIMULATION', 'first_timestep'], float)
        if t0 == 0:
            ts = np.zeros(num_samples)
            ts[1:] = np.logspace(np.log10(first_timestep), np.log10(tf), num_samples-1)
        else:
            ts = np.logspace(np.log10(t0), np.log10(tf), num_samples)
        ts[0]  = t0  # Get around any rounding errors that occurred when doing 10^log10(t0).
        ts[-1] = tf  # Get around any rounding errors that occurred when doing 10^log10(tf).
    else:  # literal string of numbers
        ts = sorted(float(time) for time in t_evel_str)
        if ts[0] < t0 or ts[-1] > tf:
            raise ValueError('Invalid time range provided, times must be within range specified by t_range.')

    return ts
