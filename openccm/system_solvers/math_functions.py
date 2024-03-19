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
from numpy import pi
from sympy import Function, Piecewise, cos


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

