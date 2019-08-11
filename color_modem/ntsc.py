# -*- coding: utf-8 -*-

import numpy

from color_modem import comb, qam
from color_modem.qam import AbstractQamColorModem


class NtscVariant(qam.QamConfig):
    def __new__(cls, fsc, lines=525):
        return NtscVariant.__base__.__new__(cls, fsc, 1300000.0, 3600000.0, lines)


NtscVariant.NTSC = NtscVariant(fsc=227.5 * 15750.0 * 1000.0 / 1001.0)
NtscVariant.NTSC_I = NtscVariant(fsc=4429687.5, lines=625)
NtscVariant.NTSC443 = NtscVariant(fsc=4433618.75)
# This is the weird mode that Raspberry Pi actually outputs when nominally set to PAL-M
NtscVariant.NTSC361 = NtscVariant(fsc=229.5 * 15750.0 * 1000.0 / 1001.0)


class NtscModem(AbstractQamColorModem):
    def __init__(self, variant=NtscVariant.NTSC, fs=13500000.0):
        super(NtscModem, self).__init__(fs, variant)

    @staticmethod
    def encode_yuv(r, g, b):
        assert len(r) == len(g) == len(b)
        y = 0.3 * r + 0.59 * g + 0.11 * b
        u = -0.1476019510016258 * r - 0.2893575108184752 * g + 0.436959461820101 * b
        v = 0.6183717846575098 * r - 0.5185533057776567 * g - 0.099818478879853 * b
        return y, u, v

    @staticmethod
    def decode_yuv(y, u, v):
        assert len(y) == len(u) == len(v)
        r = 0.9999999999999998 * y + 1.133735501874552 * v + 0.007249535771601484 * u
        g = y - 0.5766784873222262 * v - 0.3834753199055935 * u
        b = y + 0.001087790524980047 * v + 2.037050709207452 * u
        return r, g, b

    def modulate_yuv(self, frame, line, y, u, v):
        start_phase = self.config.start_phase(frame, line)
        return self.qam.modulate(start_phase, y, u, v)

    def demodulate_yuv(self, frame, line, *args, **kwargs):
        start_phase = self.config.start_phase(frame, line)
        return self.qam.demodulate(start_phase, *args, **kwargs)

    @staticmethod
    def encode_composite_level(value):
        # max excursion: 936/714
        # white level: 1
        # black level: 0
        # min excursion: -164/714
        return (value * 714.0 + 164.0) / 1100.0
        return numpy.maximum(numpy.minimum(adjusted, 1.0), 0.0)

    @staticmethod
    def decode_composite_level(value):
        return (value * 1100.0 - 164.0) / 714.0


class NtscCombModem(comb.AbstractCombModem):
    def __init__(self, variant=NtscVariant.NTSC, fs=13500000.0, *args, **kwargs):
        super(NtscCombModem, self).__init__(NtscModem(variant, fs), *args, **kwargs)
        sine = numpy.sin(self.backend.config.line_shift * 0.5)
        if abs(sine) > 0.05:
            self._factor = 0.5 / sine
        else:
            self._factor = 1e9001

    def demodulate_yuv_combed(self, frame, line, last, curr):
        # NTSC(n) = Y(x) + U(x)*sin(wt(x)+LS*n) + V(x)*cos(wt(x)+LS*n)
        # NTSC(n+1) - NTSC(n) = U(x)*(sin(wt(x)+LS*(n+1)) - sin(wt(x)+LS*n)) + V(x)*(cos(wt(x)+LS*(n+1)) - cos(wt(x)+LS*n))
        # NTSC(n+1) - NTSC(n) = U(x)*2*sin(LS/2)*cos(wt(x) + LS*(n + 1/2)) - V(x)*2*sin(LS/2)*sin(wt(x) + LS*(n + 1/2))
        # NTSC(n+1) - NTSC(n) = 2*sin(LS/2)*(U(x)*cos(wt(x) + LS*(n + 1/2)) - V(x)*sin(wt(x) + LS*(n + 1/2)))
        #
        # In particular, in NTSC-I and NTSC-M, LS == %pi, so sin(LS/2) == 1
        last = numpy.array(last, copy=False)
        curr = numpy.array(curr, copy=False)

        if not numpy.isfinite(self._factor):
            return self.backend.demodulate_yuv(frame, line, curr, strip_chroma=False)

        diff_phase = self.backend.config.start_phase(frame, line) - 0.5 * self.backend.config.line_shift
        if diff_phase < 0.0:
            diff_phase += 2.0 * numpy.pi

        diff = curr - last
        _, v, u = self.backend.qam.demodulate(diff_phase, diff, strip_chroma=False)
        u *= self._factor
        v *= -self._factor
        return curr, u, v
