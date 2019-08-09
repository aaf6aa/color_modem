# -*- coding: utf-8 -*-

import sys

from PIL import Image

from color_modem.comb import Simple3DCombModem, SimpleCombModem
from color_modem.image import ImageModem
from color_modem.niir import NiirModem
from color_modem.ntsc import NtscCombModem, NtscVariant, NtscModem
from color_modem.pal import Pal3DModem, PalVariant, PalDModem, PalSModem
from color_modem.secam import SecamModem


def main():
    #### NTSC
    # best quality 3D comb filter
    # modem = Simple3DCombModem(NtscCombModem())
    # simple 2D comb filter
    # modem = NtscCombModem()
    # simple bandpass filtering
    # modem = NtscModem()

    #### PAL
    # best quality 3D comb filter
    # modem = Pal3DModem()
    # standard PAL-D (2D comb filter)
    # modem = PalDModem()
    # simple PAL-S (bandpass filtering)
    # modem = PalSModem()

    #### SECAM
    modem = SecamModem()

    #### NIIR (SECAM IV)
    # modem = NiirModem()

    img_modem = ImageModem(modem)
    img = Image.open(sys.argv[1])
    img = img_modem.modulate(img, 0)
    img.save(sys.argv[2])
    img = img_modem.demodulate(img, 0)
    img.save(sys.argv[3])
