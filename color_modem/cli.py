# -*- coding: utf-8 -*-

import sys
import os

from PIL import Image

from color_modem.color.mac import MacModem, MacVariant
from color_modem.color.niir import NiirModem, HueCorrectingNiirModem
from color_modem.color.ntsc import NtscCombModem, NtscVariant, NtscModem
from color_modem.color.pal import Pal3DModem, PalVariant, PalDModem, PalSModem
from color_modem.color.protosecam import ProtoSecamModem, ProtoSecamVariant
from color_modem.color.secam import SecamModem, SecamVariant
from color_modem.comb import Simple3DCombModem, SimpleCombModem, ColorAveragingModem
from color_modem.image import ImageModem
from color_modem.line import LineStandard, LineConfig


def convert(filename, output_filename, modem_name):
    img = Image.open(filename)

    line_config = LineConfig(img.size)

    modems = {
        "pal": PalSModem,
        "pal2d": PalDModem,
        "secam": SecamModem,
        "niir": NiirModem,
    }
    modem = modems[modem_name](line_config)

    img_modem = ImageModem(modem)
    img = img_modem.modulate(img, 0)
    #img.save(sys.argv[2])
    img = img_modem.demodulate(img, 0)
    img.save(output_filename)

def main():
    if len(sys.argv) == 4:
        convert(sys.argv[2], sys.argv[3], sys.argv[1])
    elif len(sys.argv) == 3:
        for filename in os.listdir(sys.argv[2]):
            path = os.path.join(sys.argv[2], filename)
            convert(path, path, sys.argv[1])
    else:
        print("Usage: <modem name> <input file> <output file>")
        sys.exit(1)

    #### NTSC
    # best quality 3D comb filter
    # modem = Simple3DCombModem(NtscCombModem(line_config))
    # simple 2D comb filter
    # modem = NtscCombModem(line_config)
    # simple bandpass filtering
    # modem = NtscModem(line_config)

    #### PAL
    # best quality 3D comb filter
    # modem = Pal3DModem(line_config)
    # standard PAL-D (2D comb filter)
    # modem = PalDModem(line_config)
    # simple PAL-S (bandpass filtering)
    # modem = PalSModem(line_config)

    #### SECAM
    # better quality modulation - filters out unrepresentable color patterns
    # modem = ColorAveragingModem(SecamModem(line_config))
    # basic
    # modem = SecamModem(line_config)
    # prototype 1957 AM 819-line variant
    # modem = ProtoSecamModem(line_config)

    #### NIIR (SECAM IV)
    # better quality modulation - hue correction
    # modem = HueCorrectingNiirModem(line_config)
    # standard
    # modem = NiirModem(line_config)
    # comb filter - turned out to be a bad idea
    # modem = SimpleCombModem(HueCorrectingNiirModem(line_config))

    #### D2-MAC
    # better quality modulation - filters out unrepresentable color patterns
    # modem = ColorAveragingModem(MacModem(line_config))
    # basic
    # modem = MacModem(line_config)
