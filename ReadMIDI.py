# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 20:18:19 2015

@author: PC
"""
"sudo apt-get install swig"

import midi
pattern = midi.read_midifile("mary.mid")
print(pattern)
