# -*- coding: utf8 -*-
#   Copyright 2019 JSALT2019 Distant Supervision Team
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import argparse
import torch
import os
import sys
import distsup.aligner
from distsup import alphabet

'''
    We envision the following procedure
    - Add an "alignment_name: mypath.path
      argument in your .yaml file in the model section.
      We use tasman.yaml as example

    - With evaluate.py, like
    python evaluate.py egs/scribblelens/yamls/tasman.yaml runs/test --initialize-from runs/tasman.mode4/checkpoints/checkpoint_2228.pkl
    that will genetrate example.path (with actually hundreds of path (sizeof(test set)))

    - For readAlignment.py will read a single path  and return a list of
      alphabet index == class idx
'''
def getArgs():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--path",    required = True, help = "Name of a path file (which has the image data, length, etc)")
    args = vars(ap.parse_args())
    return args

if __name__ == "__main__":
    args       = getArgs()

    alphabet = alphabet.Alphabet("egs/scribblelens/tasman.alphabet.plus.space.mode5.json")
    myaligner = distsup.aligner.Aligner("none",alphabet)

    classIndexAlignmentList = myaligner.readAlignmentFile(args["path"])
    print (classIndexAlignmentList)

