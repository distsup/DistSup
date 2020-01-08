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

"""Some extensoins to the Python Kaldi_io package for reading waves
"""
import io
import soundfile

from kaldi_io import open_or_fd


def read_wav(file_or_fd, wav_sr=16000):
    """ [wav] = read_mat(file_or_fd)
     Reads a single wavefile, supports binary only
     file_or_fd : file, gzipped file, pipe or opened file descriptor.
    """
    fd = open_or_fd(file_or_fd)
    try:
        wav, sr = soundfile.read(io.BytesIO(fd.read()))
        assert sr == wav_sr
        wav = wav.astype('float32')
    finally:
        if fd is not file_or_fd:
            fd.close()
    return wav


def read_wav_scp(file_or_fd):
    """ generator(key, wav) = read_wav_scp(file_or_fd)
     Returns generator of (key,vector) tuples, read according to kaldi scp.
     file_or_fd : scp, gzipped scp, pipe or opened file descriptor.
     Iterate the scp:
     for key,vec in kaldi_io.read_vec_flt_scp(file):
         ...
     Read scp to a 'dictionary':
     d = { key:mat for key,mat in kaldi_io.read_mat_scp(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        for line in fd:
            line = line.decode().strip()
            if not line:
                continue
            (key, rxfile) = line.split(None, 1)
            wav = read_wav(rxfile)
            yield key, wav
    finally:
        if fd is not file_or_fd:
            fd.close()
