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
import os
import sys

from distsup import kio


if __name__ == '__main__':
    sys.stderr.write("%s %s\n" % (os.path.basename(__file__), sys.argv))
    parser = argparse.ArgumentParser()
    parser.add_argument('in_wav_scp')
    parser.add_argument('out_fx')

    args = parser.parse_args()
    ark_scp_output = f'ark:| copy-feats ark:- {args.out_fx}'
    with kio.open_or_fd(ark_scp_output, 'wb') as f:
        for key, wav in kio.read_wav_scp(args.in_wav_scp):
            kio.write_mat(f, wav[:, None], key=key)
