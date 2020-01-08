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

import distsup.utils
import logging


def main():
    logging.basicConfig(level=logging.INFO)

    for i in range(2):
        distsup.utils.log('Should appear twice', once=False)

    for i in range(2):
        distsup.utils.log('Should only appear once', once=True)

    for i in range(2):
        distsup.utils.log('Should appear twice again', other_keys=(i,), once=True)


if __name__ == '__main__':
    main()
