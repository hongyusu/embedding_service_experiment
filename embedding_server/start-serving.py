#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from serving.server import Ventilator
from serving.server.helper import get_run_args


if __name__ == '__main__':
    args = get_run_args()
    server = Ventilator(args)
    server.start()
    server.join()
