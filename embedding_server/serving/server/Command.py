#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Command:
    terminate = b'TERMINATION'
    show_config = b'SHOW_CONFIG'
    new_job = b'REGISTER'

    @staticmethod
    def is_valid(cmd):
        print(vars(Command).items())
        return any(not k.startswith('__') and v == cmd for k, v in vars(Command).items())
