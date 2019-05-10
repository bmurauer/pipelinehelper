#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import sys

if __name__ == '__main__':
    suite = unittest.TestLoader().discover('tests')
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    ret = not result.wasSuccessful()
    sys.exit(ret)
