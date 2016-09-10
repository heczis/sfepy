#!/usr/bin/env python
from __future__ import absolute_import
from argparse import ArgumentParser

import sfepy
from sfepy.base.conf import ProblemConf, get_standard_keywords
from sfepy.homogenization.homogen_app import HomogenizationApp

helps = {
    'conf' :
    'override problem description file items, written as python'
    ' dictionary without surrounding braces',
    'options' : 'override options item of problem description,',
}

def main():
    parser = ArgumentParser()
    parser.add_argument("--version", action="version",
                        version="%(prog)s " + sfepy.__version__)
    parser.add_argument('-c', '--conf', metavar='"key : value, ..."',
                        action='store', dest='conf', type=str,
                        default=None, help= helps['conf'])
    parser.add_argument('-O', '--options', metavar='"key : value, ..."',
                        action='store', dest='app_options', type=str,
                        default=None, help=helps['options'])
    parser.add_argument('filename_in')
    options = parser.parse_args()

    # the value is not used even if set:
    options.output_filename_trunk = None

    filename_in = options.filename_in

    required, other = get_standard_keywords()
    required.remove('equations')

    conf = ProblemConf.from_file_and_options(
        filename_in, options, required, other)

    app = HomogenizationApp(conf, options, 'homogen:')
    opts = conf.options
    if hasattr(opts, 'parametric_hook'):  # Parametric study.
        parametric_hook = conf.get_function(opts.parametric_hook)
        app.parametrize(parametric_hook)
    app()

if __name__ == '__main__':
    main()
