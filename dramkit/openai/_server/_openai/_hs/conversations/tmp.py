# -*- coding: utf-8 -*-

import os
from dramkit.iotools import read_lines, write_txt


if __name__ == '__main__':
    files = [x for x in os.listdir() if x.endswith('.json')]
    for file in files:
        lines = read_lines(file, encoding='utf-8')
        for line in lines:
            if '"api_key":' in line:
                lines.remove(line)
        write_txt(lines, file, encoding='utf-8')
