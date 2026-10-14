"""Replace one existing manuscript table without changing its surrounding text."""
from pathlib import Path
import re

PAPER = Path(__file__).resolve().parents[1] / 'paper/paper.tex'


def replace_table(label, tabular):
    source = PAPER.read_text(encoding='utf-8')
    blocks = list(re.finditer(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', source, re.S))
    matches = [m for m in blocks if '\\label{' + label + '}' in m.group()]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one existing table with label {label}')
    block = matches[0]
    updated, count = re.subn(r'\\begin\{tabular\}.*?\\end\{tabular\}',
                            lambda _: tabular.strip(), block.group(), flags=re.S)
    if count != 1:
        raise ValueError(f'Expected exactly one tabular in {label}')
    PAPER.write_text(source[:block.start()] + updated + source[block.end():],
                     encoding='utf-8', newline='\n')
