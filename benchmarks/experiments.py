"""Compatibility entry point. The current pipeline is reproduce.py."""
import runpy
from pathlib import Path
if __name__=='__main__':runpy.run_path(str(Path(__file__).with_name('reproduce.py')),run_name='__main__')
