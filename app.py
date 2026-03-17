"""
app.py - 진입점

실행:
    streamlit run app.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gui.main_window import run

if __name__ == '__main__':
    run()
