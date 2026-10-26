#!/usr/bin/env python3
"""Extract original figure regions from archived PDFs using Poppler; no redrawing.

Coordinates refer to a full page rendered at height 900 pixels. The original PDF
remains the source; output uses 220 dpi and includes only the listed panel(s).
"""
import argparse, json, math, re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'references/paper-figures.json'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', help='Extract only this figure ID')
    args = parser.parse_args()
    for item in json.loads(CONFIG.read_text()):
        if args.id and args.id != item['id']:
            continue
        source = ROOT / item['pdf']
        info = subprocess.check_output(['pdfinfo', str(source)], text=True)
        height = float(re.search(r'Page size:\s+[\d.]+ x ([\d.]+) pts', info).group(1))
        scale = height * 220 / 72 / 900
        x,y,w,h = [math.floor(v * scale) for v in item['crop_at_height_900']]
        target = ROOT / item['output']
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['pdftoppm','-f',str(item['page']),'-l',str(item['page']),
            '-r','220','-x',str(x),'-y',str(y),'-W',str(w),'-H',str(h),
            '-singlefile','-png',str(source),str(target.with_suffix(''))],check=True)
        print(target.relative_to(ROOT))

if __name__ == '__main__':
    main()
