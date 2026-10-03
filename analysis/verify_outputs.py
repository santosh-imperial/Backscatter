"""Check analysis completeness, consistency and report asset references."""
from pathlib import Path
from html.parser import HTMLParser
import pandas as pd
import numpy as np

p=Path(__file__).resolve().parent
d=pd.read_csv(p/'image_inventory.csv');c=pd.read_csv(p/'channel_pair_checks.csv')
assert len(d)==93 and len(d.groupby(['batch','field']))==31
assert len(c)==93 and c.same_dimensions.all()
assert d.interior_chromatic_fraction.max()==0, 'Colored export borders remain in analysis region'
assert not d.pixel_sha256.duplicated().any()
assert d.groupby(['batch','field']).size().eq(3).all()
assert np.isfinite(d[['mean_gray','std_gray','gradient_rms','dark_fraction_central']].to_numpy()).all()
assert (d.dark_fraction_low<=d.dark_fraction_central).all()
assert (d.dark_fraction_central<=d.dark_fraction_high).all()
assert ((d.analysis_width==d.width-8)&(d.analysis_height==d.height-8)).all()
class Links(HTMLParser):
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ['src','href']:
                assert (p/value).exists(),value
Links().feed((p/'report.html').read_text())
assert len(pd.read_csv(p/'exploratory_batch_comparisons.csv'))==81
print('PASS: 93 images; 31 triplets; 93 dimension-matched detector pairs; color-free interiors; no pixel duplicates; finite metrics; ordered sensitivity fractions; correct cropping; 81 comparisons; all report links/assets.')
