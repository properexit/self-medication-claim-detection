import warnings
warnings.filterwarnings("ignore")

from src.pipeline import ClaimPipeline

pipeline = ClaimPipeline()

text = """
I’m not a doctor, but I’ve been reading a lot because my dad has been in and out of the hospital lately.
It’s been stressful trying to understand lab results, scans, and what different doctors are saying.
Some of them seem confident, others less so, and it honestly feels overwhelming at times.

People often assume that antibiotics are harmless, but that hasn’t been what I’ve learned.
Just because someone feels better after taking antibiotics doesn’t mean the infection was bacterial to begin with.
A lot of illnesses improve on their own with time, rest, and supportive care.

Anyway, I’m trying to be more careful about questioning treatments and not jumping to conclusions.
If anyone has good resources for learning more about this stuff, I’d really appreciate it.
"""

out = pipeline.predict_on_long_text(text)
print(out)