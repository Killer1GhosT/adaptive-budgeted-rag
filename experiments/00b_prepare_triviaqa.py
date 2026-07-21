"""
00b_prepare_triviaqa.py
-----------------------
Standalone TriviaQA prep (streaming, no 17GB download) so we don't re-fetch
HotpotQA. Produces data/triviaqa.jsonl in the uniform schema.

Run:
    python src/00b_prepare_triviaqa.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "prep", os.path.join(os.path.dirname(__file__), "00_prepare_data.py"))
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)

if __name__ == "__main__":
    prep.prepare_triviaqa(n=600)
    print("done. data/triviaqa.jsonl ready.")
