"""Pattern-based extraction: a self-learning layer over the query parser.

Two in-memory stores accumulate knowledge as queries flow through:
  - PatternLibrary    — abstract query shapes → segment types (skip the LLM)
  - NormalizationDB   — typos the LLM corrected (feed back into normalize)

See patterns/library.py for the core Segment / PatternLibrary concepts.
"""
