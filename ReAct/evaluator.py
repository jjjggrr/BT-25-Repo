
"""
Skeleton for quantitative evaluation of retrieval + fusion quality.
Extend with your gold test set and heuristics.

Metrics (to implement):
- Numeric accuracy: numbers in final answer must match numeric slice.
- Groundedness: every qualitative claim must map to an included doc_id.
- Relevance@k: NDCG/Recall using a labeled set of relevant doc_ids.
"""
from typing import List, Dict, Any
import re

def numbers_in_text(s:str) -> List[str]:
    return re.findall(r"-?\d+(?:[.,]\d+)?", s)

def check_numeric_grounding(answer:str, numeric_records:List[Dict[str,Any]]) -> Dict[str,Any]:
    nums = numbers_in_text(answer)
    # naive: just report numbers found; later, map to table values
    return {"numbers_found": nums, "total": len(nums)}

def check_textual_citations(answer:str, docs:List[Dict[str,Any]]) -> Dict[str,Any]:
    doc_ids = {d.get("doc_id") for d in docs}
    cited = set(re.findall(r"\{doc_id:([A-Za-z0-9_]+)", answer))
    return {"cited": list(cited), "present_in_context": list(cited & doc_ids)}
