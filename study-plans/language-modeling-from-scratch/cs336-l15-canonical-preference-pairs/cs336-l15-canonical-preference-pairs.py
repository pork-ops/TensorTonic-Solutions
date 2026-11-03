def canonical_preference_pairs(records: list) -> dict:
    """
    Returns a dict with pairs: a list of prompt_id, winner_id, loser_id dictionaries.
    """
    response = dict()
    for r in records:
        prompt_id = r["prompt_id"]
        response[prompt_id] = []
        candidates = sorted(r["candidates"], key=lambda d: d['rank'])

        end = False
        for i in range(len(candidates)):
            for r in candidates[i+1:]:
                response[prompt_id].append({"winner_id": candidates[i]["id"] ,"loser_id": r["id"]})
    final = []
    for key in response:
        for p in response[key]:
            final.append({"prompt_id": key} | p )  
    return {"pairs": final}
    
