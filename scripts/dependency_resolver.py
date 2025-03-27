def resolve_flag_dependencies(flags: dict) -> dict:
    flags = flags.copy()
    if flags.get("useprelocationenricheddata"):
        flags["useprelocationdata"] = True
        flags["usepostnlpdata"] = True
        flags["useprecombineddata"] = True
    elif flags.get("useprelocationdata"):
        flags["usepostnlpdata"] = True
        flags["useprecombineddata"] = True
    elif flags.get("usepostnlpdata"):
        flags["useprecombineddata"] = True
    return flags