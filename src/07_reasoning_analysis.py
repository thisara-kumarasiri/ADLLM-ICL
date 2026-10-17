from utils.claim_extractions import run_claim_extraction
from utils.consistency_judge import run_consistency_analysis

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import Settings as cfg


def main():

    if cfg.N_SHOT == 0:
        method = "zero_shot"
    else:
        method = cfg.SELECTION_METHOD

    shot = f"{cfg.N_SHOT}-shot"

    print(
        f"Running reasoning analysis\n"
        f"Shot: {shot}\n"
        f"Model: {cfg.LLM_MODEL}\n"
        f"Method: {method}"
    )

    # --------------------------------------------------------
    # 1. Atomic claim / region extraction
    # --------------------------------------------------------

    run_claim_extraction(
        model=cfg.LLM_MODEL,
        method=method,
        shot=shot,
    )

    # --------------------------------------------------------
    # 2. Reasoning consistency across runs
    # --------------------------------------------------------

    run_consistency_analysis(
        model=cfg.LLM_MODEL,
        method=method,
        shot=shot,
    )

    print(
        "\nReasoning analysis complete."
    )


if __name__ == "__main__":
    main()