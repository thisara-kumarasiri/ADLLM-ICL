# ============================================================
# SYSTEM PROMPTS
# ============================================================

INST_SYS_CTX = """
You are a neurologist analyzing brain MRI data to detect {neuro} disease.
"""


# ============================================================
# 1. PREDICTION ONLY
# Used by methods that only require the predicted class,
# e.g. ByCS initial prediction / candidate evaluation
# ============================================================

INST_USR_PRED_ONLY = """
Now analyze the following {information} and decide whether the subject has {neuro} disease or not.

{demographic_block}

{information}:
{metric_block}

Provide only the diagnosis: AD or CN.
Do not provide any reasoning.

Diagnosis:
"""


# ============================================================
# 2. PREDICTION + PROBABILITY
# Used for Delta matrix construction or any method requiring
# an LLM-generated confidence/probability.
# ============================================================

INST_USR_PROB_SHORT = """
Now analyze the following {information} and decide whether the subject has {neuro} disease or not.

Provide the predicted diagnosis and the probability assigned to that diagnosis.
Do not provide any reasoning.

{demographic_block}

{information}:
{metric_block}

Output exactly in the following format:
Diagnosis: AD or CN
Probability: <value between 0 and 1>
"""


# ============================================================
# 3. FINAL REASONING PROMPT
# Used for final experimental inference after demonstrations
# have been selected.
# ============================================================

INST_USR_REASONING = """
{examples_block}

Now analyze the following {information} and decide whether the subject has {neuro} disease or not.
Give an evidence-based short reasoning.

{demographic_block}

{information}:
{metric_block}

Output:
"""

# ============================================================
# 4. ZERO-SHOT FINAL REASONING PROMPT
# Same task instruction as the few-shot prompt, but without
# in-context demonstrations.
# ============================================================

INST_USR_REASONING_ZERO_SHOT = """
Now analyze the following {information} and decide whether the subject has {neuro} disease or not.
Give an evidence-based short reasoning.

{demographic_block}

{information}:
{metric_block}

Output:
"""