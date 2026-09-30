"""LongMemEval answer and judge prompts, copied byte-exact from upstream.

Source: https://github.com/xiaowu0162/LongMemEval at the commit in UPSTREAM_COMMIT;
ANSWER_TEMPLATE from src/generation/run_generation.py (facts, no chain of thought),
the judge templates from get_anscheck_prompt() in src/evaluation/evaluate_qa.py.
Copyright (c) 2024 Di Wu. MIT License.
"""

from __future__ import annotations

UPSTREAM_COMMIT = "9e0b455f4ef0e2ab8f2e582289761153549043fc"

ANSWER_TEMPLATE = "I will give you several facts extracted from history chats between you and a user. Please answer the question based on the relevant facts.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer:"

QA_TEMPLATE = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."

TEMPORAL_TEMPLATE = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."

KNOWLEDGE_UPDATE_TEMPLATE = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."

PREFERENCE_TEMPLATE = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."

ABSTENTION_TEMPLATE = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."


def get_anscheck_prompt(task, question, answer, response, abstention=False):
    """Upstream's template selection: abstention first, then by question type."""
    if abstention:
        template = ABSTENTION_TEMPLATE
    elif task in ["single-session-user", "single-session-assistant", "multi-session"]:
        template = QA_TEMPLATE
    elif task == "temporal-reasoning":
        template = TEMPORAL_TEMPLATE
    elif task == "knowledge-update":
        template = KNOWLEDGE_UPDATE_TEMPLATE
    elif task == "single-session-preference":
        template = PREFERENCE_TEMPLATE
    else:
        raise NotImplementedError(task)
    return template.format(question, answer, response)


def judge_label(eval_response: str) -> bool:
    """Upstream's verdict parse: any 'yes' in the stripped, lowercased reply."""
    return "yes" in eval_response.strip().lower()
