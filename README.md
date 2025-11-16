# ESCA

Title-ESCA: An Emotional Support Conversation Agent for Enhancing Reasonable Strategy Planning and Effective Expression

There is the initial version of our code, we will further standardize our code and update it.

# For data pre-processing

You can directively use data/annotated_ensconv_with_states.json as the preprocessed data.

We also provide the annotation files: Annotation_full_file.py and vot_for_integration.py

# For knowledge pre-processing

Dowload PsyQA and ER dataset following their papers and covert them to the form of English question-answer.

python build_kb.py #conduct the knowledge base

python preprocess_add_knowledge.py #add knowledge to ESConv

# SFT the Strategy Planner and Prompt Generator

python sft_new.py --train_process ["sft_dp"] --num_train_epochs 5

python sft_new.py --train_process ["sft_pg"] --num_train_epochs 10


# RL for the Prompt Generator

python run.py --pgrl True

# test

python infer_gen.py #for evaluate the generation performance

python run.py --pgrl False #for evaluate the staretgy planning
