# ESCA

Title-ESCA: An Emotional Support Conversation Agent for Enhancing Reasonable Strategy Planning and Effective Expression

There is the initial version of our code, we will further standardize our code and update it.

# For data pre-processing

You can directively use data/annotated_ensconv_with_states.json as the preprocessed data.

We also provide the annotation files: Annotation_full_file.py and vot_for_integration.py

# For knowledge pre-processing

Dowload PsyQA and ER dataset following their papers and covert them to the form of English question-answer. Due to the licensing issue, it cannot be provided directly.

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

# Some explanations

Because our model uses llama-70b for judgement and as a seeker, its accuracy is far lower than that of chatgpt. 
Therefore, our reproduction results for PPDPP and DPDP differ from the original papers. 
However, we make sure that both the comparison model and our model use the same model API for dialogue simulation and termination judgement.

# Future work

In the future, we will continue to improve the ESCA following the steps mentioned in the future work section of the paper. 
And we will also share the update results and code, the link will be updated here. 