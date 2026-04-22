# 501R-final-project
A repo for my final project in 501R

All prompts used are in `src/prompts.py`. 

Valence arousal ratings for my prompts are found in `src/ratings_my_prompts.json`, valence arousal for claudes prompts are found in `src/ratings_claude_prompts.json`. 

This code uses an empathy/distress roberta classifier found in `src/empathy_model.py` and an emotion deberta classifier trained on GoEmotions in `src/emotion_model.py`, both of these models are from huggingface. It also uses a valence arousal classifier in `run_va_classifier.py`, instructions for using and downloading that model are below.

`src/lm_eval_experiment` allows you to inject different emotions into the LLM using the system prompt and run that LLM on a task in the lm_eval library. Download your own version of `config/lm_eval_example_config.yaml` in the `config/user_config` folder and fill it with the parameters you want to run with. The emotions available to run are in the conditions dictionary in `src/prompts.py`. You can add any prompts you want to run and inject in the LLM by adding the prompt in `src/prompts.py` and then updating the conditions dictionary.

`src/test_experiment` runs an even amount of questions from mmlu, emobench, eqbench, and truthfulqa in an open generated setting. There is a config example for this run in `config/open_ended_test.yaml`. Note that this experiment was updated to the `src/open_ended_experiment` and that should preferably be run over this.

`src/open_ended_experiment` uses `load_questions.py` to first load questions from different benchmarks in the `questions_dataset` directory to run. This allows you to check over the questions before they are run. Right now it loads questions from different specified subjects of mmlu, truthfulqa generation, emobench emotional understanding, and eqbench. If other datasets are desired add the code to load them in this file and format them like the other question json files. You can use `run_eval.py` to run the evaluation and `judge_responses.py` to judge the responses from the evaluation. Both of these files require a config, an example is in `config/open_ended_final.yaml`. 

`src/conflicting_experiment` is still being modified.

For experiments I developed prompts of varying levels of valence and arousal. To test that my prompts were robust I used a trained XLM-RoBERTa-large model from [Quantifying Valence and Arousal in Text with Multilingual Pre-trained Transformers](https://github.com/gmendes9/multilingual_va_prediction). To run this model classification to test your prompts you can download the model [here](https://drive.google.com/drive/folders/1BzdVmN51f33NHrdemJajz67MmlZljB2J) and put it in the `src/va_model` directory (or any other directory of your choice). Set `model_dir` to the directory with the va_model files in the `VAPredictor` class in `src/run_va_classifier.py` and run it.


If you are using slurm make sure to run `src/download_data.py` on the login node to download the models and datasets I used. Add any other datasets or models you want to test. 

If you are running the open ended experiment and want to run different questions add the dataset in `src/open_ended_experiment/load_questions.py` and configure it to match the format of the other dataset question files and configure it to run as an open ended tasks (don't give the LLM answer choices if they are available). 
