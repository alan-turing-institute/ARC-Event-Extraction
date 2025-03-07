import yaml
import itertools

MODELS_LARGE = ["mistral-large:123b", "llama3.3:70b"]
MODELS = {"llama3.1:8b": "llama8b", "gemma2:27b": "gemma27b", "mistral-small:24b": "mistral24b", "phi4": "phi4", "mistral-large:123b": "mistral123b", "llama3.3:70b": "llama70b"}
SENTENCE_EMBEDDINGS = {"all-MiniLM-L6-v2": "MiniLM-L6", "all-mpnet-base-v2": "all-mpnet-base", "intfloat/multilingual-e5-small": "e5small"}
DATASET = "ACE05"
SPLITS = [1,2,3,4,5]
PROMPTS = ["EDPrompter1", "EDPrompter2", "EDPrompter3", "EDPrompter4", "EAEPrompter1", "EAEPrompter2", "EAEPrompter3", "EAEPrompter4", "EAEPrompter5", "EAEPrompter6"]
PROMPTS_RANDOM_EXAMPLES = ["EDPrompter3", "EAEPrompter3", "EAEPrompter4"]
PROMPTS_SIMILAR_EXAMPLES = ["EDPrompter4", "EAEPrompter5", "EAEPrompter6"]
NUM_EXAMPLES = [1,2,3,4,5,6,7,8,9,10]
SEED = 42


for model in MODELS:
    model_short_name = MODELS[model]
    num_ctx = 8192
    if model in MODELS_LARGE:
        num_ctx = 16384
    for split in SPLITS:
        for prompt in PROMPTS:
            argument_extraction = False
            if "EAE" in prompt:
                argument_extraction = True
            exp_name = f"{DATASET}_split{split}_{model_short_name}_{prompt}"
            for split in SPLITS:
                if prompt in PROMPTS_RANDOM_EXAMPLES or prompt in PROMPTS_SIMILAR_EXAMPLES:
                    prompt_init_kwargs = {}
                    if prompt in PROMPTS_RANDOM_EXAMPLES:
                        exp_name = f"{exp_name}_" + str(SEED)
                        prompt_init_kwargs["seed"] = SEED
                    elif prompt in PROMPTS_SIMILAR_EXAMPLES:
                        for sentence_embedding in SENTENCE_EMBEDDINGS:
                            exp_name = f"{exp_name}_{sentence_embedding}"
                            prompt_init_kwargs["sentence_transformer_model"] = sentence_embedding


