from torch.utils.data import DataLoader

from arcee.utils.data import load_pattern
from arcee.prompts.prompts import DummyEAEPrompter

from arcee.dataset.dataset import EEDataset

# Paths
DATA_FILE_PATH = "data/RAMS/output/splits/split1/test.jsonlines"
PATTERN_FILE_PATH = "data/RAMS/patterns/pattern.txt"

# Load schema
schema = load_pattern(PATTERN_FILE_PATH, "rams")

# Load prompt constructor
prompt_constructor = DummyEAEPrompter()

# Load dataset
data = EEDataset.from_file(
    DATA_FILE_PATH,
    prompt_constructor,
    schema,
)

# Load dataloader
dataloader = DataLoader(
    data,
    batch_size=8,
    shuffle=False,
    collate_fn=lambda x: x,
)
        
# Iterate over loader
batch = next(iter(dataloader))


# If using examples, then load the dataset from which the examples
# can be sampled
# examples_data = []
# if examples_data_file_path is not None:
#     examples_data = load_data(examples_data_file_path)

# Temp for now: hard-coded to first ten records in dataset
for data in test_data[:10]:
    system_content = prompt_obj.get_system_content()
    prompt, _ = prompt_obj.get_prompt(
        data["tokens"],
        data["event_mentions"],
        schema,
        # examples_data,
    )
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]