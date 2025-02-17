#login to huggingface hub
import huggingface_hub

#enable tf32
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate
import torch
#prevent downloading datasets from the internet
# os.environ['HF_DATASETS_OFFLINE'] = "1"

from datasets import load_dataset, concatenate_datasets, DatasetDict

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    __version__
)
logger = logging.getLogger(__name__)

print(f"Transformers version: {__version__}")

from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict
transformers.logging.set_verbosity_info()


import pathlib
import json
import gdown
from tqdm import tqdm

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    attn_implementation: Optional[str] = field(
        default="flash_attention_2",
        metadata={"help": "The flash attention to use in the model. Default is flash_attention_2"},
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=True, #always trust remote code
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default="auto",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    save_to_disk: bool = field(
        default=True, metadata={"help": "Whether to save the dataset to disk"},
    )
    count_total_tokens: bool = field(
        default=False, metadata={"help": "Whether to count total tokens in the dataset"},
    )
    use_cache_data: bool = field(
        default= False, metadata={"help": "This will load data from cache_dir."},
    )
    base_data_dir: Optional[str] = field(
        default="./data",
        metadata={"help": "The base directory for the data files."},
    )
    google_drive_id: Optional[str] = field(
        default=None, metadata={"help": "The google id for data file."}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    is_proccessed: bool = field(
        default=False, metadata={"help": "If the dataset on hugging_face hub is already processed"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    train_dir: Optional[str] = field(default=None, metadata={"help": "The input training data dir."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split, value must be from 0.0 to 1.0"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default= 4,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")
        
        if self.dataset_name is None and self.train_file is None and self.train_dir is None and self.google_drive_id is None:
            raise ValueError("Need either a dataset name or a training/validation file/dir. or google id")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def concat_files_in_dir(data_dir):
    data = []
    path_obj= pathlib.Path(data_dir)
    files = [f.name for f in path_obj.rglob("*.json")]
    files = sorted(files)
    print("======================concatenating files in dir=========================")
    print(files)
    for f in files:
        path = os.path.join(data_dir,f)
        with open(path,"r") as f:
            data.extend(json.load(f))

    output_file = os.path.join(data_dir,"concatenated.json")
    with open(output_file,"w") as f:
        json.dump(data,f,indent=4)
    return output_file
def download_google_link(google_id, output_file):
    google_link = f"https://drive.google.com/uc?id={google_id}"
    gdown.download(google_link, output_file, quiet=False)
    return output_file
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)



    logger.info(f"Preparing base data directory {data_args.base_data_dir}")
    with training_args.main_process_first(desc="Preparing directory"):
        huggingface_hub.login("")
        if not os.path.exists(data_args.base_data_dir):
            os.makedirs(data_args.base_data_dir)

        cache_dir = data_args.base_data_dir + "/cache"
        data_args.data_cache_dir = cache_dir
        if not os.path.exists(data_args.data_cache_dir):
                os.makedirs(data_args.data_cache_dir)
    
    lm_datasets = None
    logger.info(f"Try loading dataset from cache")
    with training_args.main_process_first(desc="Try loading dataset from cache"):
        try:
            if data_args.use_cache_data:
                _middle = "default"
                if "llama" in model_args.model_name_or_path:
                    _middle = "llama"
                if "gemma" in model_args.model_name_or_path:
                    _middle = "gemma"
                
                cache_file = os.path.join(data_args.data_cache_dir, f"data_{_middle}_{data_args.block_size}.hf")
                print(f"Loading dataset from cache {cache_file} successfully")
                lm_datasets = datasets.load_from_disk(cache_file)
                logger.info(f"Loading dataset from cache successfully")
        except Exception as e:
            print(e)
            logger.info(f"Loading dataset from cache failed")

    if lm_datasets is None:
        if data_args.dataset_name is not None and data_args.is_proccessed:
            with training_args.main_process_first(desc="Try loading proccessed dataset from hub"):
                lm_datasets = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
                )
    if lm_datasets is None:   
        if data_args.dataset_name is not None:
            logger.info(f"Downloading and loading a dataset from the hub")
            print("Downloading and loading a dataset from the hub")
            raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            chunksize=40<<20,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
        )
            if "validation" not in raw_datasets.keys():
                # raw_datasets["validation"] = load_dataset(
                # data_args.dataset_name,
                # data_args.dataset_config_name,
                # split=f"train[:{data_args.validation_split_percentage}%]",
                # chunksize=40<<20,
                # cache_dir=model_args.cache_dir,
                # token=model_args.token,
                # streaming=data_args.streaming,
                # )
                # raw_datasets["train"] = load_dataset(
                # data_args.dataset_name,
                # data_args.dataset_config_name,
                # split=f"train[{data_args.validation_split_percentage}%:]",
                # chunksize=40<<20,
                # cache_dir=model_args.cache_dir,
                # token=model_args.token,
                # streaming=data_args.streaming,
                # )
                # print(f"Splitting the dataset into train and validation with ratio {data_args.validation_split_percentage*100} %")
                logger.info(f"Splitting the dataset into train and validation with ratio {data_args.validation_split_percentage*100} %")
                raw_datasets = raw_datasets["train"].train_test_split(test_size = data_args.validation_split_percentage)
                raw_datasets =  DatasetDict({"train": raw_datasets["train"],"validation": raw_datasets["test"]})
        if data_args.google_drive_id is not None:
            logger.info(f"Downloading a dataset from the google drive")
            with training_args.main_process_first(desc="Downloading data from google id"):
                google_file = data_args.base_data_dir + "/data.json"
                data_args.train_file = download_google_link(data_args.google_drive_id, google_file)
        
        if data_args.train_dir is not None:
                logger.info(f"Concatenating files in the directory {data_args.train_dir}")
                data_args.train_file = concat_files_in_dir(data_args.train_dir)
        
        if data_args.train_file is not None:  
            logger.info(f"Loading a dataset from the file {data_args.train_file}")       
            data_files = {}
            dataset_args = {}
            if data_args.train_file is not None:
                data_files["train"] = data_args.train_file
            if data_args.validation_file is not None:
                data_files["validation"] = data_args.validation_file
            extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
            )
            print(f"loading a dataset from the local {extension} file")
            print(data_files)
            if extension == "txt":
                extension = "text"
                dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
            raw_datasets = load_dataset(
                extension,
                chunksize=40<<20,
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
            if "validation" not in raw_datasets.keys():
                logger.info(f"Splitting the dataset into train and validation with ratio {data_args.validation_split_percentage*100} %")
                raw_datasets = raw_datasets["train"].train_test_split(test_size = data_args.validation_split_percentage)
                raw_datasets =  DatasetDict({"train": raw_datasets["train"],"validation": raw_datasets["test"]})
    

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if training_args.gradient_checkpointing:
         config_kwargs["use_cache"] = False
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            attn_implementation = model_args.attn_implementation,
        )
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # if training_args.peft_path is not None:
    #     logger.info("Peft from pre-trained model")
    #     model = PeftModel.from_pretrained(model, training_args.peft_path, is_trainable=True)
    # else:
    #     logger.info("Init new peft model")
    #     target_modules = training_args.trainable.split(',')
    #     modules_to_save = training_args.modules_to_save
    #     if modules_to_save is not None:
    #         modules_to_save = modules_to_save.split(',')
    #     lora_rank = training_args.lora_rank
    #     lora_dropout = training_args.lora_dropout
    #     lora_alpha = training_args.lora_alpha
    #     logger.info(f"target_modules: {target_modules}")
    #     logger.info(f"lora_rank: {lora_rank}")
    #     peft_config = LoraConfig(
    #         task_type=TaskType.CAUSAL_LM,
    #         target_modules=target_modules,
    #         inference_mode=False,
    #         r=lora_rank, lora_alpha=lora_alpha,
    #         lora_dropout=lora_dropout,
    #         modules_to_save=modules_to_save)
    #     model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))
    # # # convert the model to BetterTransformer
    # model.to_bettertransformer()

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if lm_datasets is None:
        if training_args.do_train:
            column_names = list(raw_datasets["train"].features)
        else:
            column_names = list(raw_datasets["validation"].features)
        text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output


    if hasattr(config, "max_position_embeddings"):
        max_pos_embeddings = config.max_position_embeddings
    else:
        # Define a default value if the attribute is missing in the config.
        max_pos_embeddings = 1024

    #define the block size
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > max_pos_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, max_pos_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            if max_pos_embeddings > 0:
                block_size = min(1024, max_pos_embeddings)
            else:
                block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)


    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map
    if lm_datasets is None:
        with training_args.main_process_first(desc="dataset map tokenization"):
            # if tokenizer.is_fast:
            #     logger.info("Start tokenizing with fast tokenizer")
            #     use_num_proc = None
            # else:
            #     use_num_proc = data_args.preprocessing_num_workers
            if not data_args.streaming:
                tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                # num_proc=use_num_proc, #use 1 here to to make use of fast tokenizer
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            else:
                tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )
            total_tokens = 0
            logger.info("Start calculating total number of tokens")
            if data_args.count_total_tokens:
                for sample in tqdm(tokenized_datasets["train"]):
                    total_tokens+= len(sample['input_ids'])
                logger.info(f"Total number of tokens : {total_tokens}")
                print(f"\n\nTotal number of tokens : {total_tokens}")
        with training_args.main_process_first(desc="grouping texts together"):
            if not data_args.streaming:
                lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
            else:
                lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )
        if data_args.save_to_disk and data_args.data_cache_dir is not None:
                _middle = "default"
                if "llama" in model_args.model_name_or_path:
                    _middle = "llama"
                if "gemma" in model_args.model_name_or_path:
                    _middle = "gemma"
                cache_file = os.path.join(data_args.data_cache_dir, f"data_{_middle}_{block_size}.hf")
                logger.info(f"Saving processed dataset to {cache_file}")
                print(f"Saving processed dataset to {cache_file}")
                lm_datasets.save_to_disk(cache_file)
    if training_args.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval
        else None,
    )
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        try:
            train_result = trainer.train(resume_from_checkpoint=checkpoint)

        
        
            trainer.save_model()  # Saves the tokenizer too for easy upload

            metrics = train_result.metrics

            max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()
        except Exception as e:
            with torch.no_grad():
                torch.cuda.empty_cache()
            print(e)
            return
            
        

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()