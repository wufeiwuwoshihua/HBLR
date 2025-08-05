Here is the code repository for the paper *From Hypothesis to Premises: LLM-based Backward Logical Reasoning with Selective Symbolic Translation*.

Before running the code, please make sure to insert your own `api_key` into the `LLM_response_self` function in `utils.py`.

Use the following command to run HBLR experiments by replacing `model_name` and `dataset_name` with your desired values:

```bash
python ./HBLR.py --model_name [your_model_name] --data_path ./data --dataset_name [your_dataset_name] --split dev
```

Use the following command to run Reasoning Verification by replacing `model_name` and `dataset_name` with your desired values:

```bash
python ./verifier.py --model_name [your_model_name] --data_path ./data --dataset_name [your_dataset_name] --split dev
```

The `dataset_name` argument supports the following options: `FOLIO`, `ProntoQA`, `ProofWriter`, `LogicalDeduction`, and `AR-LSAT`. Replace `[your_dataset_name]` in the command with one of these dataset names to run the corresponding HBLR experiment.