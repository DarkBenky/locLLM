# Add this datasets

- [ ] Fim mode
  - [ ] injection language for contex part and fim part maybe use some special tokens like <lanng> python </lang>
  - [x] For example that is logged in terminal we should show sample that will look like training data in FIM mode so <context> + <fim_parts> + model output

- [ ] Distill the current dataset only for code
  - [ ] create training script that will be used for fim training
    - [ ] add special token <context_start> and </context_end>
    - [ ] inject context to model fim input

- [ ] TODO: multi-GPU training (not implemented yet)

- [x] https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1
- [ ] https://huggingface.co/datasets/rajpurkar/squad/
- [x] https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
- [ ] https://huggingface.co/datasets/tatsu-lab/alpaca
- [x] https://huggingface.co/datasets/r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation
- [x] https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1
- [x] https://huggingface.co/datasets/HuggingFaceTB/smoltalk
- [x] https://huggingface.co/datasets/open-thoughts/AgentTrove
- [x] https://huggingface.co/datasets/bigcode/starcoderdata
- [X] https://huggingface.co/datasets/code-search-net/code_search_net
- [ ] https://huggingface.co/datasets/open-phi/programming_books_llama

- [x] increase the split of FIM to 80%
  - [x] Fim support for all samples from stack V3 (also code_search_net raw + starcoderdata)
- [X] increase LR ?
- [X] optimize training (tokens/sec)
- [X] threading for dataset uploading (dataSets.py)
  - [X] pre dataset checkpoint

- [ ] save the indexes to disk so it can be loaded and with flag --reload it will reload the indexes
  - [ ] make this as service always running and pc and add to github copilot some kind of global instruction to use it
    - [ ] also add this to llmOpt so model can use it to there
