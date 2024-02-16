# Awesome-VLM-Architectures
<details> 
  <summary><h3>LLaVA</h3></summary> 
  
| Title                                                                    | Architecture.Overview                                                                                                                                                                                                                     | Architecture.Components                                                                                                                                                                           | Training.Methods                                                                                                                                                                                                                                                                                                                                                                                                   | Alignment.Techniques                                                                                                                                  | Alignment.Fusion Methods                                                                                                                                                   | Datasets.Used                                 | Datasets.Purpose                                                                                                                                                                                          |
| ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [LLaVA: Large Language and Vision Assistant](https://llava-vl.github.io) | LLaVA combines a pre-trained CLIP visual encoder with Vicuna LLM, applying a simple linear layer to transform image features into language embedding tokens. This lightweight scheme enables quick iteration on data-centric experiments. | Pre-trained CLIP visual encoder for visual features, Vicuna as the LLM for language understanding, and a trainable projection matrix for converting visual features to language embedding tokens. | LLaVA uses multi-turn conversation data for instruction-tuning, applying the original auto-regressive training objective of the LLM. It involves a two-stage instruction-tuning procedure: pre-training for feature alignment using filtered CC3M to 595K image-text pairs, and fine-tuning end-to-end while keeping visual encoder weights frozen. Training includes multimodal chatbot and Science QA scenarios. | Uses a simple linear layer for aligning image features with the language model's word embedding space, allowing the model to interpret visual tokens. | The fusion of visual and text embeddings is achieved through a trainable projection matrix, facilitating the conversion of visual features into language embedding tokens. | Filtered CC3M, LLaVA-Instruct-158K, ScienceQA | CC3M for pre-training feature alignment, LLaVA-Instruct-158K for fine-tuning on multimodal instruction-following data, and ScienceQA for evaluating on a large-scale multimodal science question dataset. |
</details>
<details>  
  <summary><h3>LLaVA 1.5</h3></summary> 
  
| Title                                                                           | Architecture.Overview                                                                                                                                                                                             | Architecture.Components                                                                                                          | Training.Methods                                                                                                                                                                                                                                                                                        | Alignment.Techniques                                                                                                                                                      | Alignment.Fusion Methods                                                                                                                                                         | Datasets.Used                                                                    | Datasets.Purpose                                                                                                                                      |
| ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Improved Baselines with Visual Instruction Tuning](https://llava-vl.github.io) | LLaVA-1.5 enhances the original LLaVA by integrating a two-layer MLP for the vision-language connector, scaling up the visual encoder to CLIP-ViT-L-336px, and incorporating academic-task-oriented VQA datasets. | Two-layer MLP for vision-language connection, CLIP-ViT-L-336px as the vision encoder, and Vicuna LLM for language understanding. | The training involves using a varied set of datasets, including VQA, OCR, and region-level perception datasets, to enhance model capabilities. Training also involves scaling up input image resolution and LLM size, with significant improvements observed upon increasing the LLM to 13B parameters. | Utilizes an MLP-based vision-language connector for improved multimodal capabilities, enabling stronger and more effective alignment between visual and language domains. | Incorporates academic-task-oriented data for better alignment and understanding, using response formatting prompts to regularize output formats for short and long-form answers. | VQA, OCR, region-level VQA, visual conversation, language conversation datasets. | Enhance model capabilities in various academic tasks and visual perceptions, improve multimodal understanding and instruction-following capabilities. |
</details>

