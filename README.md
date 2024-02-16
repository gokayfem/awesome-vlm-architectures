# Awesome-VLM-Architectures
<details> 
  <summary><h3>LLaVA</h3></summary> 
    <table>
    <thead>
    <tr>
    <th>Title</th>
    <th>Architecture.Overview</th>
    <th>Architecture.Components</th>
    <div style="width:290px"><th>Training.Methods</th></div> 
    <th>Alignment.Techniques</th>
    <th>Alignment.Fusion Methods</th>
    <th>Datasets.Used</th>
    <th>Datasets.Purpose</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td><a href="https://llava-vl.github.io">LLaVA: Large Language and Vision Assistant</a></td>
    <td>LLaVA combines a pre-trained CLIP visual encoder with Vicuna LLM, applying a simple linear layer to transform image features into language embedding tokens. This lightweight scheme enables quick iteration on data-centric experiments.</td>
    <td>Pre-trained CLIP visual encoder for visual features, Vicuna as the LLM for language understanding, and a trainable projection matrix for converting visual features to language embedding tokens.</td>
    <td>LLaVA uses multi-turn conversation data for instruction-tuning, applying the original auto-regressive training objective of the LLM. It involves a two-stage instruction-tuning procedure: pre-training for feature alignment using filtered CC3M to 595K image-text pairs, and fine-tuning end-to-end while keeping visual encoder weights frozen. Training includes multimodal chatbot and Science QA scenarios.</td>
    <td>Uses a simple linear layer for aligning image features with the language model&#39;s word embedding space, allowing the model to interpret visual tokens.</td>
    <td>The fusion of visual and text embeddings is achieved through a trainable projection matrix, facilitating the conversion of visual features into language embedding tokens.</td>
    <td>Filtered CC3M, LLaVA-Instruct-158K, ScienceQA</td>
    <td>CC3M for pre-training feature alignment, LLaVA-Instruct-158K for fine-tuning on multimodal instruction-following data, and ScienceQA for evaluating on a large-scale multimodal science question dataset.</td>
    </tr>
    </tbody>
    </table>
</details>
<details>  
  <summary><h3>LLaVA 1.5</h3></summary> 
    <table>
    <thead>
    <tr>
    <th>Title</th>
    <th>Architecture.Overview</th>
    <th>Architecture.Components</th>
    <th>Training.Methods</th>
    <th>Alignment.Techniques</th>
    <th>Alignment.Fusion Methods</th>
    <th>Datasets.Used</th>
    <th>Datasets.Purpose</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td><a href="https://llava-vl.github.io">Improved Baselines with Visual Instruction Tuning</a></td>
    <td>LLaVA-1.5 enhances the original LLaVA by integrating a two-layer MLP for the vision-language connector, scaling up the visual encoder to CLIP-ViT-L-336px, and incorporating academic-task-oriented VQA datasets.</td>
    <td>Two-layer MLP for vision-language connection, CLIP-ViT-L-336px as the vision encoder, and Vicuna LLM for language understanding.</td>
    <td>The training involves using a varied set of datasets, including VQA, OCR, and region-level perception datasets, to enhance model capabilities. Training also involves scaling up input image resolution and LLM size, with significant improvements observed upon increasing the LLM to 13B parameters.</td>
    <td>Utilizes an MLP-based vision-language connector for improved multimodal capabilities, enabling stronger and more effective alignment between visual and language domains.</td>
    <td>Incorporates academic-task-oriented data for better alignment and understanding, using response formatting prompts to regularize output formats for short and long-form answers.</td>
    <td>VQA, OCR, region-level VQA, visual conversation, language conversation datasets.</td>
    <td>Enhance model capabilities in various academic tasks and visual perceptions, improve multimodal understanding and instruction-following capabilities.</td>
    </tr>
    </tbody>
    </table>
</details>


