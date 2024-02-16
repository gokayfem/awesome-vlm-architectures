# Awesome-VLM-Architectures
<details> 
  <summary>LLaVA</summary> 
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
    <td><a href="https://arxiv.org/abs/2304.08485">LLaVA: Large Language and Vision Assistant</a></td>
    <td>LLaVA combines a pre-trained language model (LLM) with a visual model to leverage the capabilities of both for multimodal understanding. It integrates a vision encoder with a language decoder for processing and understanding language-image instruction data.</td>
    <td>Key components include the CLIP visual encoder for image feature extraction and the Vicuna language model for processing language instructions. A simple linear layer connects image features to the word embedding space, aligning visual and language representations.</td>
    <td>LLaVA is trained using a two-stage instruction-tuning procedure. The first stage involves pre-training for feature alignment, utilizing a filtered dataset to align image features with LLM word embeddings. The second stage involves fine-tuning both the projection layer and LLM end-to-end on specific tasks like a multimodal chatbot and Science QA, focusing on enhancing the model&#39;s instruction-following capabilities.</td>
    <td>The model employs instruction-tuning to align text-image data, generating multimodal instruction-following data using GPT-4. This involves converting image-text pairs into formats suitable for instruction-following tasks.</td>
    <td>A trainable projection matrix is used to convert visual features into language embedding tokens, aligning image and language representations within the same dimensional space. This facilitates encoding vision and text together effectively.</td>
    <td>Filtered CC3M, LLaVA-Instruct-158K, ScienceQA</td>
    <td>Filtered CC3M is used for pre-training to align visual and language features. LLaVA-Instruct-158K, a dataset generated using GPT-4, is used for fine-tuning on multimodal tasks. ScienceQA is utilized to evaluate the model&#39;s performance on multimodal reasoning tasks.</td>
    </tr>
    </tbody>
    </table>
</details>
<details>  
  <summary>LLaVA 1.5</summary> 
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
    <td><a href="https://arxiv.org/abs/2310.03744">Improved Baselines with Visual Instruction Tuning</a></td>
    <td>This paper introduces enhancements to LLaVA&#39;s architecture, employing a CLIP-ViT-L-336px vision encoder and an MLP projection layer, which significantly improves its data efficiency and performance across a range of benchmarks.</td>
    <td>The enhanced architecture includes the CLIP-ViT-L-336px for visual encoding and a multi-layer perceptron (MLP) for the vision-language cross-modal connector, enhancing the model&#39;s multimodal understanding.</td>
    <td>LLaVA-1.5 achieves state-of-the-art performance on 11 benchmarks with simple modifications, using a two-stage training approach focusing on efficient feature alignment and fine-tuning with academic-task-oriented VQA data.</td>
    <td>The paper focuses on improving multimodal alignment through instruction tuning, employing a more powerful MLP vision-language connector over the original linear projection, facilitating better integration of visual and linguistic data.</td>
    <td>Uses an MLP-based vision-language connector for more effective fusion of visual and textual representations, aligning them closely in the embedding space.</td>
    <td>VQA-v2, GQA, academic-task-oriented VQA datasets, incorporating OCR and region-level perception data.</td>
    <td>These datasets are used to significantly enhance the model&#39;s visual understanding and reasoning capabilities, demonstrating state-of-the-art performance with academic-task-oriented data.</td>
    </tr>
    </tbody>
    </table>
</details>
<details>  
  <summary>LLaVA 1.6</summary> 
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
    <td><a href="https://llava-vl.github.io/blog/2024-01-30-llava-next/">LLaVA-NeXT: Improved reasoning, OCR, and world knowledge</a></td>
    <td>LLaVA-NeXT introduces enhancements to LLaVA with higher image resolutions, improved visual reasoning and OCR capabilities, and better world knowledge. It maintains the minimalistic design of LLaVA-1.5, focusing on data efficiency and performance.</td>
    <td>Improvements include a higher input image resolution supporting up to 672x672, 336x1344, 1344x336 pixels, an enhanced visual instruction tuning data mixture for better reasoning and OCR, and efficient deployment with SGLang.</td>
    <td>LLaVA-NeXT is trained using less than 1M visual instruction tuning samples and reuses the pretrained connector from LLaVA-1.5, achieving efficient training with just 32 A100 GPUs in about 1 day.</td>
    <td>The model leverages high-resolution images for detailed visual perception and incorporates a high-quality data mixture for robust visual conversation and instruction following.</td>
    <td>Employs dynamic high-resolution techniques (&#39;AnyRes&#39;) to improve model&#39;s visual understanding, allowing it to handle images with varying resolutions effectively.</td>
    <td>LAION-GPT-V, ShareGPT-4V, DocVQA, SynDog-EN, ChartQA, DVQA, AI2D</td>
    <td>These datasets are utilized to enhance the model&#39;s visual reasoning, OCR capabilities, and understanding of charts and diagrams, aiming for improved performance across diverse multimodal tasks.</td>
    </tr>
    </tbody>
    </table>
</details>
<details> 
  <summary>FROZEN</summary> 
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
    <td><a href="https://arxiv.org/abs/2106.13884">Multimodal Few-Shot Learning with Frozen Language Models</a></td>
    <td>Frozen introduces a method to extend few-shot learning capabilities of language models to multimodal settings (vision and language) without modifying the language model&#39;s weights. It involves training a vision encoder to encode images into a sequence of continuous embeddings.</td>
    <td>The architecture includes a pre-trained autoregressive language model based on the Transformer architecture and a vision encoder based on NF-ResNet-50. It uses the final output vector of the NF-Resnet after global pooling as a visual prefix.</td>
    <td>Training updates only the parameters of the vision encoder using paired image-caption data from the Conceptual Captions dataset. The language model&#39;s weights remain frozen, making the system modular and simple.</td>
    <td>Frozen employs a dynamic visual prefix, contrasting with static text prompts used in prefix tuning. This allows for multimodal task performance improvement through in-context learning.</td>
    <td>The visual prefix is linearly mapped and reshaped into a sequence of embeddings, functioning similarly to an embedding sequence of prefix tokens, facilitating the model&#39;s adaptation to multimodal inputs.</td>
    <td>Conceptual Captions</td>
    <td>Used for training the vision encoder to encode images into sequences of embeddings that are then processed by the language model to generate appropriate captions.</td>
    </tr>
    </tbody>
    </table>
</details>
<details> 
  <summary>Flamingo</summary>   
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
    <td><a href="https://arxiv.org/abs/2204.14198v2">Flamingo: a Visual Language Model for Few-Shot Learning</a></td>
    <td>Flamingo is a Visual Language Model that integrates pretrained vision and language models to handle interleaved visual and textual data, capable of processing sequences of text tokens interleaved with images and/or videos to produce text output. It leverages a Perceiver-based architecture for handling high-resolution images or videos.</td>
    <td>Key components include the Perceiver Resampler for reducing large feature maps to a manageable number of visual tokens, and gated cross-attention dense (GATED XATTN-DENSE) layers for conditioning the language model on visual inputs.</td>
    <td>Flamingo is trained on a diverse mixture of datasets scraped from the web, including interleaved image and text data, image-text pairs, and video-text pairs. The model minimizes a weighted sum of per-dataset expected negative log-likelihoods of text given visual inputs, using a gradient accumulation strategy over all datasets.</td>
    <td>The model uses a unique image-causal modeling approach to manage text-to-image cross-attention, allowing it to attend to visual tokens of the image that appeared just before the given text token in the interleaved sequence.</td>
    <td>Flamingo employs gated cross-attention layers (GATED XATTN-DENSE) between the pretrained language model layers, using a tanh-gating mechanism to merge the output of these newly added layers with the input representation from the residual connection, allowing for effective fusion of vision and text embeddings.</td>
    <td>MultiModal MassiveWeb (M3W), ALIGN dataset, LTIP (Long Text &amp; Image Pairs), VTP (Video &amp; Text Pairs)</td>
    <td>M3W is used for training on interleaved text and image data, ALIGN for image-text pairs, LTIP for high-quality image-text pairs, and VTP for video-text pairs.</td>
    </tr>
    </tbody>
    </table>
</details>

