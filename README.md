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
<details>
  <summary>OpenFlamingo</summary>   
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
    <td><a href="https://huggingface.co/openflamingo/OpenFlamingo-9B-vitl-mpt7b">OpenFlamingo-9B-vitl-mpt7b</a></td>
    <td>OpenFlamingo is an open-source implementation of DeepMind&#39;s Flamingo models, using a CLIP ViT-L/14 vision encoder and MPT-7B language model.</td>
    <td>Includes cross-attention modules inserted in every fourth decoder block of the pretrained, frozen language model, allowing it to cross-attend to visual features during decoding.</td>
    <td>Trained on web-scraped image-text sequences, utilizing a mixture of LAION-2B and Multimodal C4 datasets. The model employs DistributedDataParallel training across 64 A100 80GB GPUs using automatic BF16 mixed precision.</td>
    <td>Follows the Flamingo modeling paradigm, freezing the vision and language model but training connecting modules for decoding with cross-attention to visual features.</td>
    <td>Cross-attention modules facilitate fusion of vision and text embeddings, inserted at specific intervals within the language model&#39;s decoder blocks.</td>
    <td>LAION-2B, Multimodal C4</td>
    <td>Trained on image-text sequences for understanding and generating text based on visual input, enhancing capabilities in tasks like captioning, visual question answering, and image classification.</td>
    </tr>
    </tbody>
    </table>
</details>
<details>
  <summary>IDEFICS</summary>
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
    <td><a href="https://huggingface.co/HuggingFaceM4/idefics-80b">IDEFICS: an 80 billion parameters vision and language model</a></td>
    <td>IDEFICS is a large-scale vision and language model with 80 billion parameters, reproducing Flamingo&#39;s capabilities. It accepts sequences of images and text as inputs to generate text outputs.</td>
    <td>It leverages a similar architecture to GPT-4 and Flamingo, integrating vision and language processing in a cohesive model framework.</td>
    <td>The model encountered loss spikes during training, addressed through rollback strategies and learning rate adjustments. Training stability was improved with an auxiliary z-loss to normalize logits.</td>
    <td>IDEFICS follows Flamingo&#39;s approach, using pretrained vision and language backbones and focusing on cross-modal understanding. The model&#39;s performance benefits from training on multimodal web documents.</td>
    <td>The specific fusion techniques for vision and text embeddings are not detailed in the memo but are likely similar to those used in Flamingo, involving cross-attention mechanisms.</td>
    <td>OBELICS, a curated collection of interleaved image-text web documents, alongside other web-scraped datasets.</td>
    <td>OBELICS dataset aims to improve model performance on multimodal tasks by leveraging longer text contexts and diverse web document types.</td>
    </tr>
    </tbody>
    </table>
</details>
<details>
  <summary>PALI</summary>
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
    <td><a href="https://arxiv.org/abs/2209.06794">PALI: A JOINTLY-SCALED MULTILINGUAL LANGUAGE-IMAGE MODEL</a></td>
    <td>PaLI is designed for both unimodal (language, vision) and multimodal (language and vision) tasks, using a general interface that accepts image and text as input and generates text as output. The model architecture integrates a text encoder-decoder Transformer with visual tokens from a Vision Transformer (ViT).</td>
    <td>The text encoder-decoder leverages pre-trained mT5 models, and the visual component includes a newly introduced and trained ViT architecture named ViT-e, scaling up to 4 billion parameters. Additionally, the model uses pre-trained unimodal checkpoints for efficient training.</td>
    <td>PaLI models are trained using a mixture of pre-training tasks designed for a wide range of capabilities beneficial for downstream tasks. Training involves a high-volume image-language dataset, WebLI, covering 10 billion images and texts in over 100 languages. The largest model, PaLI-17B, undergoes a two-phase training process, including a high-resolution phase.</td>
    <td>The model employs a unified modeling interface, treating various tasks through an &quot;image-and-text to text&quot; framework. This approach enables task agnosticism, allowing for seamless operation across different types of vision and language tasks.</td>
    <td>The integration of vision and text embeddings is facilitated by feeding a sequence of visual tokens, derived from the Vision Transformer, to the text encoder-decoder Transformer via cross-attention, allowing for efficient fusion of multimodal information.</td>
    <td>WebLI, Conceptual Captions (CC3M-35L), OCR data from WebLI, VQ2A-CC3M, Open Images</td>
    <td>WebLI is utilized for pre-training PaLI in a multilingual setting with images and texts from the web, enhancing the model&#39;s understanding and generation capabilities across languages. Other datasets contribute to training the model on specific tasks such as captioning, OCR, and VQA, ensuring broad and versatile multimodal proficiency.</td>
    </tr>
    </tbody>
    </table>
</details>
<details>
  <summary>PALM-E</summary>
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
    <td>PaLM-E: An Embodied Multimodal Language Model <a href="https://palm-e.github.io">PaLM-E</a></td>
    <td>PaLM-E integrates continuous embodied observations (images, state estimates, or other sensor modalities) into the language embedding space of a pre-trained language model. It&#39;s a decoder-only LLM generating textual completions autoregressively based on multimodal inputs.</td>
    <td>The model uses a pre-trained PaLM as the language model and incorporates continuous observations through encoders. These encoders map sensor modalities into a sequence of vectors with the same dimension as the language model&#39;s embedding space. The continuous information and text are interleaved to form multimodal sentences.</td>
    <td>PaLM-E is trained end-to-end on datasets consisting of continuous observations and text, with a cross-entropy loss function for the non-prefix tokens. The model is based on pre-trained variants of PaLM and incorporates Vision Transformers (ViTs) for image features. Training involves both pre-trained input encoders and ones trained from scratch, with variations including model freezing and co-training across diverse data.</td>
    <td>The model employs encoders to inject continuous sensor data into the language embedding space, enabling alignment between the multimodal inputs. This process allows PaLM-E to understand and generate responses based on a combination of text and sensor data.</td>
    <td>Fusion of vision and text embeddings occurs through interleaving multimodal tokens corresponding to sensor observations with text to form multimodal sentences. These sentences are processed by the model&#39;s self-attention layers in a manner analogous to text tokens, ensuring integrated encoding of vision and text information.</td>
    <td>Internet-scale vision-and-language data, robotics tasks datasets</td>
    <td>The diverse set of datasets, including internet-scale vision-and-language data and specific robotics tasks, is used to train PaLM-E on a wide range of embodied reasoning tasks. This enables the model to benefit from cross-domain transfer learning, improving its performance on both specific robotics applications and general vision-language tasks.</td>
    </tr>
    </tbody>
    </table>
</details>
<details>
  <summary>Qwen-VL</summary>
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
    <td><a href="https://arxiv.org/abs/2308.12966">Qwen-VL: A Versatile Vision-Language Model for Understanding Localization Text Reading and Beyond</a></td>
    <td>Qwen-VL utilizes a large language model as its base, incorporating a Vision Transformer (ViT) as the visual encoder and a position-aware vision-language adapter. The model is designed for efficient handling of image features and alignment with language processing, featuring a compressed image feature sequence for integration into the language model.</td>
    <td>The key components include a foundational large language model, a Vision Transformer (ViT) for visual encoding, and a vision-language adapter with cross-attention mechanisms for efficient image feature compression and integration.</td>
    <td>The training process is divided into three stages: an initial pre-training on weakly labeled image-text pairs, multi-task pre-training with high-quality annotation data and larger input resolution, and supervised fine-tuning aimed at enhancing instruction-following and dialogue capabilities.</td>
    <td>Techniques involve the use of special tokens to differentiate between image and text inputs, and the introduction of bounding box inputs for fine-grained visual understanding.</td>
    <td>The model employs a cross-attention mechanism within the vision-language adapter to fuse visual and textual features, using positional encodings to retain spatial information after feature compression.</td>
    <td>LAION-en, LAION-zh, DataComp, Coyo, CC12M, CC3M, SBU, COCO Caption for pre-training; GQA, VGQA, VQAv2, DVQA, OCR-VQA, DocVQA, GRIT, Visual Genome, RefCOCO, RefCOCO+, RefCOCOg for multi-task pre-training.</td>
    <td>The datasets support a wide range of vision-language tasks including captioning, visual question answering, grounding, and OCR, with a focus on multilingual and fine-grained visual understanding.</td>
    </tr>
    </tbody>
    </table>  
</details>
