# Awesome Visual Language Model Architectures
<details> 
  
  ![image](https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/722f0fbb-ea52-4a8a-ab1e-bec45ca7d04f)
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

  ![image](https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/27db1037-2b48-4097-9891-019ba77fc536)
  <summary>BLIP</summary>
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
    <td><a href="https://arxiv.org/abs/2201.12086">BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation</a></td>
    <td>BLIP introduces a Multimodal Mixture of Encoder-Decoder (MED) architecture for effective multi-task pre-training and flexible transfer learning. MED can function as a unimodal encoder, an image-grounded text encoder, or an image-grounded text decoder, allowing it to adapt to a variety of vision-language tasks.</td>
    <td>Visual Transformer as image encoder, BERT-based text encoder, additional cross-attention layers for image-text interaction, and causal self-attention layers for text generation. MED supports three functionalities: unimodal encoding, image-grounded text encoding, and image-grounded text decoding.</td>
    <td>Joint optimization of three pre-training objectives: Image-Text Contrastive Learning (ITC) for aligning visual and textual features, Image-Text Matching (ITM) for learning fine-grained image-text alignment, and Image-Conditioned Language Modeling (LM) for text generation from images. Uses a combination of human-annotated and web-collected noisy image-text pairs.</td>
    <td>Uses ITC and ITM losses for text-image alignment, leveraging a multimodal representation that captures the fine-grained relationship between visual and textual information.</td>
    <td>Employs cross-attention layers to inject visual information into the text encoder for image-grounded text encoding and modifies self-attention layers in the decoder for text generation, enabling effective encoding of vision and text together.</td>
    <td>COCO, Visual Genome, Conceptual Captions, Conceptual 12M, SBU Captions, LAION</td>
    <td>Used for pre-training to learn vision-language tasks, with COCO and Visual Genome providing high-quality human-annotated pairs, and the web datasets offering a large volume of image-text pairs for scalability and robustness enhancement.</td>
    </tr>
    </tbody>
    </table>
</details>
<details>

  ![image](https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/604460f9-478c-4cc1-ba35-287447c04b26)
  <summary>BLIP-2</summary>
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
    <td><a href="https://arxiv.org/abs/2301.12597">BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models</a></td>
    <td>BLIP-2 integrates frozen pre-trained image encoders and language models, leveraging a lightweight Querying Transformer (Q-Former) for bridging the modality gap between vision and language.</td>
    <td>Key components include frozen image encoders for visual representation, frozen large language models (LLMs) for textual understanding, and the Q-Former for extracting and integrating visual features relevant to textual queries.</td>
    <td>BLIP-2 employs a two-stage pre-training strategy. The first stage focuses on vision-language representation learning using frozen image encoders. The second stage involves vision-to-language generative learning leveraging frozen LLMs.</td>
    <td>The model uses learnable query vectors in the Q-Former to perform effective vision-language alignment.</td>
    <td>Fusion involves extracting language-informative visual representations via Q-Former, which are then integrated with LLMs to generate relevant textual outputs.</td>
    <td>COCO, Visual Genome, CC3M, CC12M, SBU, LAION400M</td>
    <td>These datasets facilitate comprehensive pre-training by providing diverse image-text pairs for learning visual representations and language generation tasks.</td>
    </tr>
    </tbody>
    </table>
</details>
<details>

  ![image](https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/5839e3a6-6fb8-469c-b84e-d60a851c1642)
  <summary>InstructBLIP</summary>
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
    <td><a href="https://arxiv.org/abs/2305.06500v2">InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning </a></td>
    <td>InstructBLIP builds upon the pretrained BLIP-2 models, incorporating an image encoder, a large language model (LLM), and a Query Transformer (Q-Former) to bridge the two. The architecture is designed for instruction tuning, with the Q-Former being fine-tuned while keeping the image encoder and LLM frozen.</td>
    <td>Key components include a pre-trained BLIP-2 model (image encoder and LLM) and the Query Transformer (Q-Former), which extracts instruction-aware visual features from the image encoder&#39;s output.</td>
    <td>InstructBLIP is trained on a diverse set of instruction data, utilizing a balanced sampling strategy to synchronize learning across datasets. It employs the standard language modeling loss for instruction tuning, with specific adaptations for datasets involving scene texts by adding OCR tokens.</td>
    <td>Utilizes the Query Transformer (Q-Former) to achieve instruction-aware visual feature extraction, enabling the model to adapt visual representations to the task instruction.</td>
    <td>The Q-Former interacts with the image encoder&#39;s output through cross attention, using instruction text tokens as additional input to extract task-relevant image features. These features are then fed as soft prompt input to the LLM.</td>
    <td>26 datasets across 11 task categories, including image captioning, visual reasoning, image question answering, and more.</td>
    <td>Datasets are transformed into instruction tuning format to train the model for a wide range of vision-language tasks and evaluate its zero-shot generalization ability on unseen data and tasks.</td>
    </tr>
    </tbody>
    </table>
</details>
<details>

  ![image](https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/50a9bbc0-b888-4407-800d-71880e248916)
  <summary>KOSMOS-1</summary>
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
    <td><a href="https://arxiv.org/abs/2302.14045">Language Is Not All You Need: Aligning Perception with Language Models</a></td>
    <td>KOSMOS-1 is a multimodal large language model that integrates general modalities perception, zero-shot learning, few-shot learning, and generates outputs in an auto-regressive manner. Its backbone is a Transformer-based causal language model that incorporates text and other modalities.</td>
    <td>Key components include a Transformer-based decoder for processing input sequences, embedding modules for encoding text and modalities into vectors, and MAGNETO and XPOS for architecture improvements.</td>
    <td>Trained on web-scale multimodal corpora including monomodal data, cross-modal paired data, and interleaved multimodal data. It utilizes next-token prediction tasks for learning, with a focus on maximizing the log-likelihood of tokens.</td>
    <td>Utilizes interleaved image-text data for aligning the perception of general modalities with language models.</td>
    <td>The embedding module encodes both text tokens and input modalities into vectors, which are then processed by the Transformer-based decoder, integrating vision and text through sequential processing.</td>
    <td>The Pile, Common Crawl, English LAION-2B, LAION-400M, COYO-700M, Conceptual Captions, interleaved image-text data from Common Crawl.</td>
    <td>Text corpora for representation learning and language tasks, image-caption pairs and interleaved data for aligning perception with language models, and improving few-shot abilities.</td>
    </tr>
    </tbody>
    </table>
</details>
<details>

  ![image](https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/17420c9c-759d-4690-bfc8-e8d7792111e7)
  <summary>KOSMOS-2</summary>
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
    <td><a href="https://arxiv.org/abs/2306.14824">KOSMOS-2: Grounding Multimodal Large Language Models to the World</a></td>
    <td>KOSMOS-2 is a grounded multimodal large language model that builds upon KOSMOS-1, integrating grounding and referring capabilities. It adopts the same Transformer-based causal language model architecture and training objectives as KOSMOS-1, with the addition of grounded image-text pairs to its training data.</td>
    <td>The model incorporates the grounding capability by training on a web-scale dataset of grounded image-text pairs (GRIT), utilizing continuous coordinates of bounding boxes converted into discrete location tokens, and linking these with text spans in a unified input representation.</td>
    <td>KOSMOS-2 was trained on grounded image-text pairs, monomodal text corpora, image-caption pairs, and interleaved image-text data. The training involved a large batch size and utilized the AdamW optimizer. The model was trained on 256 V100 GPUs, and the training process included instruction tuning with vision-language and language-only instruction datasets.</td>
    <td>The model&#39;s grounding technique involves converting the continuous coordinates of bounding boxes into discrete location tokens and linking these tokens with their corresponding text spans, effectively grounding text output to visual input.</td>
    <td>KOSMOS-2 uses a unified input representation that combines image embeddings with grounded text and location tokens, enabling the model to understand and refer to specific image regions or objects directly.</td>
    <td>GRIT, monomodal text corpora, image-caption pairs, and interleaved image-text data. GRIT is a large-scale dataset of grounded image-text pairs created for training KOSMOS-2.</td>
    <td>The GRIT dataset was specifically created to train the model with grounding capabilities, while the other datasets were used to enhance the model&#39;s language understanding, multimodal perception, and in-context learning abilities.</td>
    </tr>
    </tbody>
    </table>
</details>
<details>

  ![image](https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/bedfc8b1-7aff-44af-b605-4470ad030bdf)
  <summary>MULTIINSTRUCT</summary>
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
    <td><a href="https://arxiv.org/abs/2212.10773">MULTIINSTRUCT: Improving Multi-Modal Zero-Shot Learning via Instruction Tuning </a></td>
    <td>MULTIINSTRUCT uses OFA as the base pre-trained multimodal model, adopting a Transformer-based sequence-to-sequence framework for encoding instructions, text, images, and bounding boxes within a unified token space.</td>
    <td>The architecture components include a transformer-based encoder for processing inputs (including optional images) and instructions, and a transformer-based decoder for predicting outputs.</td>
    <td>The model is fine-tuned on the MULTIINSTRUCT dataset with instruction tuning. Training involves mixing instances from multiple tasks, random shuffling, and randomly sampling instruction templates for batch-based training. It also explores transfer learning from the NATURAL INSTRUCTIONS dataset through Mixed Instruction Tuning and Sequential Instruction Tuning.</td>
    <td>Uses byte-pair encoding and VQ-GAN for aligning text and image tokens within a unified vocabulary, enabling the model to process various input/output types seamlessly.</td>
    <td>Employs a unified sequence-to-sequence model architecture to encode multimodal inputs (text, images, bounding boxes) with instructions, facilitating deep integration and alignment of vision and language modalities.</td>
    <td>MULTIINSTRUCT, NATURAL INSTRUCTIONS</td>
    <td>MULTIINSTRUCT is used for fine-tuning the model with multimodal tasks and instructions. NATURAL INSTRUCTIONS is used for exploring transfer learning to enhance model&#39;s performance on multimodal tasks.</td>
    </tr>
    </tbody>
    </table>
</details>
<details>

  ![image](https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/8afc8259-fa72-4e52-8080-a4ea12208e32)
  <summary>LaVIN</summary>
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
    <td><a href="https://arxiv.org/abs/2305.15023v3">Cheap and Quick: Efficient Vision-Language Instruction Tuning for Large Language Models </a></td>
    <td>LaVIN introduces a novel learning regime, Mixture-of-Modality Adaptation (MMA), leveraging lightweight adapters for vision-language (VL) instruction tuning. This approach connects the image encoder and LLM, optimizing the entire multimodal LLM via a small number of parameters.</td>
    <td>Key components include Mixture-of-Modality Adapter (MM-Adapter) for connecting the LLM with the image encoder using lightweight adaptation modules, and Mixture-of-Modality Training (MMT) for joint optimization of multimodal LLM in an end-to-end manner.</td>
    <td>LaVIN employs MMA, enabling efficient training by only fine-tuning inserted adapters. This scheme reduces the number of optimized parameters to a small scale (3~5M), significantly cutting training time and storage costs without additional VL pre-training.</td>
    <td>MM-Adapter facilitates automatic shifting between single- and multi-modal instructions, enhancing adaptation to VL tasks.</td>
    <td>MM-Adapter dynamically adjusts adaptations for input features through a routing function, allowing efficient integration of vision and text embeddings.</td>
    <td>ScienceQA, Alphaca-52k, LLaVA-158k</td>
    <td>ScienceQA is used for evaluating multimodal question answering performance. Alphaca-52k (text-only) and LLaVA-158k (text-image pairs) datasets are utilized for tuning and extending LaVIN to a multimodal chatbot, demonstrating its superior vision-language understanding.</td>
    </tr>
    </tbody>
    </table>
</details>
<details>

  ![image](https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/3e7c93bc-7963-4c2e-b207-226a03d152ca)
  <summary>TinyGPT-V</summary>
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
    <td><a href="https://arxiv.org/abs/2312.16862v1">TinyGPT-V: Efficient Multimodal Large Language Model via Small Backbones</a></td>
    <td>TinyGPT-V architecture includes a visual encoder (EVA of ViT), linear projection layers, and the Phi-2 language model as its backbone. It utilizes Q-Former from BLIP-2 for initial linear projection, aiming to efficiently embed visual features into the language model.</td>
    <td>Visual encoder backbone (EVA of ViT), linear projection layers for embedding visual features, Phi-2 large language model with 2.7 billion parameters, normalization and LoRA mechanisms to stabilize training and enhance model&#39;s performance.</td>
    <td>TinyGPT-V&#39;s training comprises four stages: warm-up training with image-text pairs, pre-training to process image modality inputs, instruction fine-tuning with image-text pairings for human-like learning, and multi-task learning to enhance conversation abilities and multimodal instruction tuning.</td>
    <td>Uses linear projection layers and Q-Former for embedding visual features, LoRA for fine-tuning language model, normalization techniques (RMSNorm and LayerNorm) to stabilize training.</td>
    <td>Employs linear projection layers and the Q-Former layer from BLIP-2 architecture for initial embedding of visual features into the language model, ensuring efficient encoding and fusion of vision and text.</td>
    <td>LAION, CC3M, SBU, MiniGPT-4 Stage2 for CC &amp; SBU, Text Captions, RefCOCO, RefCOCO+, RefCOCOg, Visual Genome, GQA, VQAv2, OK-VQA, AOK-VQA, LLaVA dataset, Flickr30k, Multi-task conversation, Unnatural Instructions</td>
    <td>Used for various stages of training including warm-up, pre-training, instruction fine-tuning, and multi-task learning. Supports the model&#39;s capabilities in vision-language understanding, generation, and performing tasks like visual question answering, image captioning, referring expression comprehension, object parsing, and grounding.</td>
    </tr>
    </tbody>
    </table>
</details>
<details>

  ![image](https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/80e807cb-c2cf-491a-a3b4-1223afde1981)
  <summary>CoVLM</summary>
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
    <td><a href="https://arxiv.org/abs/2311.03354v1">CoVLM: Composing Visual Entities and Relationships in Large Language Models via Communicative Decoding</a></td>
    <td>CoVLM integrates a vision module and a language model (LLM) to achieve vision-language communicative decoding. It uses communication tokens for dynamic interaction between the detection network and the LLM.</td>
    <td>Image encoder (CLIP ViT-L), detection network (YOLOX), pre-trained Pythia model for LLM. Special communication tokens facilitate vision-language modeling and communication.</td>
    <td>CoVLM was pre-trained on a large-scale grounded image-text dataset consisting of 97M image-text pairs from various sources. It utilizes a grounding pipeline to associate text spans with corresponding visual entities in images.</td>
    <td>Utilizes special communication tokens for dynamic interaction and iterative communication between vision and language components, facilitating top-down language-to-vision and bottom-up vision-to-language communication.</td>
    <td>The model embeds visual and text features into a shared embedding space, enabling seamless integration and interaction between language tokens and visual embeddings.</td>
    <td>COCO, CC3M, CC12M, Visual Genome, SBU, LAION400M</td>
    <td>Used for pre-training CoVLM by grounding image-text pairs, facilitating the association of text descriptions with their corresponding visual entities.</td>
    </tr>
    </tbody>
    </table>
</details>
<details>
  <summary>FireLLaVA</summary>
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
    <td><a href="https://fireworks.ai/blog/firellava-the-first-commercially-permissive-oss-llava-model">FireLLaVA: the first commercially permissive OSS LLaVA model</a></td>
    <td>FireLLaVA is a multi-modal, commercially permissive VLM based on the LLaVA model framework, utilizing OSS models for data generation and training. It incorporates the CodeLlama 34B Instruct model for language understanding and leverages visual language conversations generated via bounding box labels and captions.</td>
    <td>The model combines a language component, CodeLlama 34B Instruct, for processing textual input and a vision component similar to OpenAI&#39;s CLIP-ViT for interpreting visual content.</td>
    <td>Training involved generating visual language conversations with a language-only OSS model by inputting bounding box labels and captions. The instruction fine-tuning stage utilized 588K lines of visual question answering or conversation data, combining permissive original LLaVA data and Fireworks.ai generated data.</td>
    <td>Utilizes bounding box labels and captions for generating training data, aligning text and image data.</td>
    <td>The model architecture likely involves embedding fusion at some stage to integrate vision and text inputs, though specific fusion methods are not detailed.</td>
    <td>Original LLaVA training data, Fireworks.ai generated data</td>
    <td>Used for instruction fine-tuning to enable the model to understand and generate responses based on both textual and visual inputs.</td>
    </tr>
    </tbody>
    </table>
</details>
<details>

  ![image](https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/0e5e214b-be64-4aac-aba4-04c97970b9de)
  <summary>MoE-LLaVA</summary>
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
    <td><a href="https://arxiv.org/abs/2401.15947">MoE-LLaVA: Mixture of Experts for Large Vision-Language Models</a></td>
    <td>MoE-LLaVA introduces a novel architecture integrating mixtures of experts (MoE) with learnable routers within a large vision-language model framework. It features a sparse model design where each token is routed to different experts, with only the top-k experts being activated for processing.</td>
    <td>Key components include a vision encoder, a visual projection layer (MLP), a word embedding layer, multi-head self-attention (MSA) blocks, feed-forward neural networks (FFN), and MoE blocks. The architecture uses layer normalization and residual connections within each block.</td>
    <td>The training employs a three-stage MoE-Tuning strategy. Stage I focuses on adapting image tokens to the LLM with an MLP. Stage II involves training all LLM parameters except the vision encoder for multimodal understanding. Stage III specializes in initializing and training the MoE layers exclusively, utilizing the FFNs from Stage II as the initialization weights for the experts.</td>
    <td>MoE-LLaVA employs learnable routers to dynamically distribute tokens to the most relevant experts for processing, effectively aligning text and image modalities.</td>
    <td>The model concatenates visual and text tokens after processing by the vision encoder and word embedding layer, respectively. These concatenated tokens are then processed through the LLM blocks and MoE blocks, allowing for a deep integration of visual and textual information.</td>
    <td>LLaVA-PT, Hybrid-FT, SViT, LVIS, LRV, MIMIC-IT, LLaVA-FT</td>
    <td>These datasets are used across different training stages to enhance the model&#39;s multimodal understanding capabilities. LLaVA-PT is used for pretraining in Stage I, Hybrid-FT (a combination of several datasets) for Stage II to bolster multimodal instruction tuning, and LLaVA-FT for Stage III focusing on fine-tuning the MoE layers.</td>
    </tr>
    </tbody>
    </table>
</details>
<details>

  ![image](https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/44c53b8a-ad35-4eca-a68b-63af32e6ccf1)
  <summary>BLIVA</summary>
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
    <td><a href="https://arxiv.org/abs/2308.09936v3">BLIVA: A Simple Multimodal LLM for Better Handling of Text-Rich Visual Questions</a></td>
    <td>BLIVA is an augmented version of InstructBLIP with Visual Assistant, designed to incorporate both learned query embeddings from InstructBLIP and directly projected encoded patch embeddings into the LLM, inspired by LLaVA. This architecture aims to capture intricate details in text-rich visual contexts that may be missed during the query decoding process.</td>
    <td>BLIVA includes a vision tower for encoding visual representations from input images into encoded patch embeddings, a Q-former to extract refined learned query embeddings, and a projection layer allowing the LLM to grasp rich visual knowledge. These components are combined and fed directly to the LLM.</td>
    <td>BLIVA employs a two-stage training scheme: pre-training with image-text pairs from captioning datasets to align the LLM with visual information, and instruction tuning using VQA data to enhance performance. It starts with pre-training the patch embeddings projection layer, followed by fine-tuning both the Q-former and the projection layer with instruction tuning data, keeping the image encoder and LLM frozen to avoid catastrophic forgetting.</td>
    <td>The model uses learned query embeddings with an additional visual assistant branch utilizing encoded patch embeddings. This approach addresses the limitations of image information typically provided to LLMs.</td>
    <td>BLIVA merges learned query embeddings with encoded patch embeddings to improve text-image visual perception. The embeddings are concatenated and fed directly to the LLM, appended immediately after the question text embedding.</td>
    <td>Image captioning datasets, instruction tuning VQA data, YTTB-VQA (YouTube Thumbnail Visual Question-Answer pairs)</td>
    <td>The image captioning datasets are used for pre-training to align the LLM with visual information. Instruction tuning VQA data is used in the second training stage to enhance the LLM&#39;s performance. YTTB-VQA is utilized to demonstrate BLIVA&#39;s capability in processing text-rich images and its applicability in real-world scenarios.</td>
    </tr>
    </tbody>
    </table>
</details>

<details>

  ![image](https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/bfba9ced-9b0a-4959-98d2-5150051d8548)
  <summary>MIRASOL3B</summary>
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
    <td><a href="https://arxiv.org/abs/2311.05698v2">MIRASOL3B: A Multimodal Autoregressive Model for Time-Aligned and Contextual Modalities</a></td>
    <td>MIRASOL3B is a multimodal autoregressive model that decouples the autoregressive modeling into separate components for time-aligned modalities (audio and video) and contextual modalities (text). It features a Combiner mechanism to fuse audio and video features into compact, expressive representations.</td>
    <td>The model includes two main components: 1) An autoregressive component for time-aligned modalities like audio and video, which processes inputs in smaller, roughly synchronized chunks. 2) A separate autoregressive component for contextual modalities, using combined latent space as cross-attention inputs.</td>
    <td>Training involves partitioning media inputs into smaller segments for efficient processing, using a Combiner to fuse audio and video features, and applying autoregressive modeling for both time-aligned and non-time-aligned modalities. The model uses a combination of losses including latent space reconstruction, video reconstruction, and unaligned text cross-entropy loss.</td>
    <td>Cross-attention weights facilitate the coordination between the autoregressive components for time-aligned and contextual modalities.</td>
    <td>The Combiner fuses audio and video features within concurrent timeframes into a joint representation, using techniques like Transformer and Token Turing Machine (TTM) for efficient feature combination and memory usage.</td>
    <td>Video-Text Pairs (VTP), MSRVTT-QA, VGG-Sound, ActivityNet-QA, NExT-QA, Epic-Sound, Kinetics-Sound</td>
    <td>The datasets were used for pretraining and fine-tuning the model across different modalities and tasks, demonstrating the model&#39;s effectiveness in multimodal understanding and generation, particularly in video question answering and audio-video benchmarks.  </td>
    </tr>
    </tbody>
    </table>
</details>
<details> 

  ![image](https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/4156475d-e501-495e-98bb-66efdd5b03f7)
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

  ![image](https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/b46ebf3e-67fc-401e-a6ea-6f4797da372d)
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

  ![image](https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/2565afb0-901c-4438-9488-c73a86261aa5)
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

  ![image](https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/67e5bbc7-1800-46e8-8ef1-b3b72a901a12)
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
    <td><a href="https://palm-e.github.io">PaLM-E: An Embodied Multimodal Language Model </a></td>
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

  ![image](https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/0e5ff945-1271-4189-8dd9-b0abd88eacc1)
  <summary>MiniGPT-4</summary>
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
    <td><a href="https://arxiv.org/abs/2304.10592v2">MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models</a></td>
    <td>MiniGPT-4 aligns a frozen visual encoder with a frozen advanced LLM, Vicuna, using one projection layer. It incorporates a vision encoder with a pretrained ViT and Q-Former, a single linear projection layer, and the Vicuna LLM, focusing on efficiently aligning visual features with language capabilities.</td>
    <td>Vision encoder (pretrained ViT, Q-Former), single linear projection layer, Vicuna large language model (LLM)</td>
    <td>MiniGPT-4 is initially trained for 20k steps using a batch size of 256 on 4 A100 GPUs, leveraging a combined image captioning dataset for aligning visual features with Vicuna. A second-stage finetuning employs 3500 detailed image description pairs to enhance generation reliability and naturalness.</td>
    <td>The model uses a single projection layer to align encoded visual features with the Vicuna language model, while keeping other components frozen.</td>
    <td>Alignment is achieved through a two-stage training approach: initial pretraining on image-text pairs for basic vision-language knowledge, followed by finetuning with a high-quality dataset for improved usability and natural language generation.</td>
    <td>Conceptual Captions, SBU, LAION, a curated dataset of 3500 detailed image descriptions</td>
    <td>Initial datasets are used for basic vision-language alignment. The curated dataset is for enhancing the model&#39;s ability to generate detailed and natural language outputs.</td>
    </tr>
    </tbody>
    </table>
</details>
<details>

  ![image](https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/2354442a-0e96-4010-8b4f-8bc3d666427e)
  <summary>MiniGPT-v2</summary>
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
    <td><a href="https://arxiv.org/abs/2310.09478v3">MiniGPT-v2: Large Language Model As a Unified Interface for Vision-Language Multi-task Learning</a></td>
    <td>MiniGPT-v2 consists of a visual backbone (ViT), a linear projection layer, and a large language model (LLaMA-2-chat 7B). The model architecture aims for efficient processing of high-resolution images (448x448) by concatenating every four neighboring visual tokens into one and projects them into the language model&#39;s feature space.</td>
    <td>Visual Backbone: ViT (frozen during training); Linear Projection Layer: Concatenates and projects visual tokens; Large Language Model: LLaMA-2-chat (7B), serving as a unified interface for vision-language tasks.</td>
    <td>MiniGPT-v2 uses a three-stage training strategy focusing on broad vision-language knowledge acquisition with weakly-labeled and fine-grained datasets initially, then on fine-grained data for task improvement, and finally on multi-modal instruction and language datasets for enhanced multi-modal instruction response.</td>
    <td>Utilizes task-specific identifier tokens for different vision-language tasks to reduce ambiguity and improve task distinction during training.</td>
    <td>The model projects concatenated visual tokens into the language model&#39;s space for efficient processing and relies on language tokens for executing various vision-language tasks, integrating visual and textual information through linear projection and task-specific training.</td>
    <td>LAION, CC3M, SBU, GRIT-20M, COCO caption, Text Captions, RefCOCO, RefCOCO+, RefCOCOg, Visual Genome, GQA, VQAv2, OCR-VQA, OK-VQA, AOK-VQA, LLaVA dataset, Flickr30k, Multi-task conversation, Unnatural Instructions</td>
    <td>To train the model across different stages focusing on broad knowledge acquisition, task-specific improvements, and multi-modal instruction handling.</td>
    </tr>
    </tbody>
    </table>  
</details>
<details>

  ![image](https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/1ede1c4f-bdeb-48e0-ae8e-ccfbee1dea51)
  <summary>LLaVA-Plus</summary>
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
    <td><a href="https://arxiv.org/abs/2311.05437">LLaVA-Plus: Learning to Use Tools for Creating Multimodal Agents</a></td>
    <td>LLaVA-Plus integrates a wide range of vision and vision-language pre-trained models into a skill repository, systematically expanding the capabilities of large multimodal models through end-to-end training. It activates relevant tools based on users&#39; multimodal inputs, combining their execution results on-the-fly.</td>
    <td>Skill repository containing vision and vision-language models, end-to-end trained multimodal instruction-following capabilities, and a unified scheme for representing multimodal instruction-following data.</td>
    <td>Trained on curated multimodal instruction-following data covering visual understanding, generation, external knowledge retrieval, and their compositions. Incorporates new tools via instruction tuning, expanding abilities by learning to use these tools effectively.</td>
    <td>Uses raw visual signals throughout human-AI interaction sessions for improved tool use performance, planning, and reasoning.</td>
    <td>Combines user inputs, tool activation prompts, and execution results in a unified dialogue format, facilitating seamless integration of vision and text embeddings.</td>
    <td>COCO, HierText, InfoSeek, JourneyDB, Instruct P2P</td>
    <td>Used for training on visual understanding skills such as detection, segmentation, captioning, OCR, and external knowledge retrieval, as well as for generation tasks and skill compositions.</td>
    </tr>
    </tbody>
    </table>  
</details>
<details>
  <summary>BakLLaVA</summary>
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
    <td><a href="https://huggingface.co/SkunkworksAI/BakLLaVA-1">BakLLaVA</a></td>
    <td>BakLLaVA introduces significant architecture changes to the original LLaVA implementation, focusing on baking state-of-the-art multimodality into language models.</td>
    <td>Custom datasets, modified training process, better base models</td>
    <td>BakLLaVA training involves a feature alignment stage using 600K filtered CC3M for vision-language connection, followed by a visual instruction tuning stage with 150K GPT-generated multimodal instructions.</td>
    <td>Feature alignment stage for connecting vision encoder to language models</td>
    <td>Visual instruction tuning for encoding vision and text together</td>
    <td>CC3M, GPT-generated multimodal instructions, COCO, LAION-CC-SBU</td>
    <td>Feature alignment, visual instruction tuning, broad concept coverage and efficiency in training</td>
    </tr>
    </tbody>
    </table>
</details>
<details>
  <summary>CogVLM</summary>
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
    <td><a href="https://arxiv.org/abs/2311.03079v2">CogVLM: Visual Expert for Pretrained Language Models</a></td>
    <td>CogVLM integrates visual and linguistic features by adding a trainable visual expert module to each layer of a pretrained large language model, enabling deep fusion of vision-language features.</td>
    <td>Vision Transformer (ViT) encoder, MLP adapter, pretrained large language model (GPT), visual expert module</td>
    <td>Pretraining includes image captioning loss and Referring Expression Comprehension (REC) over 1.5B image-text pairs and a visual grounding dataset of 40M images. Training also involves unified instruction-supervised fine-tuning across diverse visual question-answering datasets.</td>
    <td>Deep visual-language feature alignment via a visual expert module with QKV matrix and MLP in each layer.</td>
    <td>Enables the incorporation of image features into the language model&#39;s processing layers, facilitating a deeper integration of visual and textual data.</td>
    <td>LAION-2B, COYO-700M, visual grounding dataset of 40M images, VQAv2, OKVQA, TextVQA, OCRVQA, ScienceQA</td>
    <td>Used for pretraining and instruction alignment phase, including tasks like image captioning and referring expression comprehension.</td>
    </tr>
    </tbody>
    </table>  
</details>
<details>
  <summary>FERRET</summary>
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
    <td><a href="https://arxiv.org/abs/2310.07704v1">FERRET: Refer and Ground Anything Anywhere at Any Granularity</a></td>
    <td>FERRET is a multimodal large language model (MLLM) designed for understanding spatial referring of any shape or granularity within an image and accurately grounding open-vocabulary descriptions. It utilizes a hybrid region representation integrating discrete coordinates and continuous features to represent image regions.</td>
    <td>Key components include an image encoder for extracting image embeddings, a novel spatial-aware visual sampler for extracting regional continuous features, and a language model for modeling image, text, and region features jointly.</td>
    <td>FERRET is trained on the GRIT dataset, containing 1.1M samples with hierarchical spatial knowledge. Training involves spatial-aware visual sampling, handling varying shapes and sparsity, and generating coordinates for groundable objects alongside text generation.</td>
    <td>Hybrid region representation and spatial-aware visual sampling for fine-grained alignment between text and image regions.</td>
    <td>Combines discrete coordinates and continuous visual features for input regions, enabling the model to process free-formed region inputs and accurately ground descriptions in outputs.</td>
    <td>GRIT, Visual Genome, RefCOCOs, Flickr30k</td>
    <td>GRIT for training with rich hierarchical spatial knowledge. Visual Genome, RefCOCOs, Flickr30k for object detection, phrase grounding, and evaluating model&#39;s referring and grounding capabilities.</td>
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
<details>
  <summary>Fuyu-8B</summary>
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
  <td><a href="https://www.adept.ai/blog/fuyu-8b">Fuyu-8B: A Multimodal Architecture for AI Agents</a></td>
  <td>Fuyu-8B is a simplified multimodal model designed for digital agents, supporting arbitrary image resolutions and fine-grained localization. It has a decoder-only transformer architecture without a specialized image encoder, enabling direct projection of image patches into the transformer&#39;s first layer.</td>
  <td>Decoder-only transformer, linear projection of image patches, simplified training and inference, support for arbitrary image resolutions.</td>
  <td>Simplified compared to other models, Fuyu-8B eliminates the need for a separate image encoder and multiple training stages, allowing for direct training on images of any size without complex contrastive objectives or resolution-specific phases.</td>
  <td>Uses direct projection of image patches into the transformer, avoiding the need for cross-attention or adapters between separate encoders and decoders.</td>
  <td>Image and text embeddings are combined from the outset by treating image tokens similarly to text tokens, without separate position embeddings for images, simplifying the alignment.</td>
  <td>VQAv2, OKVQA, COCO Captions, AI2D</td>
  <td>Used to evaluate the model&#39;s performance on standard image understanding tasks like visual question-answering and captioning, despite the model&#39;s focus on digital agent applications.</td>
  </tr>
  </tbody>
  </table>
</details>
<details>
  <summary>LLaVA-Med</summary>
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
    <td><a href="https://huggingface.co/microsoft/llava-med-7b-delta">LLaVA-Med: Large Language and Vision Assistant for BioMedicine</a></td>
    <td>LLaVA-Med is a large language and vision model for the biomedical domain, derived from the general-domain LLaVA. It uses curriculum learning for continuous training, beginning with biomedical concept alignment before progressing to full-blown instruction tuning.</td>
    <td>The model integrates language and vision capabilities, starting with a foundation in LLaVA and enhancing it with specialized biomedical training.</td>
    <td>Initiated with LLaVA&#39;s general-domain foundation, LLaVA-Med undergoes curriculum learning, emphasizing biomedical concept alignment followed by instruction tuning. This approach is designed for open-ended biomedical question answering, leveraging datasets like PathVQA and VQA-RAD.</td>
    <td>Curriculum learning for concept alignment and instruction tuning.</td>
    <td>Combines vision and language processing for biomedical applications, adapting LLaVA to the biomedical domain through targeted curriculum learning.</td>
    <td>PMC-15M</td>
    <td>A large-scale parallel image-text dataset from PubMed Central, with 15 million figure-caption pairs across various biomedical imagery, used for vision-language processing in biomedicine.</td>
    </tr>
    </tbody>
    </table>
</details>
<details>
  <summary>SPHINX</summary>
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
    <td><a href="https://arxiv.org/abs/2311.07575v1">SPHINX: The Joint Mixing of Weights, Tasks, and Visual Embeddings for Multi-Modal Large Language Models</a></td>
    <td>SPHINX is a multi-modal large language model (MLLM) that integrates model weight mixing, tuning tasks, and visual embeddings for enhanced vision-language alignment. It unfreezes the large language model during pre-training for cross-modal learning.</td>
    <td>The model includes a mix of vision encoders, two linear projection layers, and utilizes LLaMA-2 as the language model backbone. It employs a two-stage training paradigm comprising pre-training for vision-language alignment and fine-tuning for visual instruction-following.</td>
    <td>SPHINX uses a joint mixing strategy for model weights, tuning tasks, and visual embeddings. It involves pre-training with mixed real-world and synthetic data for robust cross-modal knowledge, followed by multi-task fine-tuning to cover a wide range of visual instruction tasks. Additionally, it introduces an efficient strategy for handling high-resolution images through mixed scales and sub-images.</td>
    <td>The model achieves vision-language alignment by unfreezing the LLM during pre-training, mixing model weights from different domains, and integrating comprehensive visual embeddings.</td>
    <td>SPHINX utilizes a weight-mixing strategy for domain-specific knowledge and a comprehensive multi-task training paradigm for visual instruction following. It mixes visual embeddings from different network architectures and training paradigms to enhance vision-language alignment.</td>
    <td>LAION-400M, LAION-COCO, RefinedWeb, VQAV2, GQA, OKVQA, A-OKVQA, OCRVQA, TextCaps, COCO, LVIS, RefCOCO, VG, Flickr30k</td>
    <td>Used for multi-modal alignment, language-only tuning, visual question answering, general vision tasks like object detection and human pose estimation, referring object localization, and understanding descriptions in the context of image regions.</td>
    </tr>
    </tbody>
    </table>
</details>
