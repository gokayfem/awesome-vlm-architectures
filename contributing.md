# Contribution Guidelines
Fill the **submission dictionary** in the ```contribution.py``` then run it and send a pull request.
- Description should be at least 100 words.

```python
submission = {
        "title": "Example Architecture",
        "authors": "John Doe, Jane Doe",
        "badges": {
            "arXiv": "https://arxiv.org/abs/2304.08485",
            "GitHub": "https://github.com/haotian-liu/LLaVA",
            "Gradio": "",
            "Model": "https://huggingface.co/NousResearch/Nous-Hermes-2-Vision-Alpha"
        },
        "image_url": "https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/fbf8bcdd-b1bb-4fd8-8723-3c82e84ef759",
        "description": "Represents a notable advancement in the realm of Vision-Language Models, Training methodologies for ViT are characterized by its scalability and the significant impact of dataset size on its performance. Initially, ViT exhibits modest accuracies without strong regularization techniques. However, its performance escalates with the scale of training, showcasing its potential to outperform traditional CNN approaches through extensive pre-training on large datasets. This process highlights the critical role of dataset selection in ViT's training regimen. It is fine-tuned on smaller datasets following a comprehensive pre-training phase that leverages large datasets like ImageNet-21k and JFT-300M to enhance model generalization and performance across a wide range of tasks. The datasets employed, including ImageNet, CIFAR-100, VTAB, ImageNet-21k, and JFT-300M, serve dual purposes: benchmarking the model's image classification capabilities and evaluating its transferability to diverse tasks with limited data, thereby establishing ViT's versatility and effectiveness in advancing image recognition tasks..."
    }
```
