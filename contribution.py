import json
import os

# Function to generate badge Markdown from URLs
def generate_badges_markdown(badges):
    markdown_badges = []
    for badge, url in badges.items():
        if url:  # Only generate Markdown for badges with URLs
            if badge == "arXiv":
                markdown_badges.append(f"[![arXiv](https://img.shields.io/badge/arXiv-{url.split('/')[-1]}-b31b1b.svg?style=flat-square)]({url})")
            elif badge == "GitHub":
                markdown_badges.append(f"[![GitHub](https://badges.aleen42.com/src/github.svg)]({url})")
            elif badge == "Gradio":
                markdown_badges.append(f"[![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)]({url})")
            elif badge == "Model":
                markdown_badges.append(f"[![Model](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md-dark.svg)]({url})")
    return " ".join(markdown_badges)

# Function to validate submission
def validate_submission(submission):
    errors = []
    
    # Validate description length (example: at least 100 words)
    if len(submission['description'].split()) < 100:
        errors.append("Description is too short. Please ensure it has at least 100 words.")
    
    return errors

# Function to generate Markdown content for the submission
def generate_markdown(submission):
    title_markdown = f"### **{submission['title']}**  \n"
    badges_markdown = generate_badges_markdown(submission['badges'])
    authors_markdown = f"{submission['authors']}  \n"
    image_markdown = f"""<p align="center">\n<img src="{submission['image_url']}" />\n</p>"""  if submission['image_url'] else ""
    description_markdown = f"\n<details>\n\n{submission['description']}\n</details>  \n"
    
    return f"{title_markdown}{badges_markdown}  \n{authors_markdown}{image_markdown}{description_markdown}  "

# Function to insert Markdown content into README.md before "Important References"
def insert_content_into_readme(markdown_content, readme_path="README.md"):
    if os.path.exists(readme_path):
        with open(readme_path, 'r') as file:
            readme_content = file.readlines()

        insertion_point = None
        for i, line in enumerate(readme_content):
            if "📚 **Important References**:" in line:
                insertion_point = i
                break

        if insertion_point is not None:
            readme_content.insert(insertion_point, "\n" + markdown_content + "\n")
            with open(readme_path, 'w') as file:
                file.writelines(readme_content)
            print("Content inserted into README.md successfully.")
        else:
            print("Could not find the insertion point in README.md.")
    else:
        print(f"Error: {readme_path} does not exist.")

# Example usage
if __name__ == "__main__":
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
    
    errors = validate_submission(submission)
    if errors:
        print("Submission has errors:")
        for error in errors:
            print(f"- {error}")
    else:
        markdown_content = generate_markdown(submission)
        print("Submission is valid! Generated Markdown Content:\n")
        # Print the markdown content or write it to a file as needed
        insert_content_into_readme(markdown_content)
       
