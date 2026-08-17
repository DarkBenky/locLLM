import torch
import questionary
from questionary import Style, Validator, ValidationError

custom_style = Style([
    ("qmark", "fg:#00d7ff bold"),
    ("question", "bold"),
    ("pointer", "fg:#00d7ff bold"),
    ("highlighted", "fg:#00d7ff bold"),
    ("selected", "fg:#00d7af"),
])


class NumberValidator(Validator):
    def validate(self, document):
        if not document.text.isdigit():
            raise ValidationError(
                message="Please enter a valid number",
                cursor_position=len(document.text),
            )


def get_gpus():
    if not torch.cuda.is_available():
        return []

    gpu_list = []
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        vram_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
        gpu_list.append({"name": name, "vram_size": vram_gb, "index": i})
    return gpu_list


def select_gpus(gpus=None):
    """
    Prompt the user to pick one or more GPUs from a checkbox list.
    Returns a list of GPU dicts, or None if the user cancelled (Ctrl+C).
    """
    if gpus is None:
        gpus = get_gpus()

    if not gpus:
        questionary.print("No CUDA GPUs found.", style="bold fg:#ff0000")
        return []

    selected = questionary.checkbox(
        "Select GPU(s)",
        choices=[
            questionary.Choice(
                title=f"{gpu['name']} ({gpu['vram_size']:.1f}GB)",
                value=gpu,
            )
            for gpu in gpus
        ],
        style=custom_style,
    ).ask()

    return selected


def prompt_batch_config(gpu):
    """
    Ask for batch size and accumulation steps for a single GPU dict,
    store them on the dict, and return it.
    """
    batch_size = questionary.text(
        "Batch size", validate=NumberValidator, style=custom_style
    ).ask()
    accumulation_steps = questionary.text(
        "Accumulation steps", validate=NumberValidator, style=custom_style
    ).ask()

    gpu["batch_size"] = int(batch_size)
    gpu["accumulation_steps"] = int(accumulation_steps)
    return gpu


def prompt_optimizer():
    choice = questionary.select(
        "Optimizer",
        choices=[
            questionary.Choice(title="AdamW (fp32) — default", value="fp32"),
            questionary.Choice(
                title="AdamW 8-bit (bitsandbytes) — saves ~7 GB VRAM", value="8bit"
            ),
        ],
        default="fp32",
        style=custom_style,
    ).ask()
    return choice if choice in ("fp32", "8bit") else "fp32"


def print_gpu_summary(gpu):
    questionary.print(
        f"\nSelected GPU: {gpu['name']} ({gpu['vram_size']:.1f}GB)",
        style="bold fg:#00d7ff",
    )
    questionary.print(f"Batch size: {gpu['batch_size']}", style="fg:#00d7af")
    questionary.print(
        f"Accumulation steps: {gpu['accumulation_steps']}", style="fg:#00d7af"
    )
    questionary.print(
        f"Effective batch size: {gpu['batch_size'] * gpu['accumulation_steps']}",
        style="bold fg:#00d7af",
    )
    questionary.print(
        f"Optimizer: {gpu.get('optimizer', 'fp32')}", style="fg:#00d7af"
    )


def select_gpus_with_config():
    selected = select_gpus()
    if not selected:
        return selected

    for gpu in selected:
        prompt_batch_config(gpu)
        gpu["optimizer"] = prompt_optimizer()
        print_gpu_summary(gpu)

    return selected

def select_only_gpu():
    selected = select_gpus()
    if not selected:
        return selected

    return selected


if __name__ == "__main__":
    result = select_gpus_with_config()
    if result is None:
        exit(1)