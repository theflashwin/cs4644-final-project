from transformers import Trainer
import torch
import torch.nn.functional as torch_func

class ContextDistillationTrainer(Trainer):
    def __init__(self, teacher_model, temp=2.0, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.temp = temp
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")

        # student forward
        student_out = model(**inputs, labels=labels)
        student_logits = student_out.logits
        student_ce_loss = student_out.loss

        # teacher forward on same device
        with torch.no_grad():
            teacher_logits = self.teacher(**inputs).logits

        # distillation loss
        s_log = torch_func.log_softmax(student_logits / self.temp, dim=-1)
        t_prob = torch_func.softmax(   teacher_logits / self.temp, dim=-1)
        kl_loss = torch_func.kl_div(s_log, t_prob, reduction="batchmean") * (self.temp ** 2)

        loss = self.alpha * kl_loss + (1.0 - self.alpha) * student_ce_loss
        return (loss, student_out) if return_outputs else loss

    # override RNG loading to skip any unpickling errors
    def _load_rng_state(self, resume_from_checkpoint):
        try:
            super()._load_rng_state(resume_from_checkpoint)
        except Exception as e:
            print(f"⚠️ Could not restore RNG state ({e}); continuing without it.")
