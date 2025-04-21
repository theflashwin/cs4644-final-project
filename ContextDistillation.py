from transformers import Trainer
import torch
import torch.nn.functional as torch_func

class ContextDistillationTrainer(Trainer):

    def __init__(self, teacher_model, temp=2.0, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.temp = temp
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        
        # pop the labels
        labels = inputs.pop("labels")

        # perform forward pass of the student
        student_out = model(**inputs, labels=labels)
        student_logits = student_out.logits
        student_loss = student_out.loss

        with torch.no_grad():
            teacher_logits = self.teacher(**inputs).logits

        student_log_probs = torch_func.log_softmax(student_logits / self.temp, dim=-1)
        teacher_probs = torch_func.softmax(teacher_logits / self.temp, dim=-1)

        kl_loss = torch_func.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.temp**2)
        loss = self.alpha * kl_loss + (1 - self.alpha) * student_loss

        return loss, student_out if return_outputs else loss