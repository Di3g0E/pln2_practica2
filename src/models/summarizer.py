import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Summarizer:
    """
    Clase que proporciona funcionalidad de resumen automático de textos
    utilizando modelos seq2seq, como BART.
    """

    def __init__(
            self,
            model_name: str = "facebook/bart-base",
            device: torch.device | None = None
    ):
        """
        Constructor de la clase Summarizer.

        :param model_name: Nombre del modelo preentrenado de Hugging Face.
        :type model_name: str
        :param device: Dispositivo para PyTorch.
        :type device: torch.device | None
        """
        self.model_name = model_name
        self.device = device if device is not None else torch.device("cpu")
        self.__tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.__gen_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name).to(device)

    def summarize(
            self,
            text: str,
            max_length: int = 250
    ) -> str:
        """
        Genera un resumen automático del texto proporcionado.

        :param text: Texto a resumir.
        :type text: str
        :param max_length: Longitud máxima del resumen generado.
        :type max_length: int
        :return: Resumen generado del texto.
        :rtype: str
        """
        input_text = f"summarize: {text}"
        inputs = self.__tokenizer.encode(
            input_text, return_tensors="pt", max_length=512, truncation=True)
        inputs = inputs.to(self.device)

        with torch.no_grad():
            summary_ids = self.__gen_model.generate(
                inputs,
                max_length=max_length,
                early_stopping=True,
                temperature=1.0
            )

        summary = self.__tokenizer.decode(
            summary_ids[0], skip_special_tokens=True)
        return summary
