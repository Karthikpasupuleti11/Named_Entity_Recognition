# 👑 GLiNER: Generalist and Lightweight Model for Named Entity Recognition

---

<div align="center">
    <div>
        <a href="https://clickpy.clickhouse.com/dashboard/gliner"><img src="https://static.pepy.tech/badge/gliner" alt="GLiNER Downloads"></a>
        <a href="https://arxiv.org/abs/2311.08526"><img src="https://img.shields.io/badge/arXiv-2311.08526-b31b1b.svg" alt="GLiNER Paper"></a>
        <a href="https://discord.gg/Y2yVxpSQnG"><img alt="GLiNER Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a>
        <a href="https://github.com/urchade/GLiNER"><img alt="GLiNER GitHub stars" src="https://img.shields.io/github/stars/urchade/GLiNER?style=social"></a>
        <a href="https://github.com/urchade/GLiNER/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/urchade/GLiNER?color=blue"></a>
        <br>
        <a href="https://colab.research.google.com/drive/1mhalKWzmfSTqMnR0wQBZvt9-ktTsATHB?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open GLiNER In Colab"></a>
        <a href="https://huggingface.co/spaces/urchade/gliner_mediumv2.1"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg" alt="Open GLiNER In HF Spaces"></a>
        <a href="https://huggingface.co/models?library=gliner&sort=trending"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow" alt="HuggingFace Models"></a>
    </div>
    <br>
</div>

GLiNER is a framework for training and deploying Named Entity Recognition (NER) models that can identify any entity type using bidirectional transformer encoders (BERT-like). Beyond standard NER, GLiNER supports multiple tasks including joint entity and relation extraction through specialized architectures. It provides a practical alternative to both traditional NER models, which are limited to predefined entity types, and Large Language Models (LLMs), which offer flexibility but require significant computational resources.


### Installation
```bash
!pip install gliner
```

### Usage
After the installation of the GLiNER library, import the `GLiNER` class. Following this, you can load your chosen model with `GLiNER.from_pretrained` and utilize `predict_entities` to discern entities within your text.

```python
from gliner import GLiNER

# Initialize GLiNER with the base model
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

# Sample text for entity prediction
text = """
Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃˈtjɐnu ʁɔˈnaldu]; born 5 February 1985) is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and the Portugal national team. Widely regarded as one of the greatest players of all time, Ronaldo has won five Ballon d'Or awards,[note 3] a record three UEFA Men's Player of the Year Awards, and four European Golden Shoes, the most by a European player. He has won 33 trophies in his career, including seven league titles, five UEFA Champions Leagues, the UEFA European Championship and the UEFA Nations League. Ronaldo holds the records for most appearances (183), goals (140) and assists (42) in the Champions League, goals in the European Championship (14), international goals (128) and international appearances (205). He is one of the few players to have made over 1,200 professional career appearances, the most by an outfield player, and has scored over 850 official senior career goals for club and country, making him the top goalscorer of all time.
"""

# Labels for entity prediction
# Most GLiNER models should work best when entity types are in lower case or title case
labels = ["Person", "Award", "Date", "Competitions", "Teams"]

# Perform entity prediction
entities = model.predict_entities(text, labels, threshold=0.5)

# Display predicted entities and their labels
for entity in entities:
    print(entity["text"], "=>", entity["label"])
```

#### Expected Output

```
Cristiano Ronaldo dos Santos Aveiro => person
5 February 1985 => date
Al Nassr => teams
Portugal national team => teams
Ballon d'Or => award
UEFA Men's Player of the Year Awards => award
European Golden Shoes => award
UEFA Champions Leagues => competitions
UEFA European Championship => competitions
UEFA Nations League => competitions
European Championship => competitions
```
