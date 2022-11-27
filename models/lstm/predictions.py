import os
import sys
import pandas as pd
from tqdm.auto import tqdm
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.lstm.model import SmileAuthenticityPredictor
from models.lstm.dataset import FacesFeaturesDataset
from models._config import nns_config as nns_conf
from config import FFS_COLS_NAMES, CLASSES, CLASSES_STRS


def show_conf_matrix(conf_matrix):
    hmap = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=0, ha='right')
    plt.ylabel('True authenticity')
    plt.xlabel('Predicted authenticity')
    plt.show()


def review_predictions(trainer, test_data):
    trained_model = SmileAuthenticityPredictor.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        num_features=len(FFS_COLS_NAMES),
        num_classes=nns_conf.num_classes
    )
    trained_model.freeze()

    test_dataset = FacesFeaturesDataset(test_data)
    predictions, auths = [], []

    for item in tqdm(test_dataset):
        ffs = item['faces_features']
        auth = item['authenticity']

        _, output = trained_model(ffs.unsqueeze(dim=0))
        prediction = torch.argmax(output, dim=1)
        predictions.append(prediction.item())
        auths.append(auth.item())

    print(classification_report(auths, predictions, target_names=CLASSES_STRS))

    cm = confusion_matrix(auths, predictions)
    cm_df = pd.DataFrame(
        cm, index=CLASSES, columns=CLASSES
    )

    show_conf_matrix(cm_df)

