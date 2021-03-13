from sklearn import metrics as metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils import check_matplotlib_support

from train_on_gcp.predict_using_GCP_trained_model import get_dataset_from_category_folder, \
    get_dataset_predictions, FILE_NAME, GT_LABEL, PREDICTED_LABEL
from train_on_gcp.set_up_a_container import mount_model_to_container
import matplotlib.pyplot as plt
import numpy as np

PORT = 8080

def local_plot_confusion_matrix(results,estimator, X, y_true, *, labels=None,
                          sample_weight=None, normalize=None,
                          display_labels=None, include_values=True,
                          xticks_rotation='horizontal',
                          values_format=None,
                          cmap='viridis', ax=None, colorbar=True):

    check_matplotlib_support("plot_confusion_matrix")

    # if not is_classifier(estimator):
    #     raise ValueError("plot_confusion_matrix only supports classifiers")

    y_pred = estimator.predict(X)
    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight,
                          labels=labels, normalize=normalize)

    if display_labels is None:
        if labels is None:
            display_labels = unique_labels(y_true, y_pred)
        else:
            display_labels = labels

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=display_labels)
    return disp.plot(include_values=include_values,
                     cmap=cmap, ax=ax, xticks_rotation=xticks_rotation,
                     values_format=values_format, colorbar=colorbar)

# def plot_confusion_matrix(results, sample_weight=None, normalize=None, display_labels=None, include_values=True,
#                           xticks_rotation='horizontal', values_format=None, cmap='viridis', ax=None, colorbar=True):
#     confusion_matrix = metrics.confusion_matrix(y_true=results[GT_LABEL],
#                                                 y_pred=results[PREDICTED_LABEL],
#                                                 labels=results[GT_LABEL],
#                                                 sample_weight=None,
#                                                 normalize=None)
#
#     disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=display_labels)
#     # return disp.plot(include_values=include_values,
#     #                  cmap=cmap, ax=ax, xticks_rotation=xticks_rotation,
#     #                  values_format=values_format, colorbar=colorbar)
#
#     np.set_printoptions(precision=2)
#
#     # Plot non-normalized confusion matrix
#     titles_options = [("Confusion matrix, without normalization", None),
#                       ("Normalized confusion matrix", 'true')]
#     for title, normalize in titles_options:
#         # disp = plot_confusion_matrix(classifier, X_test, y_test,
#         #                              display_labels=class_names,
#         #                              cmap=plt.cm.Blues,
#         #                              normalize=normalize)
#         # disp.ax_.set_title(title)
#
#         print(title)
#         print(disp.confusion_matrix)
#
#     plt.show()

if __name__ == '__main__':
    # mount model:
    contain_name = 'skirts_len'
    model_port = 8080
    model_path = '/home/idan/tmp/mounted_model/0001/'
    sudo_pass = 'eee123rrr'
    # mount_model_to_container(contain_name, model_port, model_path, sudo_pass) (first run only)

    sample_data_path = '/home/idan/AskLilyData/Validated/skirt_length/evaluate_skirt_length_images'

    # create dataset of form: [FILE_NAME, GT_LABEL, IMAGE_PATH]
    dataset_df = get_dataset_from_category_folder(sample_data_path)

    # Dataset images are predicted to a output of form: [FILE_NAME, PREDICTED_LABEL, CONFIDENCE]
    predicted_df = get_dataset_predictions(dataset_df, model_port)

    # The gt dataset and predictions are merged:
    result_df = dataset_df.merge(predicted_df, on=FILE_NAME)

    # Get confusion matrix:
    matrix = local_plot_confusion_matrix(result_df)

    print('fd')
