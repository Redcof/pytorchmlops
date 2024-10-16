from sklearn.metrics import classification_report
from torch.nn import Module


class ClassificationReport(Module):
    def __init__(self, threshold: float = 0.5, labels=None, target_names=None, sample_weight=None, digits=2,
                 zero_division='warn'):
        """Build a text report showing the main classification metrics.

            Read more in the :ref:`User Guide <classification_report>`.

            Parameters
            ----------

            labels : array-like of shape (n_labels,), default=None
                Optional list of label indices to include in the report.

            target_names : array-like of shape (n_labels,), default=None
                Optional display names matching the labels (same order).

            sample_weight : array-like of shape (n_samples,), default=None
                Sample weights.

            digits : int, default=2
                Number of digits for formatting output floating point values.
                When ``output_dict`` is ``True``, this will be ignored and the
                returned values will not be rounded.

            output_dict : bool, default=False
                If True, return output as dict.

                .. versionadded:: 0.20

            zero_division : {"warn", 0.0, 1.0, np.nan}, default="warn"
                Sets the value to return when there is a zero division. If set to
                "warn", this acts as 0, but warnings are also raised.

                .. versionadded:: 1.3
                   `np.nan` option was added.

            Returns
            -------
            report : str or dict
                Text summary of the precision, recall, F1 score for each class.
                Dictionary returned if output_dict is True. Dictionary has the
                following structure::

                    {'label 1': {'precision':0.5,
                                 'recall':1.0,
                                 'f1-score':0.67,
                                 'support':1},
                     'label 2': { ... },
                      ...
                    }

                The reported averages include macro average (averaging the unweighted
                mean per label), weighted average (averaging the support-weighted mean
                per label), and sample average (only for multilabel classification).
                Micro average (averaging the total true positives, false negatives and
                false positives) is only shown for multi-label or multi-class
                with a subset of classes, because it corresponds to accuracy
                otherwise and would be the same for all metrics.
                See also :func:`precision_recall_fscore_support` for more details
                on averages.

                Note that in binary classification, recall of the positive class
                is also known as "sensitivity"; recall of the negative class is
                "specificity".

            See Also
            --------
            precision_recall_fscore_support: Compute precision, recall, F-measure and
                support for each class.
            confusion_matrix: Compute confusion matrix to evaluate the accuracy of a
                classification.
            multilabel_confusion_matrix: Compute a confusion matrix for each class or sample.

            Examples
            --------
            >>> from sklearn.metrics import classification_report
            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]
            >>> target_names = ['class 0', 'class 1', 'class 2']
            >>> print(classification_report(y_true, y_pred, target_names=target_names))
                          precision    recall  f1-score   support
            <BLANKLINE>
                 class 0       0.50      1.00      0.67         1
                 class 1       0.00      0.00      0.00         1
                 class 2       1.00      0.67      0.80         3
            <BLANKLINE>
                accuracy                           0.60         5
               macro avg       0.50      0.56      0.49         5
            weighted avg       0.70      0.60      0.61         5
            <BLANKLINE>
            >>> y_pred = [1, 1, 0]
            >>> y_true = [1, 1, 1]
            >>> print(classification_report(y_true, y_pred, labels=[1, 2, 3]))
                          precision    recall  f1-score   support
            <BLANKLINE>
                       1       1.00      0.67      0.80         3
                       2       0.00      0.00      0.00         0
                       3       0.00      0.00      0.00         0
            <BLANKLINE>
               micro avg       1.00      0.67      0.80         3
               macro avg       0.33      0.22      0.27         3
            weighted avg       1.00      0.67      0.80         3
            <BLANKLINE>
        """
        super().__init__()
        self.labels = labels
        self.target_names = target_names
        self.sample_weight = sample_weight
        self.digits = digits
        self.zero_division = zero_division
        self.threshold = threshold
        self.y_pred = []
        self.y_true = []

    def reset(self):
        self.y_pred.clear()
        self.y_true.clear()

    def update(self, y_pred, y_true):
        """
        Parameters
        ----------
        y_pred : 1d array-like, or label indicator array / sparse matrix
                Estimated targets as returned by a classifier.
        y_true : 1d array-like, or label indicator array / sparse matrix
                Ground truth (correct) target values.
        """
        return self(y_pred, y_true)

    def forward(self, y_pred, y_true):
        """
        Parameters
        ----------
        y_pred : 1d Tensor, or label indicator array / sparse matrix
                Estimated targets as returned by a classifier.
        y_true : 1d Tensor, or label indicator array / sparse matrix
                Ground truth (correct) target values.
        """
        self.y_pred.extend(y_pred.cpu().tolist())
        self.y_true.extend(y_true.cpu().tolist())

    def compute(self, output_dict=True):
        return classification_report(self.y_true, self.y_pred,
                                     labels=self.labels,
                                     target_names=self.target_names,
                                     sample_weight=self.sample_weight,
                                     digits=self.digits,
                                     output_dict=output_dict,
                                     zero_division=self.zero_division, )
