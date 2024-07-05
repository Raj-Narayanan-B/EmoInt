from scipy.stats import pearsonr, spearmanr


class evaluate():
    def __init__(self) -> None:
        pass

    def eval(self, y_true, y_pred):
        pearsonr_ = pearsonr(y_true, y_pred)
        spearmanr_ = spearmanr(y_true, y_pred)
        return ({"Pearson": pearsonr_.statistic, "Spearman": spearmanr_.statistic})
