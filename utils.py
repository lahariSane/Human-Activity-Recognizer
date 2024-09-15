class RemoveDuplicateColumns(X):
    def fit(self, X,Y=None):
        self.duplicate_columns = X.columns[X.T.duplicated()].tolist()
        return self

    def transform(self, X, Y=None):
        return X.drop(columns=self.duplicate_columns)
