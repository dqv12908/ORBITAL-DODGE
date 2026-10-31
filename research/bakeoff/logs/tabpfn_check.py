import numpy as np, time
from sklearn.datasets import make_classification
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
X,y = make_classification(600, 20, n_informative=8, n_classes=3, random_state=0)
t=time.time()
m = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cuda:0")
print("model_path", m.model_path)
m.fit(X[:500], y[:500]); p = m.predict_proba(X[500:])
print("TABPFN_OK v2", p.shape, "acc", (p.argmax(1)==y[500:]).mean(), "secs", round(time.time()-t,1))
