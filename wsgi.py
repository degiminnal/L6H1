import numpy as np
from app.app import app
from sklearn.tree import DecisionTreeClassifier

class MyClassfier(DecisionTreeClassifier):
    def __init__(self):
        super().__init__()
        self.dic = dict()
        
    def encodeStr(self, x):
        x = x.copy()
        for i,col in enumerate(x.columns):
            if str(x[col].dtype)!="object":
                continue
            dic = {_:i for i,_ in enumerate(list(set(x[col])))}
            x[col] = np.array([dic[_] for _ in x[col]])
            self.dic[i] = dic
        return x
    
    def fit(self, x, y):
        x = self.encodeStr(x)
        return super().fit(x,y)
        
    def convert_(self,x):
        try:
            float(x)
            return float(x)
        except:
            return -1

    def predict(self, datas):
        datas = [[self.dic.get(i_,{x_:self.convert_(x_)}).get(x_, -1) for i_,x_ in enumerate(data)] for data in datas]
        return super().predict(datas)

if __name__ == "__main__":
	app.run()
