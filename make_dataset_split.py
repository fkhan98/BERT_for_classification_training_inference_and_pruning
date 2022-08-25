import numpy as np
import pandas as pd

df = pd.read_csv('./bbc-text.csv') 
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), 
                                     [int(.8*len(df)), int(.9*len(df))])

df_train.to_csv('./train.csv',index=False)
df_val.to_csv('./validation.csv',index=False)
df_test.to_csv('./test.csv',index=False)