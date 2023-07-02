import enum
from tqdm import tqdm

import pandas as pd

from dl_manager import *

class VotingTypes(enum.Enum):
    Hard = 'hard'
    Soft = 'soft'


class EnsembleModel:
    def __init__(self, voting_type, indexes):
        self.voting_type = voting_type
        self.indexes = indexes

    def predict(self, x_mat):
        x_mat = x_mat[:, self.indexes]
        p_lst = []
        if self.voting_type == VotingTypes.Soft:
            p_lst = np.mean(x_mat, axis=1)
        elif self.voting_type == VotingTypes.Hard:
            size = float(len(self.indexes))
            for i in range(x_mat.shape[0]):
                now_p = np.sum([1 if x >= 0.5 else 0 for x in x_mat[i]]) / size
                p_lst.append(now_p)
        return p_lst


class EnsembleManager:
    DS = 'dataset'
    TG = 'target'
    START_IDX = 3

    def __init__(self, input_path):
        df = pd.read_csv(input_path)
        self.df_train = df.loc[df[self.DS] == 0]
        self.df_val = df.loc[df[self.DS] == 1]
        self.df_test = df.loc[df[self.DS] == 2]
        self.columns = list(df.columns)[self.START_IDX:]

    def get_data(self, df):
        label_lst = df[self.TG].values
        value_mat = df.iloc[:, self.START_IDX:].values
        return value_mat, label_lst

    def get_score(self, model, x, y):
        p_lst = model.predict(x)
        metric = binary_evaluate(y, p_lst)
        return metric[CMtypes.F1.value]


    def train(self):
        x_train, y_train = self.get_data(self.df_train)
        x_val, y_val = self.get_data(self.df_val)
        x_test, y_test = self.get_data(self.df_test)
        columns = self.columns
        col_size = len(columns)

        prms_dict = {'voting_type':[VotingTypes.Hard, VotingTypes.Soft]}
        for c_idx in range(col_size):
            prms_dict[str(c_idx)] = [True, False]

        best_s, best_m = 0, None
        for prms in tqdm(list(ParameterGrid(prms_dict)), ascii=True, desc='Params Tuning:'):
            voting_type = prms['voting_type']

            indexes = []
            for c_idx in range(col_size):
                if prms[str(c_idx)]:
                    indexes.append(c_idx)
            if len(indexes) < 1:
                continue

            model = EnsembleModel(voting_type, indexes)
            s_train = self.get_score(model, x_train, y_train) 
            s_val = self.get_score(model, x_val, y_val) 
            s_test = self.get_score(model, x_test, y_test)

            if s_train >= s_val and s_val > best_s and abs(s_train - s_val) < 0.05:
                best_s = s_val
                best_m = model
                print('prms: %s' % prms)
                print('t:%.3f, v:%.3f, s:%.3f' % (s_train, s_val, s_test))

        return best_m

def batch_ensemble():
    args_lst = [
        ['./data/pre-op_popf_results.csv', 'pre_popf'],
        ['./data/pre-op_cr_results.csv', 'pre_cr_popf'],
        # ['./data/popf_risk_score.csv', './result/pre_popf.csv'],
        # ['./data/cr-popf_risk_score.csv', './result/pre_cr_popf.csv'],
        ['./data/post-op_popf_results.csv', 'post_popf'],
        ['./data/post-op_cr_results.csv', 'post_cr_popf']
    ]

    for input_path, tag in args_lst:
        run_ensemble(input_path, tag)

def run_ensemble(input_path, tag):
    #input_path = './data/pre-op_popf_results.csv'
    esm_manager = EnsembleManager(input_path)
    model = esm_manager.train()

    x_test, y_test = esm_manager.get_data(esm_manager.df_test)
    y_pred = model.predict(x_test)
    metric = binary_evaluate(y_test, y_pred)
    output_path = './result/ensemble/esm_result.csv'
    export_metric(output_path, metric, tag)

    output_path = './result/ensemble/%s.pkl' % tag
    save_obj(output_path, model)

def batch_check_esm():
    path_lst = [
        ['./data/pre-op_popf_results.csv',
         './data/pre_popf.pkl'],
        ['./data/pre-op_cr_results.csv',
         './data/pre_cr_popf.pkl'],
        ['./data/post-op_popf_results.csv',
         './data/post_popf.pkl'],
        ['./data/post-op_cr_results.csv',
         './data/post_cr_popf.pkl']
    ]
    for data_path, model_path in path_lst:

        esm_manager = EnsembleManager(data_path)
        model = load_obj(model_path)
        indexes = model.indexes
        n_only = os.path.basename(model_path).split('.')[0]
        print('%s model voting : %s' % (n_only, model.voting_type.value))
        print(np.array(esm_manager.columns)[indexes])
