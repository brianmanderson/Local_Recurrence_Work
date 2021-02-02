__author__ = 'Brian M Anderson'
# Created on 11/24/2020
from Deep_Learning.Base_Deeplearning_Code.Finding_Optimization_Parameters.LR_Finder import make_plot, os
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnPaths import return_paths
import pandas as pd


def plot_lrs(input_path, excel_path=None, add_to_excel=False, base_df=None, save_path=None):
    remake_save_path = False
    if save_path is None:
        remake_save_path = True
    if add_to_excel:
        if excel_path is None:
            base_path, morfeus_drive = return_paths()
            excel_path = os.path.join(morfeus_drive, 'ModelParameters.xlsx')
        if base_df is None:
            base_df = pd.read_excel(excel_path)
    for root, folders, files in os.walk(input_path):
        paths = [os.path.join(root, i) for i in folders if i.find('Iteration') != -1]
        if paths:
            print(root)
            desc = os.path.split(root)[-1]
            if remake_save_path:
                save_path = os.path.join(input_path, 'Outputs')
            try:
                out_lr_dict = make_plot(paths, metric_list=['AUC', 'loss'], title=desc, save_path=save_path, plot=True,
                                        auto_rates=True, weight_smoothing=0.99, plot_show=False)
                if excel_path is not None:
                    model_index = int(os.path.split(paths[0])[0].split('Model_Index_')[-1])
                    index = base_df.loc[base_df.Model_Index == int(model_index)]
                    if index.shape[0] > 0:
                        index = index.index[0]
                        if pd.isnull(base_df.loc[index, 'min_lr']):
                            base_df = pd.read_excel(excel_path)
                            if pd.isnull(base_df.loc[index, 'min_lr']):
                                base_df.at[index, 'min_lr'] = out_lr_dict['loss']['min_lr']
                                base_df.at[index, 'max_lr'] = out_lr_dict['loss']['max_lr']
                                base_df.at[index, 'run?'] = 1
                                base_df.to_excel(excel_path, index=0)
            except:
                xxx = 1
    return True


if __name__ == '__main__':
    pass
