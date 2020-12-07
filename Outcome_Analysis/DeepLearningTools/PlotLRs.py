__author__ = 'Brian M Anderson'
# Created on 11/24/2020
from Deep_Learning.Base_Deeplearning_Code.Finding_Optimization_Parameters.LR_Finder import make_plot, os


def down_folder(input_path, base_input_path='',output_path=''):
    folders = []
    files = []
    for _, folders, files in os.walk(input_path):
        break
    iterations_found = False
    for folder in folders:
        if folder.find('Iteration') == -1:
            delete = down_folder(os.path.join(input_path, folder),base_input_path=base_input_path,output_path=output_path)
            if delete:
                os.removedirs(os.path.join(input_path,folder)) # Delete empty folders
        else:
            iterations_found = True
            break
    if not folders and not files:
        return True
    if iterations_found:
        paths = [os.path.join(input_path, i) for i in folders if i.find('Iteration') != -1]
        if '2_Iteration' not in folders:
            return False
        try:
            print(input_path)
            desc = ''.join(input_path.split(base_input_path)[-1].split('\\'))
            save_path = os.path.join(output_path,'Outputs')
            make_plot(paths, metric_list=['loss'], title=desc, save_path=save_path, plot=False,
                      auto_rates=True, beta=0.96)
        except:
            xxx = 1
    return False


def plot_lrs(input_path):
    down_folder(input_path,base_input_path=input_path, output_path=input_path)


if __name__ == '__main__':
    pass
