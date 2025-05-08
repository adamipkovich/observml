import pandas as pd
import os

class SPMFCore:
    command : str = ""

    def __init__(self, mode = 'TopKRules', **kwargs):
        
        path = os.getcwd
        assert os.path.isfile("spmf.jar"), f"spmf.jar not found in {path} directory"
        self.mode = mode
        self.params = kwargs 
        args = ""
        for key, value in self.params.items():
            args += f" {value}"
        self.command = f'java -jar spmf.jar run {self.mode} input.csv output.csv' + args # The command needs to be a string
  
    def run(self, X: pd.DataFrame):

        X.to_csv("input.csv", header=False, index=False)
        os.system(self.command)
        ar = pd.read_csv("output.csv", names=['C0'])

        ar['Rule'] = ''
        ar['Support'] = int(0)
        ar['Confidence'] = int(0)
        for i in range(0, len(ar)):
            s1 = ar.at[i, 'C0'].split(' #SUP: ', 2)
            ar.at[i, 'Rule'] = s1[0]
            s2 = s1[1].split(' #CONF: ')
            ar.at[i, 'Support'] = int(s2[0])
            if len(s2) > 1:
                ar.at[i, 'Confidence'] = round(float(s2[1]), 3)
            else:
                if "Confidence" in ar.columns:
                    ar = ar.drop(columns=['Confidence'])

        ar = ar.sort_values(by='Support', ascending=False)
        ar = ar.reset_index(drop=True)
        ar = ar.drop(columns=['C0'])

        # delete the input and output files
        os.remove("input.csv")
        os.remove("output.csv")

        return ar
    

if __name__ == "__main__":
    # Wrapper for all association rule mining algorithms in SPMF
    
    ## TopKRules
    X = pd.read_csv("./data/traces_csoft_oper.csv")
    tkr = SPMFCore(mode='TopKRules', k=10, minconf_ar=0.8)
    y = tkr.run(X) 

    print("Done!")


