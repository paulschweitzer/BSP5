import numpy as np
import pandas as pd
from pdfid import pdfid

import utils
from train import train_linear, train_rbf, train_mlp, retrieve_scaled_benign_samples, preprocess_data


COMMANDS = ["train", "evade", "quit"]
CLASSIFIERS = ["linear", "rbf", "mlp"]
PATH = 'PDFMalware2022.csv'

def main():
    
    weights_linear = None

    gamma = None
    weights_rbf = None
    supports_rbf = None

    weights_mlp = None
    bias_mlp = None
    
    scaler = None
    model = None
    pdf_vector = None
    current_classifier = None
    done = False
    while not done:

        inp = input("> ").split()
        command = inp[0]
        
        if command not in COMMANDS:
            print("Invalid command")
            continue

        if command == "train":
            print("Choose classifier")
            cl = input(">> ")
            if cl not in CLASSIFIERS:
                print("Invalid classifier")
                continue

            current_classifier = cl
            if cl == "linear":
                weights_linear, model, scaler = train_linear(PATH)
            elif cl == "rbf":
                gamma, weights_rbf, supports_rbf, model, scaler = train_rbf(PATH)
            elif cl == "mlp":
                weights_mlp, bias_mlp, model, scaler = train_mlp(PATH)

    
        elif command == "evade":

            print("Enter path of malicious PDF file.")
            path_pdf = input(">> ")
            paths = [path_pdf]
            
            options = pdfid.get_fake_options()
            options.scan = True
            options.json = True
            
            # Fetch features from PDF file
            out = pd.DataFrame([pdfid.PDFiDMain(paths, options)['reports'][0]])
            
            
            data = utils.rename_columns(out)
            data = out.drop(columns=['version', 'filename', 'header'])
            
            print("Sample to modify")
            for col, val in data.items():
                print(f"{col}: {val[0]}")
            print()

            x0 = data
            x0 = scaler.transform(x0)[0]
            x_benign_samples = retrieve_scaled_benign_samples(PATH)
            
            args = {}
            if current_classifier == "linear":
                args['gradient_linear'] = weights_linear

            elif current_classifier == 'rbf':
                args['gamma'] = gamma
                args['weights_rbf'] = weights_rbf
                args['support_vectors_rbf'] = supports_rbf

            elif current_classifier == 'mlp':
                args['output_weights'] = np.array(weights_mlp[-1])
                args['output_bias'] = np.array(bias_mlp[-1])
                args['hidden_weights'] = np.array(weights_mlp[0])
                args['hidden_bias'] = np.array(bias_mlp[0])
            
            modified_sample_scaled = utils.modify_sample(x0, args, model, x_benign_samples, current_classifier)
            modified_sample = scaler.inverse_transform([modified_sample_scaled])
            df_modified_sample = pd.DataFrame(modified_sample.astype(int), columns=data.columns)
            
            print("Modified sample: ")
            for col in df_modified_sample:
                print(f"{col}: {df_modified_sample[col][0]}")
            print()
            
            print("Delta: ")
            delta = df_modified_sample - data
            for col in delta:
                if delta[col][0] != 0:
                    print(f"{col}: {delta[col][0]}")
            print()


        elif command == "quit":
            done = True

        




if __name__ == "__main__":
    main()
