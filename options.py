import argparse

def parse():
    print('Parsing arguments')
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', default='NTU') 
    parser.add_argument('--protocol', default='CS')
    parser.add_argument('--data_dim_skl', default=150, type=int)
    parser.add_argument('--num_classes', default=60, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--n_neuron', default=512, type=int)
    parser.add_argument('--timesteps', default=20, type=int)
    parser.add_argument('--n_dropout', default=0.6, type=float)
    parser.add_argument('--training_mode', default='end')
    parser.add_argument('--attention_mode', default='sum')
    parser.add_argument('--sum_idx', default=0, type=int)
    parser.add_argument('--train_end_to_end', default=False, type=bool)
    parser.add_argument('--marker', default='ntu_reg_no1')
    parser.add_argument('--epochs', default=100, type=int) 

    args = parser.parse_args()
    return args


